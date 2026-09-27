#include "remote_executor.h"

#ifdef ONNXSIM_BUILTIN_REMOTE_EXECUTOR

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "dlpack_bridge.h"
#include "profiler.h"
#include "remote_transport.h"

namespace {

std::atomic<uint64_t> next_request_id{1};

struct OutputOwner {
  std::vector<uint8_t> data;
  std::vector<int64_t> shape;
  DLDataType dtype{};
};

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char c : value) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(hex[(static_cast<unsigned char>(c) >> 4) & 0xf]);
          escaped.push_back(hex[static_cast<unsigned char>(c) & 0xf]);
        } else {
          escaped += c;
        }
        break;
    }
  }
  return escaped;
}

DLManagedTensor* WrapOutput(OutputOwner* owner) {
  auto* result = new DLManagedTensor{};
  result->dl_tensor.data = owner->data.data();
  result->dl_tensor.device = DLDevice{kDLCPU, 0};
  result->dl_tensor.ndim = static_cast<int32_t>(owner->shape.size());
  result->dl_tensor.dtype = owner->dtype;
  result->dl_tensor.shape = owner->shape.data();
  result->dl_tensor.strides = nullptr;
  result->dl_tensor.byte_offset = 0;
  result->manager_ctx = owner;
  result->deleter = [](DLManagedTensor* tensor) {
    delete static_cast<OutputOwner*>(tensor->manager_ctx);
    delete tensor;
  };
  return result;
}

class RemoteModelExecutor final : public ModelExecutor {
 public:
  explicit RemoteModelExecutor(RemoteExecutorOptions options)
      : options_(std::move(options)) {}

  std::vector<DLManagedTensorPtr> Run(
      const onnx::ModelProto& model,
      const std::vector<const DLManagedTensor*>& inputs) const override {
    onnx::ModelProto prepared = model;
    const std::string target = options_.target;
    std::string legalize_error;
    if (options_.legalizer &&
        !options_.legalizer(prepared, target, legalize_error)) {
      throw std::runtime_error("remote executor legalization: " +
                               legalize_error);
    }
    if (!options_.supported_ops.empty()) {
      for (const auto& node : prepared.graph().node()) {
        if (std::find(options_.supported_ops.begin(),
                      options_.supported_ops.end(),
                      node.op_type()) == options_.supported_ops.end()) {
          throw std::runtime_error(
              "remote executor unsupported op after "
              "legalization: " +
              node.op_type());
        }
      }
    }
    const std::string serialized = prepared.SerializeAsString();
    onnx_remote::Request request;
    request.request_id =
        next_request_id.fetch_add(1, std::memory_order_relaxed);
    request.profiling = options_.profiling;
    if (options_.compile_model) {
      if (!options_.send_compiled_artifact &&
          !options_.attach_compiled_artifact) {
        throw std::runtime_error(
            "remote executor compiled mode requires either "
            "send_compiled_artifact or attach_compiled_artifact");
      }
      const auto artifact = GetOrCompile(serialized);
      request.op = options_.compiled_operation;
      request.artifact_id = artifact->id;
      if (options_.send_compiled_artifact) {
        request.artifact = artifact->bytes;
      }
    } else {
      request.op = options_.operation;
      request.model.assign(serialized.begin(), serialized.end());
    }
    request.inputs.reserve(inputs.size());
    for (const DLManagedTensor* input : inputs) {
      if (input == nullptr || input->dl_tensor.ndim < 0 ||
          (input->dl_tensor.ndim > 0 && input->dl_tensor.shape == nullptr)) {
        throw std::runtime_error("remote executor received invalid input tensor");
      }
      const DLTensor& tensor = input->dl_tensor;
      int32_t onnx_dtype = 0;
      if (tensor.device.device_type != kDLCPU || tensor.dtype.lanes != 1 ||
          tensor.strides != nullptr ||
          !onnxsim::dlpack::TryDLToOnnx(tensor.dtype, &onnx_dtype)) {
        throw std::runtime_error(
            "remote executor supports only contiguous CPU ONNX-compatible "
            "tensors");
      }
      onnx_remote::Tensor wire;
      wire.dtype = static_cast<uint8_t>(onnx_dtype);
      if (tensor.ndim > 0)
        wire.shape.assign(tensor.shape, tensor.shape + tensor.ndim);
      size_t element_count = 1;
      for (int32_t i = 0; i < tensor.ndim; ++i) {
        if (tensor.shape[i] <= 0 ||
            static_cast<uint64_t>(tensor.shape[i]) >
                std::numeric_limits<size_t>::max() / element_count) {
          throw std::runtime_error("remote executor received invalid input shape");
        }
        element_count *= static_cast<size_t>(tensor.shape[i]);
      }
      const size_t element_bytes = onnxsim::dlpack::SizeOf(tensor.dtype);
      if (element_count > std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::runtime_error("remote executor received oversized input tensor");
      }
      const size_t nbytes = element_count * element_bytes;
      if (nbytes != 0 && tensor.data == nullptr) {
        throw std::runtime_error("remote executor received null input data");
      }
      const auto* begin =
          static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
      if (onnx_dtype == onnx::TensorProto::FLOAT) {
        const auto* floats = reinterpret_cast<const float*>(begin);
        wire.data.assign(floats, floats + nbytes / sizeof(float));
      } else {
        wire.raw_data.assign(begin, begin + nbytes);
      }
      request.inputs.emplace_back(std::move(wire));
    }

    const onnx_remote::Response response =
        Exchange(request, options_.host, options_.port, "execute");
    std::vector<DLManagedTensorPtr> outputs;
    outputs.reserve(response.outputs.size());
    for (auto& output : response.outputs) {
      DLDataType dtype{};
      if (!onnxsim::dlpack::TryOnnxToDL(output.dtype, &dtype)) {
        throw std::runtime_error("remote executor returned unsupported dtype " +
                                 std::to_string(output.dtype));
      }
      const size_t element_bytes = onnxsim::dlpack::SizeOf(dtype);
      size_t element_count = 1;
      for (const int64_t dimension : output.shape) {
        if (dimension <= 0 ||
            static_cast<uint64_t>(dimension) >
                std::numeric_limits<size_t>::max() / element_count) {
          throw std::runtime_error(
              "remote executor returned an invalid output shape");
        }
        element_count *= static_cast<size_t>(dimension);
      }
      if (element_count > std::numeric_limits<size_t>::max() / element_bytes) {
        throw std::runtime_error(
            "remote executor returned an oversized output shape");
      }
      if (output.dtype == onnx::TensorProto::FLOAT &&
          output.data.size() > std::numeric_limits<size_t>::max() /
                                   sizeof(float)) {
        throw std::runtime_error(
            "remote executor returned an oversized float output");
      }
      const size_t payload_bytes = output.dtype == onnx::TensorProto::FLOAT
                                       ? output.data.size() * sizeof(float)
                                       : output.raw_data.size();
      if (payload_bytes != element_count * element_bytes) {
        throw std::runtime_error(
            "remote executor returned output shape/payload mismatch");
      }
      auto owner = std::make_unique<OutputOwner>();
      owner->dtype = dtype;
      if (output.dtype == onnx::TensorProto::FLOAT) {
        owner->data.resize(output.data.size() * sizeof(float));
        std::memcpy(owner->data.data(), output.data.data(), owner->data.size());
      } else {
        owner->data = std::move(output.raw_data);
      }
      owner->shape = std::move(output.shape);
      outputs.emplace_back(WrapOutput(owner.release()));
    }
    return outputs;
  }

 private:
  struct CompiledArtifact {
    std::string id;
    std::vector<uint8_t> bytes;
    std::string manifest;
  };

  onnx_remote::Response Exchange(const onnx_remote::Request& request,
                                 const std::string& host, uint16_t port,
                                 const char* phase) const {
    onnx_remote::Request wire_request = request;
    if (wire_request.request_id == 0) {
      wire_request.request_id =
          next_request_id.fetch_add(1, std::memory_order_relaxed);
    }
    auto& profiler = onnxsim::Profiler::Instance();
    const bool collect_profile =
        profiler.enabled() &&
        options_.profiling != onnx_remote::ProfilingLevel::Off;
    const uint64_t profile_anchor =
        collect_profile ? profiler.ElapsedMicros() : 0;
    const int fd = onnx_remote::connect_tcp_timeout(
        host, port, options_.connect_timeout_ms);
    if (fd < 0) throw std::runtime_error("remote executor: connection failed");
    if (!onnx_remote::set_socket_io_timeout(fd, options_.io_timeout_ms)) {
      onnx_remote::close_socket(fd);
      throw std::runtime_error("remote executor: cannot set socket timeout");
    }
    std::string error;
    onnx_remote::Response response;
    const uint64_t rpc_start = collect_profile ? profiler.ElapsedMicros() : 0;
    const bool sent = onnx_remote::send_request(fd, wire_request, error);
    const bool received =
        sent && onnx_remote::receive_response(fd, response, error);
    const uint64_t rpc_end = collect_profile ? profiler.ElapsedMicros() : 0;
    onnx_remote::close_socket(fd);
    if (collect_profile) {
      profiler.RecordExternalEvent(
          std::string("RemoteRPC/") + phase, "remote_transport", rpc_start,
          rpc_end >= rpc_start ? rpc_end - rpc_start : 0,
          "{\"host\":\"" + JsonEscape(host) +
              "\",\"port\":" + std::to_string(port) + ",\"phase\":\"" +
              JsonEscape(phase) + "\"}");
      for (const auto& event : response.profile) {
        profiler.RecordExternalEvent(
            event.name, event.category, profile_anchor + event.start_us,
            event.duration_us,
            "{\"detail\":\"" + JsonEscape(event.detail) + "\",\"phase\":\"" +
                JsonEscape(phase) + "\",\"remote_start_us\":" +
                std::to_string(event.start_us) + "}");
      }
    }
    if (!received || !response.ok) {
      throw std::runtime_error("remote executor: " +
                               (error.empty() ? response.error : error));
    }
    if (response.request_id != wire_request.request_id) {
      throw std::runtime_error("remote executor: response request id mismatch");
    }
    return response;
  }

  std::shared_ptr<const CompiledArtifact> GetOrCompile(
      const std::string& serialized) const {
    std::unique_lock<std::mutex> cache_lock(cache_mu_, std::defer_lock);
    if (options_.cache_compiled_models) {
      // Keep the per-executor cache lock through compilation and optional
      // attachment. This is intentionally a simple single-flight policy:
      // concurrent folds for the same model cannot duplicate an expensive
      // external compiler invocation or race a runner-side load.
      cache_lock.lock();
      const auto it = compiled_cache_.find(serialized);
      if (it != compiled_cache_.end()) return it->second;
    }

    onnx_remote::Request request;
    request.op = options_.compile_operation;
    request.model.assign(serialized.begin(), serialized.end());
    request.profiling = options_.profiling;
    const std::string compile_host =
        options_.compile_host.empty() ? options_.host : options_.compile_host;
    const uint16_t compile_port =
        options_.compile_port == 0 ? options_.port : options_.compile_port;
    const onnx_remote::Response response =
        Exchange(request, compile_host, compile_port, "compile");
    std::string manifest_error;
    if (options_.manifest_validator &&
        !options_.manifest_validator(response.manifest, manifest_error)) {
      throw std::runtime_error("remote executor manifest validation: " +
                               manifest_error);
    }
    if (response.artifact_id.empty() && response.artifact.empty()) {
      throw std::runtime_error(
          "remote compiler returned neither artifact_id nor artifact bytes");
    }
    auto artifact = std::make_shared<CompiledArtifact>();
    artifact->id = response.artifact_id;
    artifact->bytes = response.artifact;
    artifact->manifest = response.manifest;
    if (artifact->id.empty()) {
      // An inline-only compiler can still be used by a stateless runner. This
      // ID is process-local and is not a device cache key.
      artifact->id =
          "inline:" + std::to_string(std::hash<std::string>{}(serialized));
    }
    if (auto& profiler = onnxsim::Profiler::Instance();
        profiler.enabled() &&
        options_.profiling != onnx_remote::ProfilingLevel::Off) {
      profiler.RecordExternalEvent(
          "RemoteArtifactReady", "remote_compile", profiler.ElapsedMicros(), 0,
          "{\"artifact_id\":\"" + JsonEscape(artifact->id) + "\",\"bytes\":" +
              std::to_string(artifact->bytes.size()) + ",\"manifest_bytes\":" +
              std::to_string(artifact->manifest.size()) + "}");
    }
    if (options_.attach_compiled_artifact) {
      onnx_remote::Request load;
      load.op = options_.load_compiled_operation;
      load.artifact_id = artifact->id;
      load.artifact = artifact->bytes;
      load.profiling = options_.profiling;
      const std::string runner_host = options_.host;
      const uint16_t runner_port = options_.port;
      Exchange(load, runner_host, runner_port, "load");
      if (auto& profiler = onnxsim::Profiler::Instance();
          profiler.enabled() &&
          options_.profiling != onnx_remote::ProfilingLevel::Off) {
        profiler.RecordExternalEvent(
            "RemoteArtifactAttached", "remote_compile",
            profiler.ElapsedMicros(), 0,
            "{\"artifact_id\":\"" + JsonEscape(artifact->id) +
                "\",\"bytes\":" + std::to_string(artifact->bytes.size()) + "}");
      }
      artifact->bytes.clear();
    }
    // Publish only after an optional attach succeeds. If the runner rejects
    // the artifact, a later retry must compile/attach again rather than
    // reusing an artifact that onnxsim believes is resident remotely.
    if (options_.cache_compiled_models) {
      compiled_cache_[serialized] = artifact;
    }
    return artifact;
  }

  RemoteExecutorOptions options_;
  mutable std::mutex cache_mu_;
  mutable std::unordered_map<std::string,
                             std::shared_ptr<const CompiledArtifact>>
      compiled_cache_;
};

}  // namespace

std::shared_ptr<const ModelExecutor> GetRemoteModelExecutor(
    const RemoteExecutorOptions& options) {
  return std::make_shared<RemoteModelExecutor>(options);
}

#endif  // ONNXSIM_BUILTIN_REMOTE_EXECUTOR
