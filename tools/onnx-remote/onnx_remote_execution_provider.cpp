#include "onnx_remote_execution_provider.h"

#ifdef ONNXSIM_REMOTE_ORT_EP

#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "core/framework/compute_capability.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "profiler.h"

namespace onnxsim::ort_remote {
namespace {

using onnxruntime::common::Status;

std::string JsonEscape(const std::string &value) {
  std::string result;
  for (const char c : value) {
    if (c == '"' || c == '\\')
      result.push_back('\\');
    result.push_back(c);
  }
  return result;
}

onnx_remote::Tensor ToWire(const Ort::Value &value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("remote ORT EP currently supports float32 only");
  }
  onnx_remote::Tensor tensor;
  tensor.dtype = 1; // TensorProto.FLOAT
  tensor.shape = info.GetShape();
  const float *data = value.GetTensorData<float>();
  const size_t elements = info.GetElementCount();
  tensor.data.assign(data, data + elements);
  return tensor;
}

void CopyFromWire(const onnx_remote::Tensor &source, Ort::Value &destination) {
  if (source.dtype != 1) {
    throw std::runtime_error("remote ORT EP returned a non-float32 tensor");
  }
  auto info = destination.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      info.GetShape() != source.shape ||
      info.GetElementCount() != source.data.size()) {
    throw std::runtime_error(
        "remote ORT EP returned an incompatible output tensor");
  }
  float *data = destination.GetTensorMutableData<float>();
  std::copy(source.data.begin(), source.data.end(), data);
}

std::string RemoteOperation(const std::string &onnx_op) {
  if (onnx_op == "Identity")
    return "identity";
  if (onnx_op == "Relu")
    return "relu";
  if (onnx_op == "Add")
    return "add";
  if (onnx_op == "Mul")
    return "mul";
  return {};
}

void ParseInteger(const onnxruntime::ProviderOptions &options, const char *key,
                  int &value) {
  const auto it = options.find(key);
  if (it == options.end())
    return;
  try {
    value = std::stoi(it->second);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("invalid remote EP option ") + key);
  }
}

void ParsePort(const onnxruntime::ProviderOptions &options, const char *key,
               uint16_t &value) {
  const auto it = options.find(key);
  if (it == options.end())
    return;
  try {
    const unsigned long parsed = std::stoul(it->second);
    if (parsed > 65535)
      throw std::out_of_range("port");
    value = static_cast<uint16_t>(parsed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("invalid remote EP option ") + key);
  }
}

class RemoteExecutionProvider final : public onnxruntime::IExecutionProvider {
public:
  explicit RemoteExecutionProvider(Options options)
      : IExecutionProvider("OnnxSimRemoteExecutionProvider"),
        options_(std::move(options)) {}

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer &graph,
      const IKernelLookup & /*kernel_lookup*/,
      const onnxruntime::GraphOptimizerRegistry & /*graph_optimizer_registry*/,
      onnxruntime::IResourceAccountant * /*resource_accountant*/)
      const override {
    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>> result;
    for (const auto index : graph.GetNodesInTopologicalOrder()) {
      const auto *node = graph.GetNode(index);
      if (node == nullptr ||
          (!node->Domain().empty() && node->Domain() != "ai.onnx") ||
          options_.supported_ops.count(node->OpType()) == 0 ||
          !node->GetExecutionProviderType().empty()) {
        continue;
      }
      auto subgraph = std::make_unique<onnxruntime::IndexedSubGraph>();
      subgraph->nodes.push_back(node->Index());
      result.push_back(std::make_unique<onnxruntime::ComputeCapability>(
          std::move(subgraph)));
    }
    return result;
  }

  Status Compile(const std::vector<FusedNodeAndGraph> &fused,
                 std::vector<NodeComputeInfo> &funcs) override {
    for (const auto &item : fused) {
      const std::string op = item.fused_node.get().OpType();
      NodeComputeInfo info;
      info.create_state_func = [](ComputeContext *, FunctionState *) {
        return 0;
      };
      info.release_state_func = [](FunctionState) {};
      info.compute_func = [this, op](FunctionState, const OrtApi *,
                                     OrtKernelContext *raw) -> Status {
        try {
          Ort::KernelContext context(raw);
          onnx_remote::Request request;
          request.request_id = next_request_id_++;
          request.op = RemoteOperation(op);
          request.profiling = options_.profiling;
          const size_t input_count = context.GetInputCount();
          request.inputs.reserve(input_count);
          for (size_t i = 0; i < input_count; ++i)
            request.inputs.push_back(ToWire(context.GetInput(i)));

          const uint64_t start = onnxsim::Profiler::Instance().ElapsedMicros();
          const int fd = onnx_remote::connect_tcp_timeout(
              options_.host, options_.port, options_.connect_timeout_ms);
          if (fd < 0)
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "remote ORT EP connection failed");
          if (!onnx_remote::set_socket_io_timeout(fd, options_.io_timeout_ms)) {
            onnx_remote::close_socket(fd);
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "remote ORT EP socket timeout setup failed");
          }
          std::string error;
          onnx_remote::Response response;
          const bool sent = onnx_remote::send_request(fd, request, error);
          const bool received =
              sent && onnx_remote::receive_response(fd, response, error);
          onnx_remote::close_socket(fd);
          if (!received || !response.ok) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "remote ORT EP request failed: ",
                                   error.empty() ? response.error : error);
          }
          if (response.request_id != request.request_id) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "remote ORT EP request id mismatch");
          }
          const uint64_t end = onnxsim::Profiler::Instance().ElapsedMicros();
          auto &profiler = onnxsim::Profiler::Instance();
          if (profiler.enabled() &&
              options_.profiling != onnx_remote::ProfilingLevel::Off) {
            profiler.RecordExternalEvent("RemoteEP/" + op, "remote_ep", start,
                                         end >= start ? end - start : 0,
                                         "{\"op\":\"" + JsonEscape(op) + "\"}");
            for (const auto &event : response.profile) {
              profiler.RecordExternalEvent(
                  event.name, event.category, start + event.start_us,
                  event.duration_us,
                  "{\"op\":\"" + JsonEscape(op) + "\",\"detail\":\"" +
                      JsonEscape(event.detail) + "\"}");
            }
          }
          if (response.outputs.size() != context.GetOutputCount()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "remote ORT EP output count mismatch");
          }
          for (size_t i = 0; i < response.outputs.size(); ++i) {
            const auto shape = response.outputs[i].shape;
            auto output = context.GetOutput(i, shape.data(), shape.size());
            CopyFromWire(response.outputs[i], output);
          }
          return Status::OK();
        } catch (const std::exception &error) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "remote ORT EP: ", error.what());
        }
      };
      funcs.push_back(std::move(info));
    }
    return Status::OK();
  }

  onnxruntime::ProviderOptions GetProviderOptions() const override {
    onnxruntime::ProviderOptions options;
    options["host"] = options_.host;
    options["port"] = std::to_string(options_.port);
    options["connect_timeout_ms"] = std::to_string(options_.connect_timeout_ms);
    options["io_timeout_ms"] = std::to_string(options_.io_timeout_ms);
    options["profiling"] = std::to_string(static_cast<int>(options_.profiling));
    std::vector<std::string> supported_ops(options_.supported_ops.begin(),
                                           options_.supported_ops.end());
    std::sort(supported_ops.begin(), supported_ops.end());
    std::string ops;
    for (const auto &op : supported_ops) {
      if (!ops.empty())
        ops += ',';
      ops += op;
    }
    options["supported_ops"] = std::move(ops);
    return options;
  }

private:
  Options options_;
  mutable std::atomic<uint64_t> next_request_id_{1};
};

} // namespace

std::unique_ptr<onnxruntime::IExecutionProvider>
CreateRemoteExecutionProvider(const Options &options) {
  return std::make_unique<RemoteExecutionProvider>(options);
}

Options OptionsFromProviderOptions(
    const onnxruntime::ProviderOptions &provider_options) {
  Options options;
  if (const auto it = provider_options.find("host");
      it != provider_options.end()) {
    options.host = it->second;
  }
  ParsePort(provider_options, "port", options.port);
  ParseInteger(provider_options, "connect_timeout_ms",
               options.connect_timeout_ms);
  ParseInteger(provider_options, "io_timeout_ms", options.io_timeout_ms);
  if (const auto it = provider_options.find("profiling");
      it != provider_options.end()) {
    if (it->second == "summary" || it->second == "1") {
      options.profiling = onnx_remote::ProfilingLevel::Summary;
    } else if (it->second == "detailed" || it->second == "2") {
      options.profiling = onnx_remote::ProfilingLevel::Detailed;
    } else if (it->second != "off" && it->second != "0") {
      throw std::invalid_argument("invalid remote EP option profiling");
    }
  }
  if (const auto it = provider_options.find("supported_ops");
      it != provider_options.end()) {
    options.supported_ops.clear();
    std::stringstream stream(it->second);
    for (std::string op; std::getline(stream, op, ',');) {
      if (!op.empty())
        options.supported_ops.insert(std::move(op));
    }
  }
  return options;
}

std::unique_ptr<onnxruntime::IExecutionProvider>
CreateRemoteExecutionProviderFromOptions(
    const onnxruntime::ProviderOptions &provider_options) {
  return CreateRemoteExecutionProvider(
      OptionsFromProviderOptions(provider_options));
}

} // namespace onnxsim::ort_remote

#endif // ONNXSIM_REMOTE_ORT_EP
