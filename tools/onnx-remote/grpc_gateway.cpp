#include "grpc_gateway.h"

#include "remote_transport.h"

#include <cstring>
#include <limits>
#include <utility>

namespace onnx_remote::grpc_gateway {
namespace {

using ProtoTensor = onnxsim::remote::v1::Tensor;
using ProtoProfile = onnxsim::remote::v1::ProfileEvent;
using ExecuteRequest = onnxsim::remote::v1::ExecuteRequest;
using ExecuteResponse = onnxsim::remote::v1::ExecuteResponse;
using CompileRequest = onnxsim::remote::v1::CompileRequest;
using CompileResponse = onnxsim::remote::v1::CompileResponse;
using LoadArtifactRequest = onnxsim::remote::v1::LoadArtifactRequest;
using LoadArtifactResponse = onnxsim::remote::v1::LoadArtifactResponse;
using CapabilitiesRequest = onnxsim::remote::v1::CapabilitiesRequest;
using CapabilitiesResponse = onnxsim::remote::v1::CapabilitiesResponse;
using NativeRequest = onnx_remote::Request;
using NativeResponse = onnx_remote::Response;

grpc::Status invalid(const std::string& message) {
  return {grpc::StatusCode::INVALID_ARGUMENT, message};
}

grpc::Status unavailable(const std::string& message) {
  return {grpc::StatusCode::UNAVAILABLE, message};
}

size_t element_bytes(uint32_t dtype) {
  switch (dtype) {
    case 1:  // FLOAT
    case 6:  // INT32
    case 12: // UINT32
      return 4;
    case 2:  // UINT8
    case 3:  // INT8
    case 9:  // BOOL
      return 1;
    case 4:  // UINT16
    case 5:  // INT16
    case 10: // FLOAT16
      return 2;
    case 7:  // INT64
    case 11: // DOUBLE
    case 13: // UINT64
      return 8;
    case 16: // BFLOAT16
      return 2;
    default:
      return 0;
  }
}

ProfilingLevel profiling_level(
    onnxsim::remote::v1::ProfilingLevel level) {
  switch (level) {
    case onnxsim::remote::v1::PROFILING_SUMMARY:
      return ProfilingLevel::Summary;
    case onnxsim::remote::v1::PROFILING_DETAILED:
      return ProfilingLevel::Detailed;
    case onnxsim::remote::v1::PROFILING_OFF:
    default:
      return ProfilingLevel::Off;
  }
}

bool copy_input(const ProtoTensor& source, Tensor& target, std::string& error) {
  if (source.shape_size() > static_cast<int>(kMaxRank)) {
    error = "tensor rank exceeds limit";
    return false;
  }
  const size_t bytes_per_element = element_bytes(source.dtype());
  if (bytes_per_element == 0) {
    error = "unsupported tensor dtype";
    return false;
  }
  uint64_t element_count = 1;
  for (const int64_t dimension : source.shape()) {
    if (dimension <= 0 ||
        element_count > kMaxTensorBytes / static_cast<uint64_t>(dimension)) {
      error = "invalid tensor shape";
      return false;
    }
    element_count *= static_cast<uint64_t>(dimension);
  }
  if (element_count > kMaxTensorBytes / bytes_per_element) {
    error = "tensor exceeds transport limit";
    return false;
  }
  target.dtype = static_cast<uint8_t>(source.dtype());
  target.shape.assign(source.shape().begin(), source.shape().end());
  const std::string& bytes = source.raw_data();
  if (bytes.size() != element_count * bytes_per_element) {
    error = "tensor payload does not match shape and dtype";
    return false;
  }
  if (target.dtype == 1) {
    target.data.resize(bytes.size() / sizeof(float));
    if (!bytes.empty()) std::memcpy(target.data.data(), bytes.data(), bytes.size());
  } else {
    target.raw_data.assign(bytes.begin(), bytes.end());
  }
  return true;
}

void copy_output(const Tensor& source, ProtoTensor* target) {
  target->set_dtype(source.dtype);
  for (int64_t dimension : source.shape) target->add_shape(dimension);
  if (source.dtype == 1) {
    target->set_raw_data(reinterpret_cast<const char*>(source.data.data()),
                         source.data.size() * sizeof(float));
  } else {
    target->set_raw_data(reinterpret_cast<const char*>(source.raw_data.data()),
                         source.raw_data.size());
  }
}

void copy_profile(const ProfileEvent& source, ProtoProfile* target) {
  target->set_name(source.name);
  target->set_category(source.category);
  target->set_start_us(source.start_us);
  target->set_duration_us(source.duration_us);
  target->set_detail(source.detail);
}

grpc::Status roundtrip(const Options& options, NativeRequest request,
                       NativeResponse& response) {
  std::string error;
  int fd = connect_tcp_timeout(options.worker_host, options.worker_port,
                               options.connect_timeout_ms);
  if (fd < 0) return unavailable("cannot connect to native worker");
  if (!set_socket_io_timeout(fd, options.io_timeout_ms) ||
      !send_request(fd, request, error) || !receive_response(fd, response, error)) {
    close_socket(fd);
    return unavailable(error.empty() ? "native worker I/O failed" : error);
  }
  close_socket(fd);
  return grpc::Status::OK;
}

template <typename Response>
void set_common_response(uint64_t request_id, const NativeResponse& native,
                         Response* response) {
  response->set_request_id(request_id);
  response->set_ok(native.ok);
  if (!native.ok) response->set_error(native.error);
}

}  // namespace

Service::Service(Options options) : options_(std::move(options)) {}

grpc::Status Service::Execute(grpc::ServerContext*, const ExecuteRequest* request,
                              ExecuteResponse* response) {
  if (request->op().empty() || request->op().size() > kMaxOpBytes)
    return invalid("operation is empty or exceeds transport limit");
  if (request->artifact_id().size() > kMaxArtifactIdBytes)
    return invalid("artifact_id exceeds transport limit");
  if (request->inputs_size() > static_cast<int>(kMaxTensors))
    return invalid("too many input tensors");
  NativeRequest native;
  native.request_id = request->request_id();
  native.op = request->op();
  native.artifact_id = request->artifact_id();
  native.profiling = profiling_level(request->profiling());
  native.inputs.reserve(request->inputs_size());
  for (const auto& input : request->inputs()) {
    Tensor tensor;
    std::string error;
    if (!copy_input(input, tensor, error)) return invalid(error);
    native.inputs.push_back(std::move(tensor));
  }
  NativeResponse native_response;
  grpc::Status status = roundtrip(options_, std::move(native), native_response);
  if (!status.ok()) return status;
  set_common_response(request->request_id(), native_response, response);
  for (const auto& output : native_response.outputs) copy_output(output, response->add_outputs());
  for (const auto& event : native_response.profile) copy_profile(event, response->add_profile());
  return grpc::Status::OK;
}

grpc::Status Service::Compile(grpc::ServerContext*, const CompileRequest* request,
                              CompileResponse* response) {
  if (request->model().size() > kMaxArtifactBytes)
    return invalid("model exceeds transport limit");
  NativeRequest native;
  native.request_id = request->request_id();
  native.op = "compile";
  native.model.assign(request->model().begin(), request->model().end());
  native.profiling = profiling_level(request->profiling());
  NativeResponse native_response;
  grpc::Status status = roundtrip(options_, std::move(native), native_response);
  if (!status.ok()) return status;
  set_common_response(request->request_id(), native_response, response);
  response->set_artifact_id(native_response.artifact_id);
  response->set_artifact(reinterpret_cast<const char*>(native_response.artifact.data()),
                         native_response.artifact.size());
  response->set_manifest(native_response.manifest);
  for (const auto& event : native_response.profile) copy_profile(event, response->add_profile());
  return grpc::Status::OK;
}

grpc::Status Service::LoadArtifact(grpc::ServerContext*, const LoadArtifactRequest* request,
                                   LoadArtifactResponse* response) {
  if (request->artifact_id().empty()) return invalid("artifact_id is required");
  if (request->artifact_id().size() > kMaxArtifactIdBytes)
    return invalid("artifact_id exceeds transport limit");
  if (request->artifact().size() > static_cast<int64_t>(kMaxArtifactBytes))
    return invalid("artifact exceeds transport limit");
  NativeRequest native;
  native.request_id = request->request_id();
  native.op = "load_compiled";
  native.artifact_id = request->artifact_id();
  native.artifact.assign(request->artifact().begin(), request->artifact().end());
  NativeResponse native_response;
  grpc::Status status = roundtrip(options_, std::move(native), native_response);
  if (!status.ok()) return status;
  set_common_response(request->request_id(), native_response, response);
  // The native v5 load operation is an upload/replace operation and does not
  // expose a cache-hit bit.  Report a successful upload conservatively; a
  // future worker capability can add an explicit hit field without changing
  // this RPC shape.
  response->set_already_present(false);
  return grpc::Status::OK;
}

grpc::Status Service::GetCapabilities(grpc::ServerContext*, const CapabilitiesRequest* request,
                                      CapabilitiesResponse* response) {
  NativeRequest native;
  native.request_id = request->request_id();
  native.op = "capabilities";
  NativeResponse native_response;
  grpc::Status status = roundtrip(options_, std::move(native), native_response);
  if (!status.ok()) return status;
  response->set_request_id(request->request_id());
  response->set_ready(native_response.ok);
  response->set_runner_id(native_response.artifact_id.empty()
                              ? options_.runner_id
                              : native_response.artifact_id);
  response->set_protocol("onnx-remote-v5");
  response->set_max_message_bytes(kMaxMessageBytes);
  response->set_profiling(native_response.ok);
  for (const auto& op : options_.supported_ops) response->add_supported_ops(op);
  for (const char* dtype : {"FLOAT", "UINT8", "INT8", "UINT16", "INT16",
                            "INT32", "INT64", "BOOL", "FLOAT16", "DOUBLE",
                            "UINT32", "UINT64", "BFLOAT16"}) {
    response->add_supported_dtypes(dtype);
  }
  return grpc::Status::OK;
}

}  // namespace onnx_remote::grpc_gateway
