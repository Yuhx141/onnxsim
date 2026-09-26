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
  target.dtype = static_cast<uint8_t>(source.dtype());
  target.shape.assign(source.shape().begin(), source.shape().end());
  const std::string& bytes = source.raw_data();
  if (target.dtype == 1) {
    if (bytes.size() % sizeof(float) != 0) {
      error = "float32 tensor payload is not aligned";
      return false;
    }
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
  response->set_request_id(request->request_id());
  response->set_ready(true);
  response->set_runner_id(options_.runner_id);
  response->set_protocol("onnx-remote-v5");
  response->set_max_message_bytes(kMaxMessageBytes);
  response->set_profiling(true);
  for (const auto& op : options_.supported_ops) response->add_supported_ops(op);
  for (const char* dtype : {"FLOAT", "FLOAT16", "BFLOAT16", "INT8", "UINT8",
                            "INT16", "UINT16", "INT32", "UINT32", "INT64",
                            "UINT64", "DOUBLE", "BOOL"}) {
    response->add_supported_dtypes(dtype);
  }
  return grpc::Status::OK;
}

}  // namespace onnx_remote::grpc_gateway
