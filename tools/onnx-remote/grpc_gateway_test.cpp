#include "grpc_gateway.h"

#include "remote_transport.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

using namespace onnx_remote;
using namespace onnx_remote::grpc_gateway;
using onnxsim::remote::v1::CapabilitiesRequest;
using onnxsim::remote::v1::CapabilitiesResponse;
using onnxsim::remote::v1::CompileRequest;
using onnxsim::remote::v1::CompileResponse;
using onnxsim::remote::v1::ExecuteRequest;
using onnxsim::remote::v1::ExecuteResponse;
using onnxsim::remote::v1::LoadArtifactRequest;
using onnxsim::remote::v1::LoadArtifactResponse;
using onnxsim::remote::v1::ModelInferRequest;
using onnxsim::remote::v1::ModelInferResponse;
using onnxsim::remote::v1::OnnxSimExecutor;
using onnxsim::remote::v1::PROFILING_DETAILED;

namespace {

constexpr uint16_t kWorkerPort = 39671;

void serve_requests() {
  const int listener = listen_tcp(kWorkerPort, 1);
  assert(listener >= 0);
  for (int i = 0; i < 6; ++i) {
    const int fd = accept_tcp(listener);
    assert(fd >= 0);

    Request request;
    Response response;
    std::string error;
    assert(receive_request(fd, request, error));
    response.request_id = request.request_id;
    if (request.op == "capabilities") {
      response.ok = true;
      response.artifact_id = "smoke-runner";
      response.manifest = "{\"ready\":true}";
    } else if (request.op == "identity" && request.inputs.size() == 1) {
      response.ok = true;
      response.outputs = request.inputs;
      response.profile.push_back(
          ProfileEvent{"worker_identity", "remote", 4, 9, "smoke"});
    } else if (request.op == "compile" && !request.model.empty()) {
      response.ok = true;
      response.artifact_id = "smoke-artifact";
      response.artifact = {9, 8, 7};
      response.manifest = "{\"target\":\"smoke\"}";
    } else if (request.op == "load_compiled" &&
               request.artifact_id == "smoke-artifact" &&
               request.artifact == std::vector<uint8_t>({9, 8, 7})) {
      response.ok = true;
      response.artifact_id = request.artifact_id;
    } else {
      response.error = "unexpected gateway smoke request: " + request.op;
    }
    assert(send_response(fd, response, error));
    close_socket(fd);
  }
  close_socket(listener);
}

}  // namespace

int main() {
  std::thread worker(serve_requests);

  Options options;
  options.worker_host = "127.0.0.1";
  options.worker_port = kWorkerPort;
  Service service(options);
  grpc::ServerBuilder builder;
  int grpc_port = 0;
  builder.AddListeningPort("127.0.0.1:0",
                          grpc::InsecureServerCredentials(), &grpc_port);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  assert(server && grpc_port > 0);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(grpc_port),
      grpc::InsecureChannelCredentials());
  std::unique_ptr<OnnxSimExecutor::Stub> stub = OnnxSimExecutor::NewStub(channel);

  grpc::ClientContext capability_context;
  CapabilitiesRequest capability_request;
  CapabilitiesResponse capability_response;
  grpc::Status status = stub->GetCapabilities(
      &capability_context, capability_request, &capability_response);
  assert(status.ok());
  assert(capability_response.ready());
  assert(capability_response.protocol() == "onnx-remote-v5");
  assert(capability_response.runner_id() == "smoke-runner");
  assert(capability_response.supported_ops_size() == 11);
  assert(capability_response.supported_ops(8) == "abs");
  assert(capability_response.supported_ops(9) == "neg");
  assert(capability_response.supported_ops(10) == "sqrt");

  ExecuteRequest invalid_request;
  invalid_request.set_op(std::string(kMaxOpBytes + 1, 'x'));
  ExecuteResponse invalid_response;
  grpc::ClientContext invalid_context;
  status = stub->Execute(&invalid_context, invalid_request, &invalid_response);
  assert(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  ExecuteRequest invalid_shape_request;
  invalid_shape_request.set_op("identity");
  onnxsim::remote::v1::Tensor* invalid_shape_input =
      invalid_shape_request.add_inputs();
  invalid_shape_input->set_dtype(1);
  invalid_shape_input->add_shape(2);
  invalid_shape_input->set_raw_data("\0\0\0\0", 4);
  ExecuteResponse invalid_shape_response;
  grpc::ClientContext invalid_shape_context;
  status = stub->Execute(&invalid_shape_context, invalid_shape_request,
                         &invalid_shape_response);
  assert(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  ExecuteRequest invalid_dtype_request;
  invalid_dtype_request.set_op("identity");
  onnxsim::remote::v1::Tensor* invalid_dtype_input =
      invalid_dtype_request.add_inputs();
  invalid_dtype_input->set_dtype(99);
  invalid_dtype_input->add_shape(1);
  invalid_dtype_input->set_raw_data("\0", 1);
  ExecuteResponse invalid_dtype_response;
  grpc::ClientContext invalid_dtype_context;
  status = stub->Execute(&invalid_dtype_context, invalid_dtype_request,
                         &invalid_dtype_response);
  assert(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  ExecuteRequest request;
  request.set_request_id(17);
  request.set_op("identity");
  request.set_profiling(PROFILING_DETAILED);
  onnxsim::remote::v1::Tensor* input = request.add_inputs();
  input->set_dtype(1);
  input->add_shape(2);
  const float values[] = {1.25f, -2.5f};
  input->set_raw_data(reinterpret_cast<const char*>(values), sizeof(values));

  grpc::ClientContext execute_context;
  ExecuteResponse response;
  status = stub->Execute(&execute_context, request, &response);
  assert(status.ok());
  assert(response.ok());
  assert(response.request_id() == request.request_id());
  assert(response.outputs_size() == 1);
  assert(response.outputs(0).raw_data().size() == sizeof(values));
  assert(std::memcmp(response.outputs(0).raw_data().data(), values,
                     sizeof(values)) == 0);
  assert(response.profile_size() == 1);
  assert(response.profile(0).name() == "worker_identity");

  ModelInferRequest infer_request;
  infer_request.set_id("infer-1");
  infer_request.set_model_name("identity");
  infer_request.set_model_version("smoke");
  onnxsim::remote::v1::Tensor* infer_input = infer_request.add_inputs();
  infer_input->set_dtype(1);
  infer_input->add_shape(2);
  infer_input->set_raw_data(reinterpret_cast<const char*>(values),
                            sizeof(values));
  ModelInferResponse infer_response;
  grpc::ClientContext infer_context;
  status = stub->ModelInfer(&infer_context, infer_request, &infer_response);
  assert(status.ok());
  assert(infer_response.ok());
  assert(infer_response.id() == "infer-1");
  assert(infer_response.model_name() == "identity");
  assert(infer_response.outputs_size() == 1);
  assert(std::memcmp(infer_response.outputs(0).raw_data().data(), values,
                     sizeof(values)) == 0);

  CompileRequest compile_request;
  compile_request.set_request_id(18);
  compile_request.set_model("serialized-model");
  CompileResponse compile_response;
  grpc::ClientContext compile_context;
  status = stub->Compile(&compile_context, compile_request, &compile_response);
  assert(status.ok());
  assert(compile_response.ok());
  assert(compile_response.artifact_id() == "smoke-artifact");
  assert(compile_response.artifact() == std::string("\x09\x08\x07", 3));

  LoadArtifactRequest load_request;
  load_request.set_request_id(19);
  load_request.set_artifact_id(compile_response.artifact_id());
  load_request.set_artifact(compile_response.artifact());
  LoadArtifactResponse load_response;
  grpc::ClientContext load_context;
  status = stub->LoadArtifact(&load_context, load_request, &load_response);
  assert(status.ok());
  assert(load_response.ok());
  assert(!load_response.already_present());

  LoadArtifactRequest repeated_load_request = load_request;
  repeated_load_request.set_request_id(20);
  LoadArtifactResponse repeated_load_response;
  grpc::ClientContext repeated_load_context;
  status = stub->LoadArtifact(&repeated_load_context, repeated_load_request,
                              &repeated_load_response);
  assert(status.ok());
  assert(repeated_load_response.ok());
  assert(repeated_load_response.already_present());

  server->Shutdown();
  worker.join();
  return 0;
}
