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
using onnxsim::remote::v1::ExecuteRequest;
using onnxsim::remote::v1::ExecuteResponse;
using onnxsim::remote::v1::OnnxSimExecutor;
using onnxsim::remote::v1::PROFILING_DETAILED;

namespace {

constexpr uint16_t kWorkerPort = 39671;

void serve_one_request() {
  const int listener = listen_tcp(kWorkerPort, 1);
  assert(listener >= 0);
  const int fd = accept_tcp(listener);
  assert(fd >= 0);

  Request request;
  Response response;
  std::string error;
  assert(receive_request(fd, request, error));
  response.request_id = request.request_id;
  response.ok = request.op == "identity" && request.inputs.size() == 1;
  if (response.ok) response.outputs = request.inputs;
  else response.error = "unexpected gateway smoke request";
  assert(send_response(fd, response, error));
  close_socket(fd);
  close_socket(listener);
}

}  // namespace

int main() {
  std::thread worker(serve_one_request);

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

  server->Shutdown();
  worker.join();
  return 0;
}
