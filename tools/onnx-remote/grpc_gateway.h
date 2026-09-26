#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "onnxsim_remote.grpc.pb.h"

namespace onnx_remote::grpc_gateway {

struct Options {
  std::string worker_host = "127.0.0.1";
  uint16_t worker_port = 39501;
  int connect_timeout_ms = 5000;
  int io_timeout_ms = 30000;
  std::string runner_id = "onnxsim-grpc-gateway";
  std::vector<std::string> supported_ops = {"identity", "relu", "add", "mul"};
};

class Service final : public onnxsim::remote::v1::OnnxSimExecutor::Service {
 public:
  explicit Service(Options options);

  grpc::Status Execute(
      grpc::ServerContext* context,
      const onnxsim::remote::v1::ExecuteRequest* request,
      onnxsim::remote::v1::ExecuteResponse* response) override;
  grpc::Status Compile(
      grpc::ServerContext* context,
      const onnxsim::remote::v1::CompileRequest* request,
      onnxsim::remote::v1::CompileResponse* response) override;
  grpc::Status LoadArtifact(
      grpc::ServerContext* context,
      const onnxsim::remote::v1::LoadArtifactRequest* request,
      onnxsim::remote::v1::LoadArtifactResponse* response) override;
  grpc::Status GetCapabilities(
      grpc::ServerContext* context,
      const onnxsim::remote::v1::CapabilitiesRequest* request,
      onnxsim::remote::v1::CapabilitiesResponse* response) override;

 private:
  Options options_;
};

}  // namespace onnx_remote::grpc_gateway
