#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
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
  std::vector<std::string> supported_ops = {"identity", "relu", "add", "mul",
                                            "sub", "div", "max", "min",
                                            "abs", "neg", "sqrt"};
  std::string tls_certificate;
  std::string tls_private_key;
  std::string tls_client_ca;
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
  grpc::Status ModelInfer(
      grpc::ServerContext* context,
      const onnxsim::remote::v1::ModelInferRequest* request,
      onnxsim::remote::v1::ModelInferResponse* response) override;

  const Options& options() const { return options_; }

 private:
  Options options_;
  // This is deliberately gateway-local knowledge: the native v5 worker does
  // not expose a cache-hit bit, so the gateway reports whether it has
  // successfully uploaded the same artifact ID and bytes before.
  std::mutex artifact_cache_mu_;
  std::unordered_map<std::string, std::string> loaded_artifacts_;
};

}  // namespace onnx_remote::grpc_gateway
