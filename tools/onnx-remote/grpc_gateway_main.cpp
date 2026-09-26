#include "grpc_gateway.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <grpcpp/grpcpp.h>

namespace {

uint16_t port_value(const char* text, const char* name) {
  char* end = nullptr;
  unsigned long value = std::strtoul(text, &end, 10);
  if (!*text || !end || *end || value > 65535) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return static_cast<uint16_t>(value);
}

void usage(const char* program) {
  std::cerr << "usage: " << program
            << " [--listen HOST:PORT] [--worker-host HOST]"
               " [--worker-port PORT] [--runner-id ID]\n";
}

}  // namespace

int main(int argc, char** argv) {
  onnx_remote::grpc_gateway::Options options;
  std::string listen = "0.0.0.0:50051";
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--listen" && i + 1 < argc) {
        listen = argv[++i];
      } else if (argument == "--worker-host" && i + 1 < argc) {
        options.worker_host = argv[++i];
      } else if (argument == "--worker-port" && i + 1 < argc) {
        options.worker_port = port_value(argv[++i], "worker port");
      } else if (argument == "--runner-id" && i + 1 < argc) {
        options.runner_id = argv[++i];
      } else if (argument == "--help") {
        usage(argv[0]);
        return 0;
      } else {
        usage(argv[0]);
        return 2;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }

  onnx_remote::grpc_gateway::Service service(std::move(options));
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "failed to start gRPC gateway on " << listen << '\n';
    return 1;
  }
  std::cout << "onnxsim gRPC gateway listening on " << listen << '\n';
  server->Wait();
  return 0;
}
