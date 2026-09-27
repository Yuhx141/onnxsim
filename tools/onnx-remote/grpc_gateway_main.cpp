#include "grpc_gateway.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
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
               " [--worker-port PORT] [--compiler-host HOST]"
               " [--compiler-port PORT] [--runner-id ID]"
               " [--tls-cert FILE --tls-key FILE [--tls-ca FILE]]\n";
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
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
      } else if (argument == "--compiler-host" && i + 1 < argc) {
        options.compiler_host = argv[++i];
      } else if (argument == "--compiler-port" && i + 1 < argc) {
        options.compiler_port = port_value(argv[++i], "compiler port");
      } else if (argument == "--runner-id" && i + 1 < argc) {
        options.runner_id = argv[++i];
      } else if (argument == "--tls-cert" && i + 1 < argc) {
        options.tls_certificate = argv[++i];
      } else if (argument == "--tls-key" && i + 1 < argc) {
        options.tls_private_key = argv[++i];
      } else if (argument == "--tls-ca" && i + 1 < argc) {
        options.tls_client_ca = argv[++i];
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
  std::shared_ptr<grpc::ServerCredentials> credentials;
  if (service.options().tls_certificate.empty() &&
      service.options().tls_private_key.empty() &&
      service.options().tls_client_ca.empty()) {
    credentials = grpc::InsecureServerCredentials();
  } else {
    if (service.options().tls_certificate.empty() ||
        service.options().tls_private_key.empty()) {
      std::cerr << "--tls-cert and --tls-key must be supplied together\n";
      return 2;
    }
    try {
      grpc::SslServerCredentialsOptions tls;
      tls.pem_key_cert_pairs.push_back({
          read_file(service.options().tls_private_key),
          read_file(service.options().tls_certificate)});
      if (!service.options().tls_client_ca.empty()) {
        tls.pem_root_certs = read_file(service.options().tls_client_ca);
        tls.force_client_auth = true;
      }
      credentials = grpc::SslServerCredentials(tls);
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
  }
  builder.AddListeningPort(listen, credentials);
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
