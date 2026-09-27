#include "remote_transport.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

using namespace onnx_remote;
namespace fs = std::filesystem;

static std::unordered_map<std::string, std::vector<uint8_t>> artifacts;
static std::mutex artifacts_mutex;
static fs::path cache_dir;

static bool safe_artifact_id(const std::string& id) {
  if (id.empty()) return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '.' || c == '_' || c == '-';
  });
}

static fs::path cache_path(const std::string& id) {
  return cache_dir / (id + ".artifact");
}

static bool persist_artifact(const std::string& id,
                             const std::vector<uint8_t>& bytes) {
  if (cache_dir.empty()) return true;
  if (!safe_artifact_id(id)) return false;
  std::error_code ec;
  fs::create_directories(cache_dir, ec);
  if (ec) return false;
  const fs::path temporary = cache_path(id).string() + ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  if (!bytes.empty())
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  output.close();
  if (!output) return false;
  fs::rename(temporary, cache_path(id), ec);
  if (ec) fs::remove(temporary, ec);
  return !ec;
}

static bool load_persisted_artifact(const std::string& id,
                                    std::vector<uint8_t>& bytes) {
  if (cache_dir.empty() || !safe_artifact_id(id)) return false;
  std::ifstream input(cache_path(id), std::ios::binary | std::ios::ate);
  if (!input) return false;
  const auto end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) > kMaxArtifactBytes) return false;
  bytes.resize(static_cast<size_t>(end));
  input.seekg(0);
  if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), end);
  return static_cast<bool>(input) || bytes.empty();
}

static Response execute(const Request& request) {
  Response response;
  response.request_id = request.request_id;
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "mock-runner";
    response.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        "\"runner_id\":\"mock-runner\",\"ready\":true,"
        "\"graph_execution\":false,"
        "\"supported_ops\":[\"load_compiled\",\"run_compiled\"],"
        "\"supported_dtypes\":[\"FLOAT\"],"
        "\"profiling\":true}";
    return response;
  }
  if (request.op == "load_compiled") {
    if (request.artifact_id.empty() || request.artifact.empty()) {
      response.error = "load_compiled requires artifact_id and artifact";
      return response;
    }
    std::lock_guard<std::mutex> lock(artifacts_mutex);
    artifacts[request.artifact_id] = request.artifact;
    if (!persist_artifact(request.artifact_id, request.artifact)) {
      response.error = "cannot persist compiled artifact";
      return response;
    }
    response.ok = true;
    response.artifact_id = request.artifact_id;
    return response;
  }
  if (request.op != "run_compiled") {
    response.error =
        "mock runner only accepts capabilities/load_compiled/run_compiled";
    return response;
  }
  {
    std::lock_guard<std::mutex> lock(artifacts_mutex);
    if (!request.artifact.empty()) artifacts[request.artifact_id] = request.artifact;
    if (artifacts.count(request.artifact_id) == 0) {
      std::vector<uint8_t> persisted;
      if (load_persisted_artifact(request.artifact_id, persisted))
        artifacts[request.artifact_id] = std::move(persisted);
    }
    if (request.artifact_id.empty() || artifacts.count(request.artifact_id) == 0) {
      response.error = "compiled artifact is not attached";
      return response;
    }
  }
  // The mock does not interpret the artifact. Identity output makes it useful
  // for testing the transport/cache handshake without a vendor runtime.
  response.outputs = request.inputs;
  response.ok = true;
  if (request.profiling != ProfilingLevel::Off)
    response.profile.push_back(ProfileEvent{"mock_run", "mock", 0, 1, "identity"});
  return response;
}

int main(int argc, char** argv) {
  uint16_t port = 39510;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc)
      port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::string(argv[i]) == "--cache-dir" && i + 1 < argc)
      cache_dir = argv[++i];
    else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-mock-runner [--port PORT]"
                   " [--cache-dir DIR]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  const int listener = listen_tcp(port);
  if (listener < 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << '\n';
    return 1;
  }
  for (;;) {
    const int fd = accept_tcp(listener);
    if (fd < 0) continue;
    Request request;
    Response response;
    std::string error;
    if (!receive_request(fd, request, error)) response.error = error;
    else response = execute(request);
    if (!response.ok && response.error.empty()) response.error = "mock runner failed";
    send_response(fd, response, error);
    close_socket(fd);
  }
}
