// DORA node adapter for the dependency-free ONNX remote transport.
//
// The node accepts a UInt8 message named `run` containing the payload from
// encode_request_payload() and emits a UInt8 `result` message containing the
// status-prefixed payload from encode_response_payload(). DORA provides the
// dataflow/discovery lifecycle; the existing TCP worker remains responsible
// for model execution and accelerator-specific profiling.

#include <node_api.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "dora_status.h"
#include "remote_capabilities.h"
#include "remote_profile.h"
#include "remote_transport.h"

using namespace onnx_remote;

namespace {

std::string env_string(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : value;
}

uint16_t env_port() {
  const char* value = std::getenv("ONNXSIM_DORA_REMOTE_PORT");
  if (value == nullptr || *value == '\0') return 39501;
  const unsigned long port = std::strtoul(value, nullptr, 10);
  return port > 0 && port <= 65535 ? static_cast<uint16_t>(port) : 39501;
}

int env_timeout(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  const long timeout = std::strtol(value, nullptr, 10);
  return timeout > 0 && timeout <= 3600000 ? static_cast<int>(timeout)
                                           : fallback;
}

bool env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && (*value == '1' || *value == 'y' || *value == 'Y' ||
                              *value == 't' || *value == 'T');
}

bool query_capabilities(Response& response, std::string& error) {
  const int fd = connect_tcp_timeout(
      env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1"), env_port(),
      env_timeout("ONNXSIM_DORA_CONNECT_TIMEOUT_MS", 2000));
  if (fd < 0) {
    error = "DORA adapter: capability connection failed";
    return false;
  }
  if (!set_socket_io_timeout(fd, env_timeout("ONNXSIM_DORA_IO_TIMEOUT_MS", 0))) {
    close_socket(fd);
    error = "DORA adapter: cannot set capability socket timeout";
    return false;
  }
  Request request;
  request.op = "capabilities";
  const bool sent = send_request(fd, request, error);
  const bool received = sent && receive_response(fd, response, error);
  close_socket(fd);
  if (!received) return false;
  if (!response.ok) {
    error = response.error.empty() ? "DORA adapter: worker is not ready"
                                   : response.error;
    return false;
  }
  return true;
}

bool forward(const uint8_t* data, size_t size, std::vector<uint8_t>& result,
             uint64_t& request_id, Response& response_out,
             std::string& error) {
  Request request;
  if (!decode_request_payload(data, size, request, error)) return false;
  request_id = request.request_id;
  const int fd = connect_tcp_timeout(
      env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1"), env_port(),
      env_timeout("ONNXSIM_DORA_CONNECT_TIMEOUT_MS", 2000));
  if (fd < 0) {
    error = "DORA adapter: remote worker connection failed";
    return false;
  }
  if (!set_socket_io_timeout(fd, env_timeout("ONNXSIM_DORA_IO_TIMEOUT_MS", 0))) {
    close_socket(fd);
    error = "DORA adapter: cannot set socket timeout";
    return false;
  }
  const bool sent = send_request(fd, request, error);
  const bool received = sent && receive_response(fd, response_out, error);
  close_socket(fd);
  if (!received) return false;
  if (response_out.request_id != request.request_id) {
    error = "DORA adapter: remote worker response request id mismatch";
    return false;
  }
  return encode_response_payload(response_out, result, error);
}

void report_error(void* context, const std::string& error,
                  uint64_t request_id = 0) {
  dora_log(context, "error", 5, error.data(), error.size());
  if (std::getenv("ONNXSIM_DORA_ANNOUNCE") != nullptr) {
    const std::string status = onnx_remote::dora::readiness_status(
        false, env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1"),
        env_port(), error);
    dora_send_output(context, "status", 6, status.data(), status.size());
  }
  Response response;
  response.ok = false;
  response.request_id = request_id;
  response.error = error;
  std::vector<uint8_t> payload;
  std::string encode_error;
  if (encode_response_payload(response, payload, encode_error)) {
    dora_send_output(context, "result", 6,
                     reinterpret_cast<const char*>(payload.data()),
                     payload.size());
  }
}

}  // namespace

int main() {
  void* context = init_dora_context_from_env();
  if (context == nullptr) {
    std::cerr << "failed to initialize DORA context\n";
    return 1;
  }

  const bool require_graph_execution =
      env_flag("ONNXSIM_DORA_REQUIRE_GRAPH_EXECUTION");
  if (std::getenv("ONNXSIM_DORA_ANNOUNCE") != nullptr ||
      require_graph_execution) {
    Response worker_capabilities;
    std::string capability_error;
    bool ready = query_capabilities(worker_capabilities, capability_error);
    if (ready && require_graph_execution) {
      CapabilitySummary summary;
      if (!parse_capability_manifest(worker_capabilities.manifest, summary,
                                     capability_error) ||
          !summary.graph_execution) {
        if (capability_error.empty())
          capability_error = "worker does not advertise graph execution";
        ready = false;
      }
    }
    const std::string host =
        env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1");
    const uint16_t port = env_port();
    if (std::getenv("ONNXSIM_DORA_ANNOUNCE") != nullptr) {
      const std::string status = onnx_remote::dora::readiness_status(
          ready, host, port, capability_error);
      dora_send_output(context, "status", 6, status.data(), status.size());
      if (ready && !worker_capabilities.manifest.empty()) {
        dora_send_output(context, "capabilities", 12,
                         worker_capabilities.manifest.data(),
                         worker_capabilities.manifest.size());
      } else {
        const std::string error =
            onnx_remote::dora::unavailable_capabilities(capability_error);
        dora_send_output(context, "capabilities", 12, error.data(), error.size());
      }
    }
    if (require_graph_execution && !ready) {
      std::cerr << (capability_error.empty()
                        ? "DORA adapter: graph-capable worker unavailable\n"
                        : "DORA adapter: " + capability_error + "\n");
      free_dora_context(context);
      return 2;
    }
  }

  for (;;) {
    void* event = dora_next_event(context);
    if (event == nullptr) break;
    const DoraEventType type = read_dora_event_type(event);
    if (type == DoraEventType_Stop) {
      free_dora_event(event);
      break;
    }
    if (type != DoraEventType_Input) {
      free_dora_event(event);
      continue;
    }

    char* id = nullptr;
    size_t id_len = 0;
    char* data = nullptr;
    size_t data_len = 0;
    read_dora_input_id(event, &id, &id_len);
    read_dora_input_data(event, &data, &data_len);
    const bool is_run =
        id != nullptr && id_len == 3 && std::memcmp(id, "run", 3) == 0;
    if (is_run && data != nullptr) {
      std::vector<uint8_t> result;
      uint64_t request_id = 0;
      Response response;
      std::string error;
      if (forward(reinterpret_cast<const uint8_t*>(data), data_len, result,
                  request_id, response, error)) {
        if (dora_send_output(context, "result", 6,
                             reinterpret_cast<const char*>(result.data()),
                             result.size()) != 0) {
          report_error(context, "DORA adapter: result output failed",
                       request_id);
        }
        if (std::getenv("ONNXSIM_DORA_PUBLISH_PROFILE") != nullptr &&
            !response.profile.empty()) {
          const std::string profile = profile_json(response);
          dora_send_output(context, "profile", 7, profile.data(), profile.size());
        }
        if (std::getenv("ONNXSIM_DORA_PUBLISH_PROFILE_EVENTS") != nullptr) {
          receive_profile(response, [&](const ProfileEvent& event) {
            const std::string profile = profile_event_json(response.request_id,
                                                            event);
            dora_send_output(context, "profile_event", 13, profile.data(),
                             profile.size());
          });
        }
      } else {
        report_error(context,
                     error.empty() ? "DORA adapter: request failed" : error,
                     request_id);
      }
    }
    free_dora_event(event);
  }
  free_dora_context(context);
  return 0;
}
