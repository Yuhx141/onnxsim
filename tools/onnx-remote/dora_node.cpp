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

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (const char c : value) {
    switch (c) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(hex[(static_cast<unsigned char>(c) >> 4) & 0xf]);
          escaped.push_back(hex[static_cast<unsigned char>(c) & 0xf]);
        } else {
          escaped += c;
        }
        break;
    }
  }
  return escaped;
}

bool query_capabilities(Response& response, std::string& error) {
  const int fd = connect_tcp_timeout(
      env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1"), env_port(),
      env_timeout("ONNXSIM_DORA_CONNECT_TIMEOUT_MS", 2000));
  if (fd < 0) {
    error = "DORA adapter: capability connection failed";
    return false;
  }
  set_socket_io_timeout(fd, env_timeout("ONNXSIM_DORA_IO_TIMEOUT_MS", 0));
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
  set_socket_io_timeout(fd, env_timeout("ONNXSIM_DORA_IO_TIMEOUT_MS", 0));
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

  if (std::getenv("ONNXSIM_DORA_ANNOUNCE") != nullptr) {
    Response worker_capabilities;
    std::string capability_error;
    const bool ready = query_capabilities(worker_capabilities, capability_error);
    const std::string host =
        env_string("ONNXSIM_DORA_REMOTE_HOST", "127.0.0.1");
    const uint16_t port = env_port();
    std::string status =
        std::string("{\"schema_version\":1,\"status\":\"") +
        (ready ? "ready" : "unavailable") +
        "\",\"protocol\":\"onnx-remote-v5\",\"host\":\"" +
        json_escape(host) + "\",\"port\":" + std::to_string(port);
    if (!ready && !capability_error.empty())
      status += ",\"error\":\"" + json_escape(capability_error) + "\"";
    status += "}";
    dora_send_output(context, "status", 6, status.data(), status.size());
    if (ready && !worker_capabilities.manifest.empty()) {
      dora_send_output(context, "capabilities", 12,
                       worker_capabilities.manifest.data(),
                       worker_capabilities.manifest.size());
    } else {
      const std::string reason = capability_error.empty()
                                     ? "worker unavailable"
                                     : capability_error;
      const std::string error =
          "{\"schema_version\":1,\"ready\":false,\"error\":\"" +
          json_escape(reason) + "\"}";
      dora_send_output(context, "capabilities", 12, error.data(), error.size());
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
