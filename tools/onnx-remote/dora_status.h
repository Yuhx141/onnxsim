#pragma once

#include <cstdint>
#include <string>

namespace onnx_remote::dora {

inline std::string json_escape(const std::string& value) {
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

inline std::string readiness_status(bool ready, const std::string& host,
                                    uint16_t port,
                                    const std::string& error = {}) {
  std::string status =
      std::string("{\"schema_version\":1,\"status\":\"") +
      (ready ? "ready" : "unavailable") +
      "\",\"protocol\":\"onnx-remote-v5\",\"host\":\"" +
      json_escape(host) + "\",\"port\":" + std::to_string(port);
  if (!ready && !error.empty())
    status += ",\"error\":\"" + json_escape(error) + "\"";
  return status + "}";
}

inline std::string unavailable_capabilities(const std::string& error) {
  const std::string reason = error.empty() ? "worker unavailable" : error;
  return "{\"schema_version\":1,\"ready\":false,\"error\":\"" +
         json_escape(reason) + "\"}";
}

}  // namespace onnx_remote::dora
