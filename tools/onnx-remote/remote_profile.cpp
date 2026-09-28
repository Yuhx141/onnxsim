#include "remote_profile.h"

#include <sstream>

namespace onnx_remote {
namespace {

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(hex[(static_cast<unsigned char>(character) >> 4) & 0xf]);
          escaped.push_back(hex[static_cast<unsigned char>(character) & 0xf]);
        } else {
          escaped.push_back(character);
        }
        break;
    }
  }
  return escaped;
}

}  // namespace

void receive_profile(const Response& response,
                     const ProfileReceiver& receiver) {
  if (!receiver) return;
  for (const auto& event : response.profile) receiver(event);
}

std::string profile_event_json(uint64_t request_id,
                               const ProfileEvent& event) {
  std::ostringstream json;
  json << "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
       << "\"request_id\":" << request_id << ",\"event\":{"
       << "\"name\":\"" << json_escape(event.name)
       << "\",\"category\":\"" << json_escape(event.category)
       << "\",\"start_us\":" << event.start_us
       << ",\"duration_us\":" << event.duration_us
       << ",\"detail\":\"" << json_escape(event.detail) << "\"}}";
  return json.str();
}

std::string profile_json(const Response& response) {
  std::ostringstream json;
  json << "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\",\"request_id\":"
       << response.request_id << ",\"events\":[";
  for (size_t i = 0; i < response.profile.size(); ++i) {
    if (i != 0) json << ',';
    const auto& event = response.profile[i];
    json << "{\"name\":\"" << json_escape(event.name)
         << "\",\"category\":\"" << json_escape(event.category)
         << "\",\"start_us\":" << event.start_us
         << ",\"duration_us\":" << event.duration_us
         << ",\"detail\":\"" << json_escape(event.detail) << "\"}";
  }
  json << "]}";
  return json.str();
}

std::string profile_chrome_trace_json(const Response& response) {
  std::ostringstream json;
  json << "{\"displayTimeUnit\":\"us\",\"schema_version\":1,"
       << "\"protocol\":\"onnx-remote-v5\",\"request_id\":"
       << response.request_id << ",\"traceEvents\":[";
  for (size_t i = 0; i < response.profile.size(); ++i) {
    if (i != 0) json << ',';
    const auto& event = response.profile[i];
    json << "{\"name\":\"" << json_escape(event.name)
         << "\",\"cat\":\"" << json_escape(event.category)
         << "\",\"ph\":\"X\",\"ts\":" << event.start_us
         << ",\"dur\":" << event.duration_us
         << ",\"pid\":1,\"tid\":\""
         << json_escape(event.category) << "\",\"args\":{\"detail\":\""
         << json_escape(event.detail) << "\"}}";
  }
  json << "]}";
  return json.str();
}

}  // namespace onnx_remote
