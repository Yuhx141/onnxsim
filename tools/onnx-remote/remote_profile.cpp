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
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default: escaped.push_back(character); break;
    }
  }
  return escaped;
}

}  // namespace

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

}  // namespace onnx_remote
