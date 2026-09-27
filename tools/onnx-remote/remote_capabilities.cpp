#include "remote_capabilities.h"

#include <cctype>
#include <cstdlib>

namespace onnx_remote {
namespace {

size_t value_start(const std::string& json, const char* key) {
  const std::string marker = std::string("\"") + key + "\"";
  const size_t key_start = json.find(marker);
  if (key_start == std::string::npos) return std::string::npos;
  size_t cursor = key_start + marker.size();
  while (cursor < json.size() &&
         std::isspace(static_cast<unsigned char>(json[cursor])))
    ++cursor;
  if (cursor >= json.size() || json[cursor++] != ':') return std::string::npos;
  while (cursor < json.size() &&
         std::isspace(static_cast<unsigned char>(json[cursor])))
    ++cursor;
  return cursor;
}

bool json_string(const std::string& json, const char* key, std::string& value) {
  const size_t begin = value_start(json, key);
  if (begin == std::string::npos || begin >= json.size() || json[begin] != '"')
    return false;
  const size_t string_begin = begin + 1;
  const size_t end = json.find('"', string_begin);
  if (end == std::string::npos) return false;
  value = json.substr(string_begin, end - string_begin);
  return true;
}

bool json_bool(const std::string& json, const char* key, bool& value) {
  const size_t begin = value_start(json, key);
  if (begin == std::string::npos) return false;
  if (json.compare(begin, 4, "true") == 0) {
    value = true;
    return true;
  }
  if (json.compare(begin, 5, "false") == 0) {
    value = false;
    return true;
  }
  return false;
}

}  // namespace

bool parse_capability_manifest(const std::string& manifest,
                               CapabilitySummary& summary,
                               std::string& error) {
  if (manifest.size() > 64 * 1024) {
    error = "capability manifest exceeds limit";
    return false;
  }
  const size_t schema_start = value_start(manifest, "schema_version");
  if (schema_start == std::string::npos) {
    error = "capability manifest has no schema_version";
    return false;
  }
  char* end = nullptr;
  const long version = std::strtol(manifest.c_str() + schema_start, &end, 10);
  if (end == manifest.c_str() + schema_start || version != 1) {
    error = "unsupported capability manifest schema";
    return false;
  }
  summary.schema_version = static_cast<int>(version);
  if (!json_string(manifest, "protocol", summary.protocol) ||
      !json_string(manifest, "runner_id", summary.runner_id)) {
    error = "capability manifest is missing protocol or runner_id";
    return false;
  }
  if (summary.protocol != "onnx-remote-v5") {
    error = "unsupported capability manifest protocol";
    return false;
  }
  // Optional for legacy reference workers; graph-capable workers must publish
  // this explicitly as true.
  json_bool(manifest, "graph_execution", summary.graph_execution);
  json_bool(manifest, "profiling", summary.profiling);
  return true;
}

}  // namespace onnx_remote
