#pragma once

#include <cstdint>
#include <string>

namespace onnx_remote {

struct CapabilitySummary {
  int schema_version = 0;
  std::string protocol;
  std::string runner_id;
  bool graph_execution = false;
  bool profiling = false;
};

// Parse the small stable fields needed by discovery and graph dispatch. The
// full manifest remains opaque to vendor-specific consumers.
bool parse_capability_manifest(const std::string& manifest,
                               CapabilitySummary& summary,
                               std::string& error);

}  // namespace onnx_remote
