#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onnx_remote {

struct CapabilitySummary {
  int schema_version = 0;
  std::string protocol;
  std::string runner_id;
  bool graph_execution = false;
  bool profiling = false;
  // Lowercase operation names from the manifest's supported_ops array. Empty
  // when the worker publishes a legacy manifest without that array.
  std::vector<std::string> supported_ops;
};

// Parse the small stable fields needed by discovery and graph dispatch. The
// full manifest remains opaque to vendor-specific consumers.
bool parse_capability_manifest(const std::string& manifest,
                               CapabilitySummary& summary,
                               std::string& error);

}  // namespace onnx_remote
