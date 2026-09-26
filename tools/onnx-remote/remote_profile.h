#pragma once

#include <string>

#include "remote_transport.h"

namespace onnx_remote {

// Serialize worker-relative profile events for lightweight ROS2/DORA/UI
// consumers. The binary response remains the lossless transport format.
std::string profile_json(const Response& response);

// Serialize the same worker-relative events as Chrome Trace Event Format.
// `ts` and `dur` are microseconds, matching the native response fields, so
// the result can be loaded directly by Perfetto or chrome://tracing.
std::string profile_chrome_trace_json(const Response& response);

}  // namespace onnx_remote
