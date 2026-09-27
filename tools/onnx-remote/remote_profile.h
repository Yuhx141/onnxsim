#pragma once

#include <functional>
#include <string>

#include "remote_transport.h"

namespace onnx_remote {

using ProfileReceiver = std::function<void(const ProfileEvent&)>;

// Deliver lossless binary profile events to an application-owned sink. This
// avoids requiring JSON or another parser on constrained hosts.
void receive_profile(const Response& response, const ProfileReceiver& receiver);

// Serialize worker-relative profile events for lightweight ROS2/DORA/UI
// consumers. The binary response remains the lossless transport format.
std::string profile_json(const Response& response);

// Serialize the same worker-relative events as Chrome Trace Event Format.
// `ts` and `dur` are microseconds, matching the native response fields, so
// the result can be loaded directly by Perfetto or chrome://tracing.
std::string profile_chrome_trace_json(const Response& response);

}  // namespace onnx_remote
