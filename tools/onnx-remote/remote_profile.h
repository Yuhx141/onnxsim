#pragma once

#include <string>

#include "remote_transport.h"

namespace onnx_remote {

// Serialize worker-relative profile events for lightweight ROS2/DORA/UI
// consumers. The binary response remains the lossless transport format.
std::string profile_json(const Response& response);

}  // namespace onnx_remote
