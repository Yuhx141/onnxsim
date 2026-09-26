#pragma once

#include <string>

namespace onnx_remote {

// Names used by ONNX nodes are conventionally title-cased while the
// dependency-free reference worker uses compact lowercase operation names.
// Keep this mapping independent from ORT headers so it can be tested in every
// build configuration.
inline std::string remote_operation_name(const std::string &onnx_op) {
  if (onnx_op == "Identity")
    return "identity";
  if (onnx_op == "Relu")
    return "relu";
  if (onnx_op == "Add")
    return "add";
  if (onnx_op == "Mul")
    return "mul";
  return {};
}

} // namespace onnx_remote
