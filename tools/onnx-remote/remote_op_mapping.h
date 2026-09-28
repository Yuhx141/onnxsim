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
  if (onnx_op == "Sub")
    return "sub";
  if (onnx_op == "Div")
    return "div";
  if (onnx_op == "Max")
    return "max";
  if (onnx_op == "Min")
    return "min";
  if (onnx_op == "Abs")
    return "abs";
  if (onnx_op == "Neg")
    return "neg";
  if (onnx_op == "Sqrt")
    return "sqrt";
  if (onnx_op == "Exp")
    return "exp";
  if (onnx_op == "Log")
    return "log";
  if (onnx_op == "Tanh")
    return "tanh";
  return {};
}

} // namespace onnx_remote
