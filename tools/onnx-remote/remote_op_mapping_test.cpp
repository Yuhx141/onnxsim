#include "remote_op_mapping.h"

#include <cassert>

int main() {
  using onnx_remote::remote_operation_name;
  assert(remote_operation_name("Identity") == "identity");
  assert(remote_operation_name("Relu") == "relu");
  assert(remote_operation_name("Add") == "add");
  assert(remote_operation_name("Mul") == "mul");
  assert(remote_operation_name("Sub") == "sub");
  assert(remote_operation_name("Div") == "div");
  assert(remote_operation_name("Max") == "max");
  assert(remote_operation_name("Min") == "min");
  assert(remote_operation_name("Abs") == "abs");
  assert(remote_operation_name("Neg") == "neg");
  assert(remote_operation_name("Sqrt") == "sqrt");
  assert(remote_operation_name("Exp") == "exp");
  assert(remote_operation_name("Log") == "log");
  assert(remote_operation_name("Tanh") == "tanh");
  assert(remote_operation_name("Conv").empty());
  assert(remote_operation_name("ai.onnx::Relu").empty());
  return 0;
}
