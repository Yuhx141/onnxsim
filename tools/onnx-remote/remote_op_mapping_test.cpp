#include "remote_op_mapping.h"

#include <cassert>

int main() {
  using onnx_remote::remote_operation_name;
  assert(remote_operation_name("Identity") == "identity");
  assert(remote_operation_name("Relu") == "relu");
  assert(remote_operation_name("Add") == "add");
  assert(remote_operation_name("Mul") == "mul");
  assert(remote_operation_name("Conv").empty());
  assert(remote_operation_name("ai.onnx::Relu").empty());
  return 0;
}
