#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "onnx/checker.h"
#include "onnx/common/ir_pb_converter.h"

static bool check(const onnx::ModelProto& model, bool full, const char* phase) {
  try {
    onnx::checker::check_model(model, full);
    std::cout << phase << (full ? ".full=passed\n" : ".basic=passed\n");
    return true;
  } catch (const std::exception& error) {
    std::string message = error.what();
    while (!message.empty() && message.back() == '\n') message.pop_back();
    std::cout << phase << (full ? ".full=failed: " : ".basic=failed: ")
              << message << '\n';
    return false;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: roundtrip INPUT.onnx OUTPUT.onnx\n";
    return 2;
  }
  std::cout.setf(std::ios::unitbuf);
  onnx::ModelProto source;
  std::ifstream input(argv[1], std::ios::binary);
  if (!input || !source.ParseFromIstream(&input)) return 3;
  const bool source_basic = check(source, false, "source");
  check(source, true, "source");
  try {
    std::shared_ptr<onnx::Graph> graph = onnx::ImportModelProto(source);
    if (!graph) return 4;
    std::cout << "ir_import=passed\n";
    for (const auto* output : graph->outputs()) {
      std::cout << "ir_output=" << output->uniqueName()
                << " producer=" << output->node()->kind().toString()
                << " uses=" << output->uses().size() << '\n';
    }
    auto target = onnx::PrepareOutput(source);
    onnx::ExportModelProto(&target, graph);
    std::ofstream output(argv[2], std::ios::binary);
    if (!target.SerializeToOstream(&output)) return 5;
    std::cout << "ir_export=passed\n";
    const bool target_basic = check(target, false, "export");
    const bool target_full = check(target, true, "export");
    return source_basic && target_basic && target_full ? 0 : 1;
  } catch (const std::exception& error) {
    std::cout << "ir_error=" << error.what() << '\n';
    return 1;
  }
}
