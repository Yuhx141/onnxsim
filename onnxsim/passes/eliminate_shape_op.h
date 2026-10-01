// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include "onnx/common/tensor.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct EliminateShapeOp final : public PredicateBasedPass {
  explicit EliminateShapeOp()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_shape_op"; }

  bool patternMatchPredicate(Node* node) override {
    if (!CheckKind(node, "Shape") || !HasDimsOfInputOfNode(node, 0)) {
      return false;
    }
    const auto [start, end] = normalizedRange(node);
    const auto& sizes = node->input()->sizes();
    return std::all_of(
        sizes.cbegin() + start, sizes.cbegin() + end,
        [](const auto& dim) { return dim.is_int && dim.dim >= 0; });
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const auto [start, end] = normalizedRange(node);
    const auto& sizes = node->input()->sizes();

    Tensor tensor;
    tensor.sizes().push_back(end - start);
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    std::transform(sizes.begin() + start, sizes.begin() + end,
                   std::back_inserter(tensor.int64s()),
                   [](const auto& dim) { return dim.dim; });

    Value* value = graph.addInitializerAndCreateValue(tensor);
    if (!tryReplacingAllUsesWith(node->output(), value)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }

 private:
  static std::pair<int64_t, int64_t> normalizedRange(const Node* node) {
    const int64_t rank = static_cast<int64_t>(node->input()->sizes().size());
    auto [start, end] = FetchStartAndEndAttrOfShape(node, rank);
    start = std::clamp<int64_t>(start, 0, rank);
    end = std::clamp<int64_t>(end, 0, rank);
    return {start, std::max(start, end)};
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
