// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before (how PyTorch and other exporters unroll L1/L2 normalization; see
// https://github.com/onnx/optimizer/issues/143):
//   N = ReduceL2(X, axes=[a], keepdims=1)    # or ReduceL1
//   Y = Div(X, N)
// After:
//   Y = LpNormalization<axis=a, p=2>(X)      # p=1 for ReduceL1
//
// LpNormalization divides by the same p-norm along `axis` with no epsilon
// clamp, so the fusion is exact -- including the zero-norm case, which is
// inf/NaN in both forms. A `Clip`/`Max` with an epsilon between the reduction
// and the Div (what `torch.nn.functional.normalize` exports) is therefore
// deliberately *not* matched: it would change the result for near-zero norms.
//
// Requirements: the Div's numerator is the very value that was reduced, the
// reduction has a single axis and keepdims=1, the reduction's output feeds
// only this Div, and X is float16/float/double (LpNormalization's types).

#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct FuseLpNormalization final : public PredicateBasedPass {
  explicit FuseLpNormalization()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_lp_normalization"; }

  struct Match {
    Value* x = nullptr;
    Node* reduce = nullptr;
    int64_t axis = -1;
    int64_t p = 2;
  };

  static bool TryMatch(Node* div, Match& m) {
    if (div->kind() != kDiv || div->inputs().size() != 2) {
      return false;
    }
    Value* x = div->input(0);
    Value* denom = div->input(1);
    const bool l1 = CheckKind(denom, Symbol("ReduceL1"));
    if (!l1 && !CheckKind(denom, Symbol("ReduceL2"))) {
      return false;
    }
    Node* reduce = denom->node();
    if (denom->uses().size() != 1 || reduce->input(0) != x) {
      return false;
    }
    if (GetValueFromAttrWithDefault<int64_t>(reduce, kkeepdims, 1) != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    if (!GetValueFromAttrOrInput(reduce, kaxes, 1, axes) || axes.size() != 1) {
      return false;
    }
    switch (x->elemType()) {
      case TensorProto_DataType_FLOAT:
      case TensorProto_DataType_DOUBLE:
      case TensorProto_DataType_FLOAT16:
        break;
      default:
        return false;
    }
    m.x = x;
    m.reduce = reduce;
    m.axis = axes[0];
    m.p = l1 ? 1 : 2;
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    Match m;
    return TryMatch(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!TryMatch(n, m)) {
      return false;
    }
    Node* lp = graph.create(Symbol("LpNormalization"), 1);
    lp->addInput(m.x);
    lp->i_(kaxis, m.axis);
    lp->i_(Symbol("p"), m.p);
    lp->insertBefore(n);
    lp->output()->setElemType(n->output()->elemType());
    if (n->output()->has_sizes()) {
      lp->output()->setSizes(n->output()->sizes());
    }
    n->output()->replaceAllUsesWith(lp->output());
    // Sever `n`'s inputs so the reduction becomes unused, then drop it
    // (a PredicateBasedPass only removes the node it matched on).
    n->removeAllInputs();
    if (!m.reduce->hasUsesInCurrentGraph()) {
      m.reduce->destroy();
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
