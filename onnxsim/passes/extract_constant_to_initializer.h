// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 A = Constant()
// After:
//	 A is in the initializer list
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: A is the output of a Constant node
#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

struct ExtractConstantToInitializer final : public PredicateBasedPass {
  explicit ExtractConstantToInitializer()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "extract_constant_to_initializer";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConstant && node->hasAttribute(kvalue);
  }

  // This pass instance is reused across every round of onnxsim's
  // simplification fixed point, but a batch of reserved names is only valid
  // for the graph state it was reserved against -- other passes/rounds can
  // introduce new names in between. Drop any leftover reservation from a
  // prior runPass() call so nextReservedName() always reserves fresh against
  // the current graph.
  bool initializePass(Graph&) override {
    reserved_names_.clear();
    reserved_used_ = 0;
    return false;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Tensor t = node->t(kvalue);
    Value* new_init;
    if (node->output()->has_unique_name() &&
        std::find(graph.outputs().rbegin(), graph.outputs().rend(),
                  node->output()) == graph.outputs().rend()) {
      t.setName(node->output()->uniqueName());
      // t is a local temporary this call never touches again -- move it
      // into the new initializer instead of copying
      // (addInitializerAndCreateValue has an rvalue overload for exactly this)
      // to skip a second deep copy of its raw_data/typed-data fields on top of
      // the one `node->t(kvalue)` already paid above. On a model with many
      // Constant nodes this pass was ~half its own cost (see
      // bench/RESULTS_profiling_survey.md's ONNXSIM_PROFILE_PASS_PHASES data).
      new_init = graph.addInitializerAndCreateValue(std::move(t));
      node->output()->setUniqueName(nextReservedName(graph), false);
    } else {
      // addInitializerAndCreateValue -> addInitializer auto-generates a name
      // via getNextUniqueName() when t's is empty (the common case here, a
      // Constant node's embedded tensor rarely carries its own name); reserve
      // one up front instead so that path also goes through the batched
      // reservation below rather than paying a fresh full-graph scan here.
      // Always overwrite: the attribute TensorProto's own name is unrelated
      // to this graph's namespace and can collide with an existing value
      // (or with another Constant's tensor of the same name).
      t.setName(nextReservedName(graph));
      new_init = graph.addInitializerAndCreateValue(std::move(t));
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), new_init);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }

 private:
  // A model with many Constant nodes (e.g. the window-partition index/shape
  // constants in a shifted-window-attention model) makes this pass call
  // Graph::getNextUniqueName() once per match -- and each such call pays a
  // full graph (and subgraph) scan of its own, turning this otherwise-linear
  // pass into an accidental O(matches * graph size) cost (see onnxsim issue
  // #651's follow-up). Draw fresh names from a small batch reserved via
  // Graph::reserveUniqueNames() (one scan per batch) instead of one scan per
  // name; refilled lazily so a graph with few Constant nodes still pays only
  // one (cheap) reservation.
  static constexpr size_t kNameBatchSize = 256;
  std::vector<std::string> reserved_names_;
  size_t reserved_used_ = 0;

  std::string nextReservedName(Graph& graph) {
    if (reserved_used_ >= reserved_names_.size()) {
      reserved_names_ = graph.reserveUniqueNames(kNameBatchSize);
      reserved_used_ = 0;
    }
    return std::move(reserved_names_[reserved_used_++]);
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
