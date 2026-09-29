// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once

#include <chrono>
#include <unordered_map>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/string_utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// onnxsim's own passes live in this nested namespace so their class
// names never collide (ODR) with the same-named passes compiled into
// onnxoptimizer; RegisterOrReplace still keys them by getPassName().
namespace onnxsim_passes {

// The operator domain is part of a node's identity: `ai.onnx::Foo` and
// `custom::Foo` (e.g. local functions) are different operators even when
// their inputs and attributes match. "" and "ai.onnx" name the same domain.
inline std::string CSEDomainOf(const Node* n) {
  if (!n->has_domain() || n->domain() == "ai.onnx") {
    return "";
  }
  return n->domain();
}

struct DomainAwareCSEHash {
  std::size_t operator()(const Node* n) const {
    return CSENodeHash()(n) ^
           (std::hash<std::string>()(CSEDomainOf(n)) * 0x9e3779b97f4a7c15ULL);
  }
};

struct DomainAwareCSEEqual {
  bool operator()(const Node* l, const Node* r) const {
    if (l && r && CSEDomainOf(l) != CSEDomainOf(r)) {
      return false;
    }
    return CSEEqual()(l, r);
  }
};

struct EliminateCommonSubexpression final : public FullGraphBasedPass {
  explicit EliminateCommonSubexpression()
      : FullGraphBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "eliminate_common_subexpression";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::CountBased;
  }

  unsigned int EliminateCommonSubexpressions(Graph& graph) {
    // No longer cleared here: see EliminateDuplicateInitializer's identical
    // change for why (TensorContentDigest's cache now outlives a single pass
    // call; clearing it is Optimizer::optimize(Graph&, ...)'s job).
    const bool profiling = GetPassPhaseProfilingEnabled();
    auto node_list = graph.nodes();
    unsigned int cse_removed = 0;
    uint64_t nodes_seen = 0;
    uint64_t nodes_filtered_out = 0;
    uint64_t nodes_replaced = 0;
    double filter_ms = 0.0;
    double lookup_ms = 0.0;
    double replace_ms = 0.0;
    std::unordered_map<Node*, Node*, DomainAwareCSEHash, DomainAwareCSEEqual>
        hash_map;
    // See eliminate_deadend.h's identical use of GraphMayHaveCapturedValues
    // for why this turns hasUses() from an accidental O(nodes) cost on every
    // one of this loop's O(nodes) iterations into O(1), for the overwhelming
    // majority of graphs that have no control-flow ops at all.
    const bool may_have_captures = GraphMayHaveCapturedValues(graph);
    for (auto it = node_list.begin(); it != node_list.end(); ++it) {
      auto node = *it;
      auto kind = node->kind();
      nodes_seen++;
      bool skip;
      if (profiling) {
        const auto t0 = std::chrono::steady_clock::now();
        skip = (may_have_captures ? !node->hasUses()
                                  : !node->hasUsesInCurrentGraph()) ||
               !IsSupportedByCSE(node);
        const auto t1 = std::chrono::steady_clock::now();
        filter_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      } else {
        skip = (may_have_captures ? !node->hasUses()
                                  : !node->hasUsesInCurrentGraph()) ||
               !IsSupportedByCSE(node);
      }
      if (skip) {
        nodes_filtered_out++;
        continue;
      }
      VLOG(2) << Str("kind: ", kind.toString(), ", ", node->name(),
                     " is processing");
      // A single emplace instead of find()-then-[]/at(): see
      // EliminateDuplicateInitializer's identical fix for why. Note this
      // (and thus lookup_ms below) internally calls CSENodeHash and, on a
      // bucket collision, CSEEqual -- see cse_util.h's node_hash_ms/
      // node_equal_ms for that same work's own breakdown.
      std::chrono::steady_clock::time_point t0;
      if (profiling) t0 = std::chrono::steady_clock::now();
      auto insertion = hash_map.emplace(node, node);
      if (profiling) {
        const auto t1 = std::chrono::steady_clock::now();
        lookup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
      if (!insertion.second) {
        auto other = insertion.first->second;
        auto outputs = other->outputs();
        auto replaced_outputs = node->outputs();
        std::chrono::steady_clock::time_point t2;
        if (profiling) t2 = std::chrono::steady_clock::now();
        for (int i = 0; i < outputs.size(); ++i) {
          if (tryReplacingAllUsesWith(replaced_outputs[i], outputs[i])) {
            VLOG(1) << Str("kind: ", kind.toString(), ", ", node->name(), " [",
                           i, "] output has been replaced by ", other->name());
            cse_removed++;
            nodes_replaced++;
          }
        }
        if (profiling) {
          const auto t3 = std::chrono::steady_clock::now();
          replace_ms +=
              std::chrono::duration<double, std::milli>(t3 - t2).count();
        }
      }
    }
    if (profiling) {
      RecordCSEPassTiming(nodes_seen, nodes_filtered_out, nodes_replaced,
                          filter_ms, lookup_ms, replace_ms);
    }
    return cse_removed;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    auto cse_removed = this->EliminateCommonSubexpressions(graph);
    VLOG(1) << Str("cse_removed count: ", cse_removed);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, cse_removed, false, false));
  }
};
}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
