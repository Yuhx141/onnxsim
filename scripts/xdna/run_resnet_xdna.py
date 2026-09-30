#!/usr/bin/env python3
"""Execute a static QDQ ResNet graph with selected kernels on XDNA.

Convolutions and supported operator kernels run through compiled IRON/XRT
artifacts. Unsupported ONNX ops run in NumPy on the host.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from conv_lowering import plan_all_convs
from conv_reference import im2col_nchw
from resnet_codegen import build_codegen_plan
try:
    from .benchmark_fused_bottleneck import bind_fused_bottleneck
    from .resnet_bottleneck import plan_bottleneck_blocks
except ImportError:  # executed directly as a script
    from benchmark_fused_bottleneck import bind_fused_bottleneck
    from resnet_bottleneck import plan_bottleneck_blocks


@dataclass
class _DeviceValue:
    """A graph edge retained in an XRT allocation with a known memory layout."""

    tensor: Any
    shape: tuple[int, ...]
    scale: float
    zero_point: int
    as_real: bool = False
    producer: str = ""
    layout: str = "nhwc"


def _attrs(node: Any) -> dict[str, Any]:
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def _values(model: Any) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _quantize(x: np.ndarray, scale: np.ndarray, zero: np.ndarray, axis: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    zero = np.asarray(zero)
    if scale.size > 1:
        shape = [1] * x.ndim
        shape[axis] = scale.size
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
    info = np.iinfo(zero.dtype)
    quantized = np.rint(np.asarray(x, dtype=np.float32) / scale) + zero
    return np.clip(quantized, info.min, info.max).astype(zero.dtype)


def _dequantize(x: np.ndarray, scale: np.ndarray, zero: np.ndarray, axis: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    zero = np.asarray(zero)
    if scale.size > 1:
        shape = [1] * x.ndim
        shape[axis] = scale.size
        scale = scale.reshape(shape)
        zero = zero.reshape(shape)
    return (np.asarray(x).astype(np.float32) - zero.astype(np.float32)) * scale


def _centered_int8(raw: np.ndarray, zero: int, label: str) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.dtype == np.int8 and zero == 0:
        return raw
    if raw.dtype == np.uint8 and zero == 128:
        # Rotate the unsigned range around 128 with one signed-byte copy.
        return np.bitwise_xor(raw.view(np.int8), np.int8(-128))
    centered = raw.astype(np.int16) - zero
    if np.any((centered < -128) | (centered > 127)):
        raise ValueError(f"{label} cannot be represented as signed int8")
    return centered.astype(np.int8)


def _max_pool(x: np.ndarray, attrs: dict[str, Any]) -> np.ndarray:
    kernel = tuple(attrs["kernel_shape"])
    strides = tuple(attrs.get("strides", (1,) * len(kernel)))
    dilations = tuple(attrs.get("dilations", (1,) * len(kernel)))
    pads = tuple(attrs.get("pads", (0,) * (2 * len(kernel))))
    if x.ndim != 4 or len(kernel) != 2:
        raise ValueError("ResNet runner currently supports 2D NCHW MaxPool")
    n, c, h, w = x.shape
    pt, pl, pb, pr = pads
    padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
    out_h = (h + pt + pb - dilations[0] * (kernel[0] - 1) - 1) // strides[0] + 1
    out_w = (w + pl + pr - dilations[1] * (kernel[1] - 1) - 1) // strides[1] + 1
    # One strided slice per kernel tap, reduced with an elementwise maximum (no per-pixel Python loop).
    output = None
    for ky in range(kernel[0]):
        for kx in range(kernel[1]):
            tap = padded[
                :, :,
                ky * dilations[0] : ky * dilations[0] + (out_h - 1) * strides[0] + 1 : strides[0],
                kx * dilations[1] : kx * dilations[1] + (out_w - 1) * strides[1] + 1 : strides[1],
            ]
            output = np.array(tap, dtype=x.dtype) if output is None else np.maximum(output, tap, out=output)
    return output


class XDNAResNetRunner:
    def __init__(
        self,
        model: Any,
        manifest: dict[str, Any],
        cpu_small_m: int = 0,
        cpu_backend: str = "numpy",
        cpu_threads: int = 1,
        fused_block_prefix: str | None = None,
        fused_block_xclbin: str | None = None,
        fused_block_insts: str | None = None,
        fused_blocks: list[tuple[str, str, str]] | None = None,
        fused_stages: list[tuple[tuple[str, str, str], str, str]] | None = None,
        fused_stage_blocked: bool = False,
        fused_body_groups: list[list[str]] | None = None,
        fused_body_chunk_caps: dict[str, int | None] | None = None,
        host_maxpool: bool = False,
        fused_body_rt: bool = False,
        device_network_stages: list[list[str]] | None = None,
        layer_engine: bool = False,
        parallel_projection_blocks: list[tuple[str, str, str]] | None = None,
        maxpool_uint8_artifact: tuple[str, str] | None = None,
        maxpool_runtime: str = "iron",
        runtime_backend: str | None = None,
        capture_outputs: bool = False,
    ):
        self.model = model
        self.nodes = list(model.graph.node)
        self.arrays = _values(model)
        self.optimize_small_m = bool(manifest.get("optimize_small_m", False))
        self.cpu_small_m = max(0, int(cpu_small_m))
        self.cpu_backend = cpu_backend
        self.cpu_threads = max(1, int(cpu_threads))
        if runtime_backend is not None:
            maxpool_runtime = runtime_backend
        if maxpool_runtime not in {"iron", "xrt"}:
            raise ValueError("runtime backend must be 'iron' or 'xrt'")
        self.runtime_backend = maxpool_runtime
        self.maxpool_runtime = maxpool_runtime
        self._torch_initialized = False
        self.codegen = build_codegen_plan(
            model, strict=True, optimize_small_m=self.optimize_small_m
        )
        self.conv_plans = {
            plan.node_index: plan
            for plan in plan_all_convs(model, optimize_small_m=self.optimize_small_m)
        }
        self.nodes_by_output = {name: node for node in self.nodes for name in node.output}
        self.consumers_by_input: dict[str, list[int]] = {}
        for index, node in enumerate(self.nodes):
            for name in node.input:
                if name:
                    self.consumers_by_input.setdefault(name, []).append(index)
        self._fused_relu_for_conv: dict[int, tuple[int, str]] = {}
        graph_outputs = {str(value.name) for value in getattr(model.graph, "output", ())}
        for conv_index, plan in self.conv_plans.items():
            if not plan.fused_relu:
                continue
            conv_output = self.nodes[conv_index].output[0]
            if conv_output in graph_outputs:
                continue
            frontier = [conv_output]
            seen: set[str] = set()
            relu_candidates: list[tuple[int, str]] = []
            while frontier:
                value = frontier.pop()
                if value in seen:
                    continue
                seen.add(value)
                for consumer_index in self.consumers_by_input.get(value, ()):
                    consumer = self.nodes[consumer_index]
                    if consumer.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                        frontier.extend(consumer.output)
                    elif consumer.op_type == "Relu":
                        relu_candidates.append((consumer_index, consumer.output[0]))
            if len(relu_candidates) == 1:
                self._fused_relu_for_conv[conv_index] = relu_candidates[0]
        self._dq_only_used_by_conv: set[int] = set()
        for index, node in enumerate(self.nodes):
            if node.op_type != "DequantizeLinear" or not node.output:
                continue
            consumers = [
                (consumer, input_index)
                for consumer in self.nodes
                for input_index, input_name in enumerate(consumer.input)
                if input_name == node.output[0]
            ]
            if consumers and all(
                consumer.op_type == "Conv" and input_index in (0, 1)
                for consumer, input_index in consumers
            ):
                self._dq_only_used_by_conv.add(index)
        self.specs = [entry for entry in manifest.get("kernels", []) if entry.get("compiled_artifact")]
        self.operation_specs = {
            int(entry["node_index"]): entry
            for entry in manifest.get("operation_kernels", [])
            if entry.get("compiled_artifact")
            or entry.get("status") == "zero_copy_device_view"
        }
        if host_maxpool:
            # Tiny pooling is cheaper on the host than a separate xclbin (each switch between
            # xclbins costs ~0.75 ms on this device).
            self.operation_specs = {
                index: entry for index, entry in self.operation_specs.items()
                if entry.get("op_type") != "MaxPool"
            }
        graph_outputs = {str(value.name) for value in getattr(model.graph, "output", ())}
        self._host_qadd_fusions: dict[int, dict[str, Any]] = {}
        self._host_qadd_input_dqs: set[int] = set()

        def single_consumer(value: str, op_type: str) -> int | None:
            users = self.consumers_by_input.get(value, ())
            if len(users) != 1 or self.nodes[users[0]].op_type != op_type:
                return None
            return users[0]

        def scalar_qparams(node: Any) -> tuple[float, int] | None:
            if len(node.input) < 3 or node.input[1] not in self.arrays or node.input[2] not in self.arrays:
                return None
            scale = np.asarray(self.arrays[node.input[1]]).reshape(-1)
            zero = np.asarray(self.arrays[node.input[2]]).reshape(-1)
            if scale.size != 1 or zero.size != 1 or scale.dtype.kind != "f":
                return None
            return float(scale[0]), int(zero[0])

        self._maxpool_qdq_fusions: dict[int, dict[str, Any]] = {}
        self._maxpool_input_dqs: set[int] = set()
        if maxpool_uint8_artifact is not None:
            xclbin, insts = maxpool_uint8_artifact
            if not Path(xclbin).is_file() or not Path(insts).is_file():
                raise ValueError("uint8 MaxPool xclbin and instruction stream must both exist")
            for pool_index, pool_node in enumerate(self.nodes):
                if pool_node.op_type != "MaxPool" or pool_index not in self.operation_specs:
                    continue
                input_dq = self.nodes_by_output.get(str(pool_node.input[0]))
                if input_dq is None or input_dq.op_type != "DequantizeLinear":
                    continue
                input_q = self.nodes_by_output.get(str(input_dq.input[0]))
                if input_q is None or input_q.op_type != "QuantizeLinear":
                    continue
                relu = self.nodes_by_output.get(str(input_q.input[0]))
                if relu is None or relu.op_type != "Relu":
                    continue
                input_qparams = scalar_qparams(input_q)
                input_dqparams = scalar_qparams(input_dq)
                if (input_qparams is None or input_qparams != input_dqparams
                        or np.asarray(self.arrays[input_q.input[2]]).dtype != np.uint8
                        or input_qparams[1] <= 0
                        or single_consumer(str(input_q.output[0]), "DequantizeLinear") is None
                        or single_consumer(str(input_dq.output[0]), "MaxPool") != pool_index):
                    continue
                output_q_index = single_consumer(str(pool_node.output[0]), "QuantizeLinear")
                if output_q_index is None:
                    continue
                output_q = self.nodes[output_q_index]
                output_dq_index = single_consumer(str(output_q.output[0]), "DequantizeLinear")
                if (output_dq_index is None or scalar_qparams(output_q) != input_qparams
                        or np.asarray(self.arrays[output_q.input[2]]).dtype != np.uint8):
                    continue
                output_dq = self.nodes[output_dq_index]
                if scalar_qparams(output_dq) != input_qparams:
                    continue
                self._maxpool_qdq_fusions[pool_index] = {
                    "input_raw": str(input_q.output[0]),
                    "input_dq_index": self.nodes.index(input_dq),
                    "output_q_index": output_q_index,
                    "output_dq_index": output_dq_index,
                    "scale": input_qparams[0],
                    "zero_point": input_qparams[1],
                }
                self._maxpool_input_dqs.add(self.nodes.index(input_dq))
                self.operation_specs[pool_index]["compiled_artifact"] = {
                    "xclbin": str(xclbin), "insts": str(insts),
                }
            if not self._maxpool_qdq_fusions:
                raise ValueError("uint8 MaxPool fusion requires a Relu->Q->DQ input and matching pool Q->DQ output")

        # Fold the common quantized residual sequence into one host executor
        # step when no native quantized Add artifact is available.
        for add_index, add_node in enumerate(self.nodes):
            if (add_node.op_type != "Add" or add_index in self.operation_specs
                    or len(add_node.input) != 2 or not add_node.output
                    or add_node.output[0] in graph_outputs):
                continue
            input_dq_indices = []
            input_qparams = []
            valid = True
            for input_name in add_node.input:
                if input_name in graph_outputs:
                    valid = False
                    break
                dq_node = self.nodes_by_output.get(input_name)
                if dq_node is None or dq_node.op_type != "DequantizeLinear":
                    valid = False
                    break
                dq_index = self.nodes.index(dq_node)
                dq_users = self.consumers_by_input.get(input_name, ())
                if not dq_users or any(
                    user != add_index
                    and not (
                        self.nodes[user].op_type == "Conv"
                        and any(name == input_name and input_index in (0, 1)
                                for input_index, name in enumerate(self.nodes[user].input))
                    )
                    for user in dq_users
                ):
                    valid = False
                    break
                qparams = scalar_qparams(dq_node)
                if qparams is None:
                    valid = False
                    break
                input_dq_indices.append(dq_index)
                input_qparams.append(qparams)
            if not valid:
                continue
            relu_index = single_consumer(add_node.output[0], "Relu")
            if relu_index is None or self.nodes[relu_index].output[0] in graph_outputs:
                continue
            relu_node = self.nodes[relu_index]
            quantize_index = single_consumer(relu_node.output[0], "QuantizeLinear")
            if quantize_index is None:
                continue
            quantize_node = self.nodes[quantize_index]
            quantize_params = scalar_qparams(quantize_node)
            if quantize_params is None or np.asarray(self.arrays[quantize_node.input[2]]).dtype != np.uint8:
                continue
            dequantize_index = single_consumer(quantize_node.output[0], "DequantizeLinear")
            if dequantize_index is None:
                continue
            dequantize_node = self.nodes[dequantize_index]
            dequantize_params = scalar_qparams(dequantize_node)
            if (dequantize_params is None
                    or np.asarray(self.arrays[dequantize_node.input[2]]).dtype != np.uint8):
                continue
            self._host_qadd_fusions[add_index] = {
                "input_dq_indices": tuple(input_dq_indices),
                "input_qparams": tuple(input_qparams),
                "relu_index": relu_index,
                "relu_output": relu_node.output[0],
                "quantize_index": quantize_index,
                "quantize_output": quantize_node.output[0],
                "quantize_params": quantize_params,
                "dequantize_index": dequantize_index,
                "dequantize_output": dequantize_node.output[0],
                "dequantize_params": dequantize_params,
            }
            self._host_qadd_input_dqs.update(input_dq_indices)
        self._executed = {"xdna_conv": 0, "xdna_maxpool": 0, "qdq_maxpool_fusions": 0, "xdna_quantized_add_relu": 0, "device_resident_pool_outputs": 0, "device_view_ops": 0, "cpu_conv": 0, "host_ops": 0, "skipped_conv_dq": 0, "fused_relu": 0, "fused_bottleneck": 0, "device_resident_handoffs": 0, "device_edge_readbacks": 0}
        self._profile: dict[str, float] = {}
        self._device_readback_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._conv_times: list[dict[str, Any]] = []
        self._fused_times: list[dict[str, Any]] = []
        self._capture_outputs: dict[str, np.ndarray] = {}
        self._capture_enabled = capture_outputs
        self._workspace_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._maxpool_workspace_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._qadd_workspace_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        # ONNX weights are constants. Keep their padded GEMM layout so steady
        # state inference only packs the changing activation matrix.
        self._packed_weight_cache: dict[tuple[Any, ...], tuple[np.ndarray, ...]] = {}
        self._cpu_weight_cache: dict[int, np.ndarray] = {}
        self._constant_dequant: dict[int, np.ndarray] = {}
        self._cpu_numpy_matrix_cache: dict[tuple[int, int], tuple[np.ndarray, bool]] = {}
        self._torch_weight_cache: dict[int, Any] = {}
        self._torch_int8_weight_cache: dict[int, tuple[Any, ...]] = {}
        fused_specs = list(fused_blocks or ())
        self.fused_stage_blocked = fused_stage_blocked or bool(fused_body_groups) or bool(device_network_stages)
        self.fused_body_groups = fused_body_groups
        self.fused_body_rt = fused_body_rt
        self.device_network_stages = device_network_stages
        self.layer_engine = layer_engine
        self.fused_body_chunk_caps = fused_body_chunk_caps or {}
        stage_specs = list(fused_stages or ())
        parallel_specs = list(parallel_projection_blocks or ())
        if any((fused_block_prefix, fused_block_xclbin, fused_block_insts)):
            if not all((fused_block_prefix, fused_block_xclbin, fused_block_insts)):
                raise ValueError("fused block requires its node prefix, xclbin, and instruction stream")
            fused_specs.append((fused_block_prefix, fused_block_xclbin, fused_block_insts))
        bottleneck_plans = {block.prefix: block for block in plan_bottleneck_blocks(model)}
        prepared_blocks: dict[str, tuple[Any, dict[str, Any], set[int], str, str]] = {}
        prepared_stages: dict[str, tuple[list[tuple[Any, dict[str, Any]]], set[int], str, str]] = {}
        parallel_prefixes = {prefix for prefix, _xclbin, _insts in parallel_specs}
        for prefix, xclbin, insts in [*fused_specs, *parallel_specs]:
            if prefix in prepared_blocks:
                raise ValueError(f"fused block {prefix!r} was specified more than once")
            block = bottleneck_plans.get(prefix)
            if block is None:
                raise ValueError(f"no bottleneck block found for prefix {prefix!r}")
            if not Path(xclbin).is_file() or not Path(insts).is_file():
                raise ValueError("fused block xclbin and instruction stream must exist")
            binding = bind_fused_bottleneck(model, block)
            if prefix in parallel_prefixes and (
                *binding["chunk_counts"], binding["skip_chunk_count"]
            ) != (1, 1, 1, 1):
                raise ValueError(
                    f"parallel projection block {prefix!r} requires one weight chunk per Conv"
                )
            prepared_blocks[prefix] = (block, binding, set(binding["covered_nodes"]), str(xclbin), str(insts))

        for prefixes, xclbin, insts in stage_specs:
            if not 1 <= len(prefixes) <= 16:
                raise ValueError("linked fused stage requires one to sixteen block prefixes")
            if not Path(xclbin).is_file() or not Path(insts).is_file():
                raise ValueError("fused stage xclbin and instruction stream must exist")
            stage_blocks = []
            covered: set[int] = set()
            previous_binding = None
            for prefix in prefixes:
                if prefix in prepared_blocks or any(
                    prefix == block.prefix
                    for items, _covered, _stage_xclbin, _stage_insts in prepared_stages.values()
                    for block, _binding in items
                ):
                    raise ValueError(f"fused stage block {prefix!r} overlaps another fused artifact")
                block = bottleneck_plans.get(prefix)
                if block is None:
                    raise ValueError(f"no bottleneck block found for prefix {prefix!r}")
                binding = bind_fused_bottleneck(
                    model, block, blocked=self.fused_stage_blocked,
                    max_chunk=self.fused_body_chunk_caps.get(prefix) or None,
                )
                if previous_binding is not None and (
                    previous_binding["output_shape"] != binding["input_shape"]
                    or previous_binding["output_raw_name"] != binding["input_raw_name"]
                ):
                    raise ValueError(f"linked stage boundary before {prefix!r} is not a direct compatible QDQ edge")
                previous_binding = binding
                stage_blocks.append((block, binding))
                covered.update(binding["covered_nodes"])
            first_prefix = prefixes[0]
            prepared_stages[first_prefix] = (stage_blocks, covered, str(xclbin), str(insts))

        # Strix Halo supports 16 simultaneous hardware contexts. Reserve those
        # slots across pooling, fused blocks, and any remaining XDNA Conv shapes.
        self.context_cache_limit = min(16, max(1, int(os.environ.get("XRT_CONTEXT_CACHE_SIZE", "16"))))
        self.context_budget_fallback_blocks: list[str] = []
        self._forced_cpu_convs: set[int] = set()
        active_prefixes = set(prepared_blocks)
        active_stages = set(prepared_stages)
        pool_contexts = {
            str(item["compiled_artifact"]["xclbin"])
            for item in self.operation_specs.values()
            if item.get("op_type") == "MaxPool"
        }

        def conv_contexts(covered: set[int]) -> dict[str, list[int]]:
            contexts: dict[str, list[int]] = {}
            for conv_index, plan in self.conv_plans.items():
                if conv_index in covered or conv_index in self._forced_cpu_convs:
                    continue
                if self.cpu_small_m and plan.gemm_shape[0] <= self.cpu_small_m:
                    continue
                artifact = self._artifact(plan)
                contexts.setdefault(str(artifact["compiled_artifact"]["xclbin"]), []).append(conv_index)
            return contexts

        while True:
            covered = {
                index
                for prefix in active_prefixes
                for index in prepared_blocks[prefix][2]
            }
            covered.update(
                index for key in active_stages for index in prepared_stages[key][1]
            )
            remaining_conv_contexts = conv_contexts(covered)
            remaining_qadd_contexts = {
                str(item["compiled_artifact"]["xclbin"])
                for index, item in self.operation_specs.items()
                if item.get("op_type") == "Add"
                and item.get("quantization")
                and item.get("compiled_artifact")
                and not set(item["quantization"].get("fused_node_indices") or (index,)).issubset(covered)
            }
            active_contexts = pool_contexts | {
                prepared_blocks[prefix][3] for prefix in active_prefixes
            } | {prepared_stages[key][2] for key in active_stages} | set(remaining_conv_contexts) | remaining_qadd_contexts
            if len(active_contexts) <= self.context_cache_limit:
                break

            if remaining_conv_contexts:
                # Free the least-work Conv specialization first; it remains
                # correct on CPU and costs fewer contexts than evicting a hot
                # fused block on every iteration.
                conv_xclbin, conv_indices = min(
                    remaining_conv_contexts.items(),
                    key=lambda item: sum(math.prod(self.conv_plans[index].gemm_shape) for index in item[1]),
                )
                self._forced_cpu_convs.update(conv_indices)
                continue

            if not active_prefixes:
                if not active_stages:
                    raise RuntimeError("compiled operator kernels exceed the XRT context limit")
                demote_stage = min(
                    active_stages,
                    key=lambda key: (
                        sum(math.prod(self.conv_plans[index].gemm_shape) for index in prepared_stages[key][1] if index in self.conv_plans),
                        key,
                    ),
                )
                self._forced_cpu_convs.update(index for index in prepared_stages[demote_stage][1] if index in self.conv_plans)
                active_stages.remove(demote_stage)
                self.context_budget_fallback_blocks.extend(item[0].prefix for item in prepared_stages[demote_stage][0])
                continue
            demote = min(
                active_prefixes,
                key=lambda prefix: (
                    sum(
                        math.prod(self.conv_plans[index].gemm_shape)
                        for index in prepared_blocks[prefix][2]
                        if index in self.conv_plans
                    ),
                    prefix,
                ),
            )
            self._forced_cpu_convs.update(
                index for index in prepared_blocks[demote][2] if index in self.conv_plans
            )
            active_prefixes.remove(demote)
            self.context_budget_fallback_blocks.append(demote)

        fused_specs = [
            (prefix, prepared_blocks[prefix][3], prepared_blocks[prefix][4])
            for prefix in prepared_blocks if prefix in active_prefixes
        ]
        self._fused_blocks: dict[str, dict[str, Any]] = {}
        self._fused_stages: dict[str, dict[str, Any]] = {}
        self._fused_nodes: dict[int, tuple[str, bool]] = {}
        self._fused_stage_nodes: dict[int, tuple[str, bool]] = {}
        self._fused_input_handoffs: dict[int, str] = {}
        if fused_specs and self.runtime_backend == "iron":
            import aie.iron as iron

        if active_stages:
            import aie.iron as iron

        for prefix, xclbin, insts in fused_specs:
            block, binding, covered, _xclbin, _insts = prepared_blocks[prefix]
            start_index = min(covered)
            if any(index in self._fused_nodes for index in covered):
                raise ValueError(f"fused block {prefix!r} overlaps another fused block")
            input_shape = binding["input_shape"]
            output_shape = binding["output_shape"]
            input_count = int(np.prod(input_shape))
            output_count = int(np.prod(output_shape))
            if self.runtime_backend == "xrt":
                try:
                    from xdna_xrt_runtime import load_kernel
                except ImportError:
                    from .xdna_xrt_runtime import load_kernel
                block_kernel = load_kernel(str(xclbin), str(insts))
                block_input = block_kernel.tensor((input_count,), np.int8, 3)
            else:
                block_kernel = self._kernel(str(xclbin), str(insts))
                block_input = iron.tensor(
                    np.zeros(input_count, dtype=np.int8), dtype=np.int8, device="npu"
                )
            block_parameters = None
            main_parameters = skip_parameters = None
            if prefix in parallel_prefixes:
                if self.runtime_backend == "xrt":
                    main_parameters = block_kernel.tensor((binding["main_params"].size,), np.uint8, 4)
                    skip_parameters = block_kernel.tensor((binding["skip_params"].size,), np.uint8, 5)
                    main_parameters.load_constant(binding["main_params"].reshape(-1))
                    skip_parameters.load_constant(binding["skip_params"].reshape(-1))
                else:
                    main_parameters = iron.tensor(binding["main_params"], dtype=np.uint8, device="npu")
                    skip_parameters = iron.tensor(binding["skip_params"], dtype=np.uint8, device="npu")
            else:
                if self.runtime_backend == "xrt":
                    block_parameters = block_kernel.tensor((binding["params"].size,), np.uint8, 4)
                    block_parameters.load_constant(binding["params"].reshape(-1))
                else:
                    block_parameters = iron.tensor(binding["params"], dtype=np.uint8, device="npu")
            if self.runtime_backend == "xrt":
                output_index = 6 if prefix in parallel_prefixes else 5
                block_output = block_kernel.tensor((output_count,), np.int8, output_index)
            else:
                block_output = iron.zeros(output_count, dtype=np.int8, device="npu")
            self._fused_blocks[prefix] = {
                "binding": binding,
                "start": start_index,
                "input": block_input,
                "parameters": block_parameters,
                "main_parameters": main_parameters,
                "skip_parameters": skip_parameters,
                "parallel_projection": prefix in parallel_prefixes,
                "output": block_output,
                "kernel": block_kernel,
                "direct_xrt": self.runtime_backend == "xrt",
                "xclbin": str(xclbin),
            }
            for index in covered:
                self._fused_nodes[index] = (prefix, index == start_index)
        for first_prefix in active_stages:
            blocks, covered, xclbin, insts = prepared_stages[first_prefix]
            if any(index in self._fused_nodes or index in self._fused_stage_nodes for index in covered):
                raise ValueError(f"linked fused stage {first_prefix!r} overlaps another fused region")
            bindings = [binding for _block, binding in blocks]
            input_count = int(np.prod(bindings[0]["input_shape"]))
            output_count = int(np.prod(bindings[-1]["output_shape"]))
            input_tensor = iron.tensor(np.zeros(input_count, dtype=np.int8), dtype=np.int8, device="npu")
            network = None
            if self.device_network_stages:
                network = (
                    self._prepare_layer_engine(model, first_prefix, covered)
                    if self.layer_engine else self._prepare_device_network(model, first_prefix, covered)
                )
                covered = network["covered"]
                input_tensor, output_tensor = network["input"], network["output"]
                parameter_tensor, extra_tensor = network["parameters"], network["scratch"]
                tap_tensor = None
            elif self.fused_stage_blocked:
                try:
                    from blocked_stage import pack_blocked_params, pack_rt_params
                except ImportError:
                    from .blocked_stage import pack_blocked_params, pack_rt_params
                if self.fused_body_rt:
                    # runtime-shaped kernels: every weight slot starts with a geometry descriptor
                    stage_params = [pack_rt_params(binding) for binding in bindings]
                else:
                    stage_params = [pack_blocked_params(binding, header=bool(self.fused_body_groups)) for binding in bindings]
            else:
                stage_params = [binding["params"] for binding in bindings]
            if network is None:
                parameter_tensor = iron.tensor(np.concatenate(stage_params), dtype=np.uint8, device="npu")
                output_tensor = iron.zeros(output_count, dtype=np.int8, device="npu")
                tap_tensor = None
                if os.environ.get("ONNXSIM_XDNA_STAGE_TAP"):
                    # Debug: artifact built with --tap drains block 0's output as a 4th argument.
                    first_out = bindings[0]["output_shape"]
                    tap_tensor = iron.zeros(int(np.prod(first_out)), dtype=np.int8, device="npu")
                extra_tensor = None
                if self.fused_body_groups:
                    # Whole-body artifact: the 4th argument is a DDR scratch buffer holding
                    # every intermediate block boundary (all but the last block's output).
                    scratch = sum(int(np.prod(binding["output_shape"])) for binding in bindings[:-1])
                    extra_tensor = iron.zeros(max(scratch, 1), dtype=np.int8, device="npu")
            self._fused_stages[first_prefix] = {
                "network": network,
                "extra": extra_tensor,
                "tap": tap_tensor,
                "blocks": blocks, "bindings": bindings, "input": input_tensor,
                "parameters": parameter_tensor, "output": output_tensor,
                "kernel": self._kernel(xclbin, insts), "xclbin": xclbin,
            }
            first_index = min(covered)
            for index in covered:
                self._fused_stage_nodes[index] = (first_prefix, index == first_index)
        self._plan_fused_input_handoffs()
        # Constant-only DequantizeLinear nodes (weights/biases) are evaluated once here and then
        # skipped in the run loop: visiting ~50 of them per inference cost more than the maths.
        self._static_nodes: set[int] = set()
        for index, node in enumerate(self.nodes):
            if (
                node.op_type == "DequantizeLinear"
                and index not in self._fused_stage_nodes
                and index not in self._fused_nodes
                and all(name in self.arrays for name in node.input if name)
                and len(node.input) >= 3
            ):
                attrs = _attrs(node)
                self.arrays[node.output[0]] = _dequantize(
                    self.arrays[node.input[0]], self.arrays[node.input[1]], self.arrays[node.input[2]],
                    int(attrs.get("axis", 1)),
                )
                self._static_nodes.add(index)

    def runtime_subgraph_report(self) -> list[dict[str, Any]]:
        """Describe actual executor assignment over each planned graph region."""
        node_executor: dict[int, str] = {}
        node_dispatch: dict[int, str] = {}
        for index, node in enumerate(self.nodes):
            if node.op_type == "Constant":
                node_executor[index] = "constant_data"
        for index, (prefix, _is_start) in self._fused_stage_nodes.items():
            node_executor[index] = f"fused_stage:{prefix}"
            node_dispatch[index] = f"stage:{prefix}"
        for index, (prefix, _is_start) in self._fused_nodes.items():
            node_executor[index] = f"fused_bottleneck:{prefix}"
            node_dispatch[index] = f"bottleneck:{prefix}"
        for index, spec in self.operation_specs.items():
            artifact = spec.get("compiled_artifact")
            quant = spec.get("quantization") or {}
            if artifact:
                executor = f"native_{spec.get('op_type', self.nodes[index].op_type)}:{artifact.get('key', artifact.get('xclbin', 'artifact'))}"
                for fused_index in quant.get("fused_node_indices", (index,)):
                    fused_index = int(fused_index)
                    if fused_index not in node_executor:
                        node_executor[fused_index] = executor
                        node_dispatch[fused_index] = f"operation:{index}"
            elif spec.get("status") == "zero_copy_device_view":
                node_executor.setdefault(index, "device_view_if_resident")
        for index, plan in self.conv_plans.items():
            if index in node_executor:
                continue
            if index in self._forced_cpu_convs or (self.cpu_small_m and plan.gemm_shape[0] <= self.cpu_small_m):
                node_executor[index] = "host_small_conv"
                continue
            try:
                artifact = self._artifact(plan)
                node_executor[index] = f"native_conv:{artifact['key']}"
                node_dispatch[index] = f"conv:{index}"
            except (KeyError, RuntimeError):
                node_executor[index] = "host_conv_missing_artifact"
        for conv_index, (relu_index, _output) in self._fused_relu_for_conv.items():
            if conv_index in self.conv_plans and node_executor.get(conv_index, "").startswith("native_conv:"):
                node_executor.setdefault(relu_index, node_executor[conv_index])

        producer_by_value = {
            str(output): index
            for index, node in enumerate(self.nodes)
            for output in node.output if output
        }
        reports: list[dict[str, Any]] = []
        for region in self.codegen.graph_regions:
            instruction_records = []
            segments: list[dict[str, Any]] = []
            for instruction in region.instructions:
                index = int(instruction["node_index"])
                executor = node_executor.get(index, "host")
                record = {
                    "node_index": index,
                    "op_type": str(instruction["op_type"]),
                    "executor": executor,
                    "dispatch_unit": node_dispatch.get(index),
                }
                instruction_records.append(record)
                if segments and segments[-1]["executor"] == executor:
                    segments[-1]["node_indices"].append(index)
                    segments[-1]["op_types"].append(str(instruction["op_type"]))
                else:
                    segments.append({
                        "executor": executor,
                        "node_indices": [index],
                        "op_types": [str(instruction["op_type"])],
                    })
            native = {
                record["dispatch_unit"] for record in instruction_records
                if record["dispatch_unit"] is not None
            }
            has_host = any(record["executor"].startswith("host") for record in instruction_records)
            if not native:
                status = "host_only"
            elif has_host:
                status = "hybrid_region"
            elif len(native) == 1:
                status = "single_device_dispatch"
            else:
                status = "multi_dispatch_device_region"
            region_indices = {int(item["node_index"]) for item in region.instructions}
            runtime_boundaries: dict[str, set[tuple[int, int, str, str]]] = {}
            for instruction in region.instructions:
                target = int(instruction["node_index"])
                target_executor = node_executor.get(target, "host")
                target_is_device = target_executor.startswith(("native_", "fused_stage:", "fused_bottleneck:"))
                for value in instruction["inputs"]:
                    source = producer_by_value.get(str(value))
                    if source is None or source not in region_indices:
                        continue
                    source_executor = node_executor.get(source, "host")
                    source_is_device = source_executor.startswith(("native_", "fused_stage:", "fused_bottleneck:"))
                    if source_is_device != target_is_device:
                        runtime_boundaries.setdefault(str(value), set()).add(
                            (source, target, source_executor, target_executor)
                        )
            reports.append({
                "region_id": region.region_id,
                "planned_nodes": list(region.node_indices),
                "planned_op_types": list(region.op_types),
                "input_values": list(region.input_values),
                "constant_values": list(region.constant_values),
                "output_values": list(region.output_values),
                "peak_live_bytes_known": region.peak_live_bytes,
                "device_lowering_gaps": list(region.device_lowering_gaps),
                "status": status,
                "native_dispatch_count": len(native),
                "internal_device_host_boundary_values": sorted(runtime_boundaries),
                "internal_device_host_boundary_count": sum(len(edges) for edges in runtime_boundaries.values()),
                "instruction_assignment": instruction_records,
                "executor_segments": segments,
            })
        return reports

    def _plan_fused_input_handoffs(self) -> None:
        """Find adjacent fused blocks whose raw QDQ edge can stay on device."""
        for target_prefix, target in self._fused_blocks.items():
            target_binding = target["binding"]
            target_block = target_binding["block"]
            target_conv = self.nodes[target_block.main_conv_indices[0]]
            input_dq_name = str(target_conv.input[0])
            input_dq = self.nodes_by_output.get(input_dq_name)
            if input_dq is None or input_dq.op_type != "DequantizeLinear":
                continue
            if self.consumers_by_input.get(input_dq_name) != [(target_block.main_conv_indices[0], 0)]:
                continue
            dq_scale = float(np.asarray(self.arrays[str(input_dq.input[1])]).reshape(-1)[0])
            dq_zero = int(np.asarray(self.arrays[str(input_dq.input[2])]).reshape(-1)[0])
            dq_index = next(i for i, node in enumerate(self.nodes) if node is input_dq)
            for source_prefix, source in self._fused_blocks.items():
                if source_prefix == target_prefix:
                    continue
                source_binding = source["binding"]
                if source_binding["output_raw_name"] != target_binding["input_raw_name"]:
                    continue
                if tuple(source_binding["output_shape"]) != tuple(target_binding["input_shape"]):
                    continue
                if (source_binding["output_zero_point"] != dq_zero
                        or not math.isclose(float(source_binding["output_scale"]), dq_scale, rel_tol=1e-6)
                        or dq_zero != 128):
                    continue
                self._fused_input_handoffs[dq_index] = source_prefix
                break

    def _host_value(self, value: Any) -> np.ndarray:
        if not isinstance(value, _DeviceValue):
            return np.asarray(value)
        cache_key = (id(value.tensor), value.shape, value.layout)
        raw = self._device_readback_cache.get(cache_key)
        if raw is None:
            started = time.perf_counter()
            device_data = value.tensor.numpy()
            if value.layout == "nhwc":
                n, c, h, w = value.shape
                raw = device_data.view(np.uint8).reshape(n, h, w, c).transpose(0, 3, 1, 2).copy()
            elif value.layout == "nchw":
                raw = device_data.reshape(value.shape).copy()
            elif value.layout == "flat_hwc":
                raw = device_data.view(np.uint8).reshape(value.shape).copy()
            else:
                raise ValueError(f"unsupported device tensor layout {value.layout!r}")
            self._device_readback_cache[cache_key] = raw
            self._executed["device_edge_readbacks"] += 1
            self._profile["device_edge_readback_ms"] = self._profile.get("device_edge_readback_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        if value.as_real:
            return _dequantize(raw, np.asarray([value.scale], dtype=np.float32), np.asarray([value.zero_point], dtype=np.uint8), axis=1)
        return raw

    def _qdq_device_view(self, node: Any, values: dict[str, Any]) -> _DeviceValue | None:
        """Forward unchanged scalar uint8 Q/DQ edges without touching host memory."""
        if len(node.input) < 3:
            return None
        value = values.get(str(node.input[0]))
        if not isinstance(value, _DeviceValue) or value.zero_point != 128:
            return None
        if str(node.input[1]) not in self.arrays or str(node.input[2]) not in self.arrays:
            return None
        scale = np.asarray(self.arrays[str(node.input[1])])
        zero = np.asarray(self.arrays[str(node.input[2])])
        if scale.size != 1 or zero.size != 1 or zero.dtype != np.uint8 or int(zero.reshape(-1)[0]) != 128:
            return None
        if not math.isclose(float(scale.reshape(-1)[0]), value.scale, rel_tol=1e-7, abs_tol=0.0):
            return None
        is_quantize = node.op_type == "QuantizeLinear"
        if value.as_real != is_quantize:
            return None
        return _DeviceValue(
            value.tensor, value.shape, value.scale, value.zero_point,
            as_real=not is_quantize, producer=f"{node.op_type.lower()}:{node.name or node.output[0]}",
            layout=value.layout,
        )

    def _device_view(self, index: int, node: Any, values: dict[str, Any]) -> _DeviceValue | None:
        """Apply proven identity/view operations to a resident device edge."""
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            return self._qdq_device_view(node, values)
        spec = self.operation_specs.get(index)
        if spec is None or spec.get("status") != "zero_copy_device_view":
            return None
        if node.op_type == "Mul":
            params = spec.get("parameters") or {}
            if float(params.get("scalar", float("nan"))) != 1.0:
                return None
            value = next((values.get(str(name)) for name in node.input
                          if isinstance(values.get(str(name)), _DeviceValue)), None)
            if value is None or tuple(spec.get("input_shapes", [value.shape])[0]) != value.shape:
                return None
            return _DeviceValue(value.tensor, value.shape, value.scale, value.zero_point,
                                value.as_real, f"device-view:{node.op_type}", value.layout)
        value = values.get(str(node.input[0])) if node.input else None
        if not isinstance(value, _DeviceValue):
            return None
        if node.op_type == "GlobalAveragePool":
            if len(value.shape) != 4 or value.shape[-2:] != (1, 1):
                return None
            output_shape = tuple(spec["output_shapes"][0])
            if output_shape != value.shape:
                return None
        elif node.op_type == "Flatten":
            if len(value.shape) != 4 or value.shape[-2:] != (1, 1) or value.layout != "nhwc":
                return None
            output_shape = tuple(spec["output_shapes"][0])
            if output_shape != (value.shape[0], value.shape[1]):
                return None
            return _DeviceValue(value.tensor, output_shape, value.scale, value.zero_point,
                                value.as_real, f"device-view:{node.op_type}", "flat_hwc")
        else:
            return None
        return _DeviceValue(value.tensor, output_shape, value.scale, value.zero_point,
                            value.as_real, f"device-view:{node.op_type}", value.layout)

    def _quant_source(self, value_name: str, values: dict[str, np.ndarray]) -> tuple[np.ndarray, float, int]:
        dq = self.nodes_by_output.get(value_name)
        if dq is None or dq.op_type != "DequantizeLinear":
            raise ValueError(f"Conv tensor {value_name!r} is not produced by DequantizeLinear")
        raw_name, scale_name, zero_name = dq.input[:3]
        raw = values[raw_name]
        if isinstance(raw, _DeviceValue):
            raw = self._host_value(raw)
        scale = np.asarray(self.arrays[scale_name], dtype=np.float32).reshape(-1)
        zero = np.asarray(self.arrays[zero_name]).reshape(-1)
        if scale.size != 1 or zero.size != 1:
            raise ValueError("Conv activation and weight quantization must be per-tensor")
        return raw, float(scale[0]), int(zero[0])

    def _artifact(self, plan: Any) -> dict[str, Any]:
        tm, tk, tn = plan.tile
        m, k, n = plan.gemm_shape
        cm = max(tm * 8, math.ceil(m / (tm * 8)) * tm * 8)
        ck = math.ceil(k / tk) * tk
        matching = [
            item for item in self.specs
            if tuple(item["tile"]) == plan.tile
            and tuple(item["compiled_shape"])[0] == cm
            and tuple(item["compiled_shape"])[1] == ck
            and tuple(item["compiled_shape"])[2] >= n
        ]
        if not matching:
            raise RuntimeError(f"no compiled XDNA artifact matches Conv {plan.node_name} {plan.gemm_shape}")
        # The NPU2 runtime can retain 16 xclbin contexts. This ResNet has 20
        # shapes, so share four selected output widths with the next wider
        # compiled artifact and avoid repeatedly evicting/reloading contexts.
        shared_n = {
            (256, 64): 128,
            (512, 128): 256,
            (1024, 256): 512,
            (512, 1024): 2048,
        }.get((k, n)) if self.optimize_small_m else None
        if shared_n is not None:
            shared = [item for item in matching if int(item["compiled_shape"][2]) == shared_n]
            if shared:
                return shared[0]
        return min(matching, key=lambda item: tuple(item["compiled_shape"])[2])

    @staticmethod
    @lru_cache(maxsize=32)
    def _kernel(xclbin: str, insts: str) -> Any:
        from aie.utils import NPUKernel
        return NPUKernel(xclbin, insts)

    def _pool_kernel(self, xclbin: str, insts: str) -> Any:
        return self._artifact_kernel(xclbin, insts)

    def _artifact_kernel(self, xclbin: str, insts: str) -> Any:
        if self.maxpool_runtime == "xrt":
            try:
                from xdna_xrt_runtime import load_kernel
            except ImportError:
                from .xdna_xrt_runtime import load_kernel
            return load_kernel(xclbin, insts)
        return self._kernel(xclbin, insts)

    def _pool_tensor(self, xclbin: str, insts: str, shape: tuple[int, ...], dtype: Any, argument_index: int) -> Any:
        if self.maxpool_runtime == "xrt":
            return self._pool_kernel(xclbin, insts).tensor(shape, dtype, argument_index)
        import aie.iron as iron
        return iron.tensor(shape, dtype=dtype, device="npu")

    def _run_maxpool_kernel(self, index: int, x: np.ndarray) -> _DeviceValue:
        """Upload padded NCHW input and retain the pooling result in XRT memory."""
        if self.maxpool_runtime == "iron":
            import aie.iron as iron
            from aie.iron.device import from_name
            iron.set_current_device(from_name("npu2", n_cols=None))
        spec = self.operation_specs[index]
        params = spec["parameters"]
        artifact = spec["compiled_artifact"]
        started = time.perf_counter()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"compiled MaxPool node {index} requires batch-one NCHW input")
        pads = tuple(int(value) for value in params["pads"])
        pt, pl, pb, pr = pads
        padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=-np.inf)
        expected = (1, int(params["channels"]), int(params["input_height"]), int(params["input_width"]))
        if padded.shape != expected:
            raise ValueError(f"compiled MaxPool node {index} expects padded input {expected}, got {padded.shape}")
        output_shape = (1, int(params["channels"]), int(params["output_height"]), int(params["output_width"]))
        workspace_key = (str(artifact["xclbin"]), str(artifact["insts"]), expected, output_shape)
        workspaces = self._maxpool_workspace_cache.get(workspace_key)
        if workspaces is None:
            workspaces = (
                self._pool_tensor(str(artifact["xclbin"]), str(artifact["insts"]), (math.prod(expected),), np.float32, 3),
                self._pool_tensor(str(artifact["xclbin"]), str(artifact["insts"]), (math.prod(output_shape),), np.float32, 4),
            )
            self._maxpool_workspace_cache[workspace_key] = workspaces
        input_tensor, output_tensor = workspaces
        with input_tensor.overwrite() as host_input:
            np.copyto(host_input, padded.reshape(-1))
        self._profile["maxpool_pad_upload_ms"] = self._profile.get("maxpool_pad_upload_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        launch_start = time.perf_counter()
        self._pool_kernel(str(artifact["xclbin"]), str(artifact["insts"]))(input_tensor, output_tensor)
        self._profile["maxpool_kernel_ms"] = self._profile.get("maxpool_kernel_ms", 0.0) + (time.perf_counter() - launch_start) * 1000.0
        self._executed["xdna_maxpool"] += 1
        self._executed["device_resident_pool_outputs"] += 1
        return _DeviceValue(output_tensor, output_shape, 1.0, 0, producer=f"maxpool:{index}", layout="nchw")

    def _run_quantized_maxpool_kernel(self, index: int, raw: np.ndarray) -> _DeviceValue:
        """Pool the uint8 Relu-Q tensor directly and preserve its QDQ edge on-device."""
        if self.maxpool_runtime == "iron":
            import aie.iron as iron
            from aie.iron.device import from_name
            iron.set_current_device(from_name("npu2", n_cols=None))
        spec = self.operation_specs[index]
        params = spec["parameters"]
        artifact = spec["compiled_artifact"]
        raw_shape = tuple(int(value) for value in spec["input_shapes"][0])
        x = np.asarray(raw, dtype=np.uint8).reshape(raw_shape)
        pads = tuple(int(value) for value in params["pads"])
        pt, pl, pb, pr = pads
        transfer_width = (int(raw_shape[3]) + pl + pr + 3) & ~3
        transfer_right_pad = transfer_width - int(raw_shape[3]) - pl
        padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, transfer_right_pad)), constant_values=0)
        expected = (1, int(params["channels"]), int(params["input_height"]), transfer_width)
        if padded.shape != expected:
            raise ValueError(f"uint8 MaxPool node {index} expects padded input {expected}, got {padded.shape}")
        output_shape = (1, int(params["channels"]), int(params["output_height"]), int(params["output_width"]))
        workspace_key = (str(artifact["xclbin"]), str(artifact["insts"]), expected, output_shape, "u8")
        workspaces = self._maxpool_workspace_cache.get(workspace_key)
        if workspaces is None:
            workspaces = (
                self._pool_tensor(str(artifact["xclbin"]), str(artifact["insts"]), (math.prod(expected),), np.uint8, 3),
                self._pool_tensor(str(artifact["xclbin"]), str(artifact["insts"]), (math.prod(output_shape),), np.uint8, 4),
            )
            self._maxpool_workspace_cache[workspace_key] = workspaces
        input_tensor, output_tensor = workspaces
        started = time.perf_counter()
        with input_tensor.overwrite() as host_input:
            np.copyto(host_input, padded.reshape(-1))
        self._profile["maxpool_pad_upload_ms"] = self._profile.get("maxpool_pad_upload_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        launch_start = time.perf_counter()
        self._pool_kernel(str(artifact["xclbin"]), str(artifact["insts"]))(input_tensor, output_tensor)
        self._profile["maxpool_kernel_ms"] = self._profile.get("maxpool_kernel_ms", 0.0) + (time.perf_counter() - launch_start) * 1000.0
        self._executed["xdna_maxpool"] += 1
        self._executed["qdq_maxpool_fusions"] += 1
        self._executed["device_resident_pool_outputs"] += 1
        fusion = self._maxpool_qdq_fusions[index]
        return _DeviceValue(
            output_tensor, output_shape, float(fusion["scale"]), int(fusion["zero_point"]),
            as_real=True, producer=f"maxpool:{index}", layout="nhwc",
        )

    def _run_quantized_add_relu(self, index: int, values: dict[str, Any]) -> _DeviceValue:
        """Run the compiled residual Add+ReLU+Quantize kernel on XDNA."""
        spec = self.operation_specs[index]
        quant = spec["quantization"]
        artifact = spec["compiled_artifact"]
        shape = tuple(int(v) for v in spec["output_shapes"][0])
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"quantized Add node {index} requires batch-one NCHW tensors")
        elements = math.prod(shape)
        key = (str(artifact["xclbin"]), str(artifact["insts"]), elements)
        cached = self._qadd_workspace_cache.get(key)
        if cached is None:
            if self.runtime_backend == "xrt":
                kernel = self._artifact_kernel(str(artifact["xclbin"]), str(artifact["insts"]))
                cached = (
                    kernel.tensor((elements,), np.uint8, 3),
                    kernel.tensor((elements,), np.uint8, 4),
                    kernel.tensor((elements,), np.uint8, 5),
                )
            else:
                import aie.iron as iron
                from aie.iron.device import from_name
                iron.set_current_device(from_name("npu2", n_cols=None))
                cached = tuple(
                    iron.tensor(np.zeros(elements, dtype=np.uint8), dtype=np.uint8, device="npu")
                    for _ in range(3)
                )
            self._qadd_workspace_cache[key] = cached
        lhs_workspace, rhs_workspace, output_tensor = cached

        qadd_kernel = self._artifact_kernel(str(artifact["xclbin"]), str(artifact["insts"]))

        def input_tensor(name: str, workspace: Any, argument_index: int) -> Any:
            value = values.get(name)
            if (
                isinstance(value, _DeviceValue)
                and value.layout == "nhwc"
                and value.shape == shape
                and hasattr(value.tensor, "bo")
                and value.tensor._device is qadd_kernel.device
                and value.tensor.group == qadd_kernel.kernel.group_id(argument_index)
            ):
                # XRT buffers are byte-compatible; the kernel consumes uint8 bit patterns.
                return value.tensor
            raw = self._host_value(value) if isinstance(value, _DeviceValue) else np.asarray(value)
            nhwc = np.asarray(raw, dtype=np.uint8).reshape(shape).transpose(0, 2, 3, 1).copy()
            with workspace.overwrite() as host:
                np.copyto(host, nhwc.reshape(-1))
            return workspace

        lhs = input_tensor(str(quant["raw_inputs"][0]), lhs_workspace, 3)
        rhs = input_tensor(str(quant["raw_inputs"][1]), rhs_workspace, 4)
        qadd_kernel(lhs, rhs, output_tensor)
        self._executed["xdna_quantized_add_relu"] += 1
        return _DeviceValue(
            output_tensor, shape, float(quant["output_scale"]),
            int(quant["output_zero_point"]), producer=f"qadd:{index}", layout="nhwc",
        )

    def _run_conv(self, index: int, node: Any, values: dict[str, np.ndarray]) -> np.ndarray:
        total_start = time.perf_counter()
        plan = self.conv_plans[index]
        if index in self._forced_cpu_convs or (self.cpu_small_m and plan.gemm_shape[0] <= self.cpu_small_m):
            return self._run_small_conv_cpu(index, node, values, plan, total_start)
        stage_start = time.perf_counter()
        in_raw, in_scale, in_zero = self._quant_source(node.input[0], values)
        wt_raw, wt_scale, wt_zero = self._quant_source(node.input[1], values)
        x = _centered_int8(in_raw, in_zero, f"Conv {plan.node_name} activation")
        panels = im2col_nchw(x, plan)
        pack_ms = (time.perf_counter() - stage_start) * 1000.0
        output = np.empty(plan.output_shape, dtype=np.float32)
        spec = self._artifact(plan)
        cm, ck, cn = (int(value) for value in spec["compiled_shape"])
        n_out_group = plan.weight_shape[0] // plan.groups
        m_rows = plan.gemm_shape[0]
        khkwc = plan.gemm_shape[1]
        packed_key = (index, cm, ck, cn)
        packed_weights = self._packed_weight_cache.get(packed_key)
        if packed_weights is None:
            weight_start = time.perf_counter()
            weights = _centered_int8(wt_raw, wt_zero, f"Conv {plan.node_name} weights")
            packed_groups = []
            for group in range(plan.groups):
                w = weights[group * n_out_group : (group + 1) * n_out_group].reshape(n_out_group, khkwc).T
                b = np.zeros((ck, cn), dtype=np.int8)
                b[:khkwc, :n_out_group] = w
                packed_groups.append(b)
            packed_weights = tuple(packed_groups)
            self._packed_weight_cache[packed_key] = packed_weights
            self._profile["conv_static_weight_pack_ms"] = self._profile.get("conv_static_weight_pack_ms", 0.0) + (time.perf_counter() - weight_start) * 1000.0
        bias = values[node.input[2]].astype(np.float32).reshape(-1) if len(node.input) > 2 else None
        kernel_info = spec["compiled_artifact"]
        kernel = self._artifact_kernel(kernel_info["xclbin"], kernel_info["insts"])

        launch_start = time.perf_counter()
        for group in range(plan.groups):
            matrix_start = time.perf_counter()
            a = np.zeros((cm, ck), dtype=np.int8)
            a[:m_rows, :khkwc] = panels[group]
            b = packed_weights[group]
            matrix_ms = (time.perf_counter() - matrix_start) * 1000.0
            self._profile["conv_matrix_padding_ms"] = self._profile.get("conv_matrix_padding_ms", 0.0) + matrix_ms
            # Use the loaded kernel identity as well as artifact paths.  Some
            # compiler outputs reuse paths while reloading artifacts during a
            # long-lived RPC process; only BOs allocated from this exact XRT
            # kernel/context are safe to pass to it.
            workspace_key = (id(kernel), kernel_info["xclbin"], kernel_info["insts"], cm, ck, cn)
            workspaces = self._workspace_cache.get(workspace_key)
            if workspaces is None:
                alloc_start = time.perf_counter()
                if self.runtime_backend == "xrt":
                    workspaces = (
                        kernel.tensor((cm, ck), np.int8, 3),
                        kernel.tensor((ck, cn), np.int8, 4),
                        kernel.tensor((cm, cn), np.int32, 5),
                    )
                else:
                    import aie.iron as iron
                    workspaces = (
                        iron.tensor((cm, ck), dtype=np.int8, device="npu"),
                        iron.tensor((ck, cn), dtype=np.int8, device="npu"),
                        iron.tensor((cm, cn), dtype=np.int32, device="npu"),
                    )
                self._workspace_cache[workspace_key] = workspaces
                self._profile["conv_buffer_alloc_ms"] = self._profile.get("conv_buffer_alloc_ms", 0.0) + (time.perf_counter() - alloc_start) * 1000.0
            at, bt, ct = workspaces
            upload_start = time.perf_counter()
            with at.overwrite() as host_a:
                np.copyto(host_a, a)
            with bt.overwrite() as host_b:
                np.copyto(host_b, b)
            with ct.overwrite() as host_c:
                host_c.fill(0)
            self._profile["conv_host_upload_prep_ms"] = self._profile.get("conv_host_upload_prep_ms", 0.0) + (time.perf_counter() - upload_start) * 1000.0
            dispatch_start = time.perf_counter()
            kernel(at, bt, ct)
            self._profile["conv_kernel_call_ms"] = self._profile.get("conv_kernel_call_ms", 0.0) + (time.perf_counter() - dispatch_start) * 1000.0
            read_start = time.perf_counter()
            values_out = np.asarray(ct.numpy()[:m_rows, :n_out_group], dtype=np.float32).copy()
            self._profile["conv_readback_ms"] = self._profile.get("conv_readback_ms", 0.0) + (time.perf_counter() - read_start) * 1000.0
            values_out *= in_scale * wt_scale
            if bias is not None:
                values_out += bias[group * n_out_group : (group + 1) * n_out_group]
            if plan.fused_relu:
                values_out = np.maximum(values_out, 0)
            shaped = values_out.reshape(plan.output_shape[0], plan.output_shape[2], plan.output_shape[3], n_out_group)
            output[:, group * n_out_group : (group + 1) * n_out_group] = shaped.transpose(0, 3, 1, 2)
        launch_ms = (time.perf_counter() - launch_start) * 1000.0
        self._profile["conv_pack_and_prepare_ms"] = self._profile.get("conv_pack_and_prepare_ms", 0.0) + pack_ms
        self._profile["conv_dispatch_and_post_ms"] = self._profile.get("conv_dispatch_and_post_ms", 0.0) + launch_ms
        self._executed["xdna_conv"] += 1
        self._conv_times.append({
            "node_index": index,
            "node_name": plan.node_name,
            "input_shape": list(plan.input_shape),
            "output_shape": list(plan.output_shape),
            "gemm_shape": list(plan.gemm_shape),
            "tile": list(plan.tile),
            "artifact": spec["key"],
            "elapsed_ms": (time.perf_counter() - total_start) * 1000.0,
        })
        return output

    def _run_small_conv_cpu(
        self, index: int, node: Any, values: dict[str, np.ndarray], plan: Any, total_start: float
    ) -> np.ndarray:
        stage_start = time.perf_counter()
        in_raw, in_scale, in_zero = self._quant_source(node.input[0], values)
        wt_raw, wt_scale, wt_zero = self._quant_source(node.input[1], values)
        x = _centered_int8(in_raw, in_zero, f"Conv {plan.node_name} activation")
        weights = self._cpu_weight_cache.get(index)
        if weights is None:
            weights = _centered_int8(wt_raw, wt_zero, f"Conv {plan.node_name} weights")
            self._cpu_weight_cache[index] = weights
        if self.cpu_backend == "torch-int8" and plan.groups == 1:
            import torch
            if not hasattr(torch, "_int_mm"):
                raise RuntimeError("--cpu-backend torch-int8 requires PyTorch with torch._int_mm support")
            if not self._torch_initialized:
                torch.set_num_threads(self.cpu_threads)
                self._torch_initialized = True
            attrs = _attrs(node)
            pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
            tx = torch.from_numpy(np.ascontiguousarray(x))
            if any(pads):
                tx = torch.nn.functional.pad(tx, (pads[1], pads[3], pads[0], pads[2]))
            kh, kw = plan.weight_shape[2:]
            dh, dw = plan.dilation
            sh, sw = plan.stride
            effective_h = dh * (kh - 1) + 1
            effective_w = dw * (kw - 1) + 1
            windows = tx.unfold(2, effective_h, sh).unfold(3, effective_w, sw)
            windows = windows[..., ::dh, ::dw]
            panels_tensor = windows.permute(0, 2, 3, 1, 4, 5).reshape(
                tx.shape[0], -1, tx.shape[1] * kh * kw
            ).contiguous()
            panels = panels_tensor[0].numpy()[None, ...]
        elif self.cpu_backend in {"numpy", "torch-int8"}:
            panels = im2col_nchw(x, plan)
        else:
            panels = None
        bias = values[node.input[2]].astype(np.float32).reshape(-1) if len(node.input) > 2 else None
        self._profile["cpu_conv_prepare_ms"] = self._profile.get("cpu_conv_prepare_ms", 0.0) + (time.perf_counter() - stage_start) * 1000.0
        execute_start = time.perf_counter()
        # The configured threshold is intended for batch-1 tiny feature maps,
        # where NumPy's integer GEMM avoids a device launch per Conv.
        batch, out_channels, out_h, out_w = plan.output_shape
        out_per_group = out_channels // plan.groups
        if self.cpu_backend in {"torch", "torch-int8"}:
            # oneDNN's CPU convolution avoids materializing and multiplying a
            # large int32 im2col matrix. Each centered int8 value is exactly
            # representable in float32; keep this experimental backend
            # opt-in because long reductions can round the integer accumulator.
            import torch
            import torch.nn.functional as torch_f

            if not self._torch_initialized:
                torch.set_num_threads(self.cpu_threads)
                self._torch_initialized = True

            if self.cpu_backend == "torch-int8":
                # CPU int8 GEMM keeps the quantized Conv accumulator exact and
                # avoids float conversion and oneDNN Conv setup on tiny maps.
                tw_groups = self._torch_int8_weight_cache.get(index)
                if tw_groups is None:
                    tw_groups = tuple(
                        torch.from_numpy(np.ascontiguousarray(
                            weights[g * out_per_group : (g + 1) * out_per_group]
                            .reshape(out_per_group, -1).T
                        ))
                        for g in range(plan.groups)
                    )
                    self._torch_int8_weight_cache[index] = tw_groups
                raw = np.empty(plan.output_shape, dtype=np.float32)
                for group in range(plan.groups):
                    panel = torch.from_numpy(panels[group])
                    valid_rows = panel.shape[0]
                    if valid_rows % 16:
                        padded_panel = torch.zeros(
                            (math.ceil(valid_rows / 16) * 16, panel.shape[1]), dtype=torch.int8
                        )
                        padded_panel[:valid_rows] = panel
                        panel = padded_panel
                    acc = torch._int_mm(panel, tw_groups[group]).numpy()[:valid_rows]
                    raw[:, group * out_per_group : (group + 1) * out_per_group] = acc.reshape(
                        batch, out_h, out_w, out_per_group
                    ).transpose(0, 3, 1, 2)
            else:
                attrs = _attrs(node)
                pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
                strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
                dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
                tx = torch.from_numpy(np.array(x, dtype=np.float32, copy=True, order="C"))
                tw = self._torch_weight_cache.get(index)
                if tw is None:
                    tw = torch.from_numpy(np.array(weights, dtype=np.float32, copy=True, order="C"))
                    self._torch_weight_cache[index] = tw
                conv_padding = (0, 0)
                if pads[0] == pads[2] and pads[1] == pads[3]:
                    conv_padding = (pads[0], pads[1])
                elif any(pads):
                    tx = torch_f.pad(tx, (pads[1], pads[3], pads[0], pads[2]))
                raw = torch_f.conv2d(
                    tx,
                    tw,
                    bias=None,
                    stride=strides,
                    padding=conv_padding,
                    dilation=dilations,
                    groups=plan.groups,
                ).numpy()
        else:
            raw = np.empty(plan.output_shape, dtype=np.float32)
            for group in range(plan.groups):
                cache_key = (index, group)
                cached = self._cpu_numpy_matrix_cache.get(cache_key)
                if cached is None:
                    w = weights[group * out_per_group : (group + 1) * out_per_group]
                    reduction = int(np.prod(w.shape[1:]))
                    # Centered int8 x int8 partial sums are below 2**24 for short reductions,
                    # so a float32 BLAS GEMM is bit-exact and far faster than an int32 matmul.
                    exact_float = reduction * 128 * 127 < (1 << 24)
                    matrix = w.reshape(out_per_group, -1).T
                    cached = (np.ascontiguousarray(matrix.astype(np.float32 if exact_float else np.int32)), exact_float)
                    self._cpu_numpy_matrix_cache[cache_key] = cached
                matrix, exact_float = cached
                if exact_float:
                    acc = panels[group].astype(np.float32) @ matrix
                else:
                    acc = panels[group].astype(np.int32) @ matrix
                raw[:, group * out_per_group : (group + 1) * out_per_group] = acc.reshape(
                    batch, out_h, out_w, out_per_group
                ).transpose(0, 3, 1, 2)
        execute_ms = (time.perf_counter() - execute_start) * 1000.0
        raw = raw.astype(np.float32)
        raw *= in_scale * wt_scale
        if bias is not None:
            raw += bias.reshape(1, -1, 1, 1)
        output = np.maximum(raw, 0) if plan.fused_relu else raw
        elapsed = (time.perf_counter() - total_start) * 1000.0
        self._profile["cpu_conv_execute_ms"] = self._profile.get("cpu_conv_execute_ms", 0.0) + execute_ms
        self._executed["cpu_conv"] += 1
        self._conv_times.append({
            "node_index": index,
            "node_name": plan.node_name,
            "input_shape": list(plan.input_shape),
            "output_shape": list(plan.output_shape),
            "gemm_shape": list(plan.gemm_shape),
            "backend": "cpu",
            "elapsed_ms": elapsed,
        })
        return output

    def _run_fused_bottleneck(self, values: dict[str, np.ndarray], block: dict[str, Any]) -> None:
        binding = block["binding"]
        total_start = time.perf_counter()
        if block.get("direct_xrt"):
            activation = values[binding["input_raw_name"]]
            handoff_reason = "resident"
            source_group = None
            target_group = int(block["kernel"].kernel.group_id(3))
            if not isinstance(activation, _DeviceValue):
                handoff_reason = "host_value"
            elif activation.as_real:
                handoff_reason = "dequantized_value"
            elif activation.layout != "nhwc":
                handoff_reason = "layout_mismatch"
            elif tuple(activation.shape) != tuple(binding["input_shape"]):
                handoff_reason = "shape_mismatch"
            elif not hasattr(activation.tensor, "bo"):
                handoff_reason = "non_xrt_buffer"
            else:
                source_group = activation.tensor.group
                if activation.tensor._device is not block["kernel"].device:
                    handoff_reason = "device_mismatch"
                elif source_group != target_group:
                    handoff_reason = "group_mismatch"
            can_handoff = (
                isinstance(activation, _DeviceValue)
                and not activation.as_real
                and activation.layout == "nhwc"
                and tuple(activation.shape) == tuple(binding["input_shape"])
                and hasattr(activation.tensor, "bo")
                and activation.tensor._device is block["kernel"].device
                and activation.tensor.group == block["kernel"].kernel.group_id(3)
            )
            if can_handoff:
                block_input = activation.tensor
                self._executed["device_resident_handoffs"] += 1
            else:
                raw = self._host_value(activation).reshape(binding["input_shape"])
                channel_last = raw.transpose(0, 2, 3, 1).copy().view(np.int8).reshape(-1)
                with block["input"].overwrite() as host_input:
                    np.copyto(host_input, channel_last)
                block_input = block["input"]
            launch_start = time.perf_counter()
            if block["parallel_projection"]:
                block["kernel"](
                    block_input, block["main_parameters"], block["skip_parameters"],
                    block["output"], output_indices=(3,),
                )
            else:
                block["kernel"](
                    block_input, block["parameters"], block["output"], output_indices=(2,),
                )
            elapsed_ms = (time.perf_counter() - launch_start) * 1000.0
            output_shape = tuple(int(value) for value in binding["output_shape"])
            if self._capture_enabled:
                output_raw = block["output"].numpy().view(np.uint8).reshape(
                    output_shape[0], output_shape[2], output_shape[3], output_shape[1]
                ).transpose(0, 3, 1, 2).copy()
                self._capture_outputs[f"fused_block:{binding['block'].prefix}"] = output_raw
            output_value = _DeviceValue(
                block["output"], output_shape, float(binding["output_scale"]),
                int(binding["output_zero_point"]), producer=binding["block"].prefix,
                layout="nhwc",
            )
            values[binding["output_raw_name"]] = output_value
            values[binding["output_dequant_name"]] = _DeviceValue(
                block["output"], output_shape, output_value.scale, output_value.zero_point,
                as_real=True, producer=output_value.producer, layout="nhwc",
            )
            self._fused_times.append({
                "prefix": binding["block"].prefix,
                "input_shape": list(binding["input_shape"]),
                "output_shape": list(output_shape),
                "device_resident_input": bool(can_handoff),
                "runtime": "xrt",
                "input_handoff_reason": handoff_reason,
                "input_bo_group": source_group,
                "input_kernel_group": target_group,
                "elapsed_ms": elapsed_ms,
            })
            self._profile["fused_bottleneck_kernel_call_ms"] = self._profile.get(
                "fused_bottleneck_kernel_call_ms", 0.0
            ) + elapsed_ms
            self._profile["fused_bottleneck_total_ms"] = self._profile.get(
                "fused_bottleneck_total_ms", 0.0
            ) + (time.perf_counter() - total_start) * 1000.0
            self._executed["fused_bottleneck"] += 1
            return
        activation = values[binding["input_raw_name"]]
        if isinstance(activation, _DeviceValue):
            if tuple(activation.shape) != tuple(binding["input_shape"]):
                raise ValueError(f"device handoff shape mismatch for {binding['block'].prefix}")
            block_input = activation.tensor
            self._executed["device_resident_handoffs"] += 1
        else:
            start = time.perf_counter()
            raw = np.asarray(activation).reshape(binding["input_shape"])
            channel_last = raw.transpose(0, 2, 3, 1).copy().view(np.int8).reshape(-1)
            with block["input"].overwrite() as host_input:
                np.copyto(host_input, channel_last)
            block_input = block["input"]
            self._profile["fused_bottleneck_input_prep_ms"] = (
                self._profile.get("fused_bottleneck_input_prep_ms", 0.0)
                + (time.perf_counter() - start) * 1000.0
            )

        launch_start = time.perf_counter()
        if block["parallel_projection"]:
            block["kernel"](
                block_input, block["main_parameters"], block["skip_parameters"], block["output"]
            )
        else:
            block["kernel"](block_input, block["parameters"], block["output"])
        if self._capture_enabled:
            self._capture_outputs[f"fused_block:{binding['block'].prefix}"] = (
                block["output"].numpy().view(np.uint8).reshape(
                    binding["output_shape"][0], binding["output_shape"][2],
                    binding["output_shape"][3], binding["output_shape"][1],
                ).transpose(0, 3, 1, 2).copy()
            )
        kernel_ms = (time.perf_counter() - launch_start) * 1000.0
        self._profile["fused_bottleneck_kernel_call_ms"] = (
            self._profile.get("fused_bottleneck_kernel_call_ms", 0.0)
            + kernel_ms
        )
        self._fused_times.append({
            "prefix": binding["block"].prefix,
            "input_shape": list(binding["input_shape"]),
            "output_shape": list(binding["output_shape"]),
            "device_resident_input": isinstance(activation, _DeviceValue),
            "elapsed_ms": kernel_ms,
        })

        output_shape = tuple(int(value) for value in binding["output_shape"])
        device_output = _DeviceValue(
            block["output"],
            output_shape,
            float(binding["output_scale"]),
            int(binding["output_zero_point"]),
            producer=binding["block"].prefix,
        )
        values[binding["output_raw_name"]] = device_output
        values[binding["output_dequant_name"]] = _DeviceValue(
            block["output"], output_shape, device_output.scale, device_output.zero_point,
            as_real=True, producer=device_output.producer,
        )
        self._profile["fused_bottleneck_total_ms"] = (
            self._profile.get("fused_bottleneck_total_ms", 0.0)
            + (time.perf_counter() - total_start) * 1000.0
        )
        self._executed["fused_bottleneck"] += 1

    def _prepare_device_network(self, model: Any, first_prefix: str, covered: set[int]) -> dict[str, Any]:
        """Tensors + packing for the stem+pool+stage-column artifact (resnet_stage_design.py --stem)."""
        import aie.iron as iron

        try:
            import stem_pool
            from blocked_stage import pack_rt_params
            from resnet_stage_design import stage_specs, stem_spec
        except ImportError:
            from . import stem_pool
            from .blocked_stage import pack_rt_params
            from .resnet_stage_design import stage_specs, stem_spec
        cols, binds = stage_specs(model, self.device_network_stages)
        stem = stem_pool.extract_stem(model)
        spec = stem_spec(model, len(cols))
        params = np.concatenate(
            [stem_pool.pack_stem_params(stem)]
            + [pack_rt_params(b, slot_bytes=c["slot"]) for c, bs in zip(cols, binds) for b in bs]
        )
        scratch = sum((c["repeat"] + 1) * c["out_obj"] for c in cols) - cols[-1]["out_obj"]
        scratch += max(c["act_obj"] for c in cols) + cols[0]["act_obj"]
        pre = stem_pool.stem_nodes(model, binds[0][0]["input_raw_name"])
        return {
            "covered": set(covered) | pre, "stem": stem, "cols": cols, "binds": binds,
            "input": iron.tensor(np.zeros(spec["chunks"] * spec["chunk_in"], dtype=np.int8), dtype=np.int8, device="npu"),
            "output": iron.zeros(cols[-1]["out_obj"], dtype=np.int8, device="npu"),
            "parameters": iron.tensor(params, dtype=np.uint8, device="npu"),
            "scratch": iron.zeros(scratch, dtype=np.int8, device="npu"),
        }

    def _prepare_layer_engine(self, model: Any, first_prefix: str, covered: set[int]) -> dict[str, Any]:
        """Tensors + packing for the layer-sequential engine artifact (layer_engine_design.py --net bodyr --looped)."""
        import aie.iron as iron

        try:
            import layer_engine as le
            from layer_engine_net import jobs_from_bindings
            from resnet_stage_design import stage_specs
        except ImportError:
            from . import layer_engine as le
            from .layer_engine_net import jobs_from_bindings
            from .resnet_stage_design import stage_specs
        _cols, binds = stage_specs(model, self.device_network_stages)
        jobs, _outs = jobs_from_bindings(binds, reuse_slots=True)
        packs = [le.pack_job(job, le.ENGINE_SLOT_BYTES) for job in jobs]
        params = np.concatenate([np.concatenate([pack[col].reshape(-1) for pack in packs]) for col in range(le.COLS)])
        slots = 1 + max(job.out_slot for job in jobs)
        arena = iron.zeros(slots * le.SLOT_BYTES, dtype=np.int8, device="npu")
        return {
            "covered": set(covered), "engine": {"jobs": jobs, "le": le}, "binds": binds,
            "input": arena, "output": arena, "parameters": iron.tensor(params, dtype=np.uint8, device="npu"),
            "scratch": None,
        }

    def _run_layer_engine(self, values: dict[str, Any], stage: dict[str, Any]) -> None:
        """Pooled map in -> every bottleneck block through the layer-sequential engine -> last block output."""
        engine = stage["network"]["engine"]
        le, jobs = engine["le"], engine["jobs"]
        first, last = stage["bindings"][0], stage["bindings"][-1]
        started = time.perf_counter()
        _, cin, height, width = first["input_shape"]
        raw = np.asarray(values[first["input_raw_name"]]).reshape(cin, height * width)
        arena = stage["input"]
        with arena.overwrite() as host:
            host.view(np.uint8)[: le.SLOT_BYTES] = le.to_arena(raw.T.copy(), jobs[0].in_layout)
        self._profile["device_network_prepare_ms"] = self._profile.get("device_network_prepare_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        launch = time.perf_counter()
        stage["kernel"](arena, stage["parameters"], arena)
        elapsed_ms = (time.perf_counter() - launch) * 1000.0
        self._profile["fused_stage_kernel_call_ms"] = self._profile.get("fused_stage_kernel_call_ms", 0.0) + elapsed_ms
        final = jobs[-1]
        slot = arena.numpy().view(np.uint8)[final.out_slot * le.SLOT_BYTES : (final.out_slot + 1) * le.SLOT_BYTES]
        dense = le.from_arena(slot, final.out_layout)
        _, channels, out_h, out_w = last["output_shape"]
        out = dense.reshape(out_h, out_w, channels).transpose(2, 0, 1)[None].copy()
        values[last["output_raw_name"]] = out
        values[last["output_dequant_name"]] = (
            (out.astype(np.float32) - float(last["output_zero_point"])) * np.float32(last["output_scale"])
        )
        self._fused_times.append({
            "prefix": "layer_engine", "input_shape": list(first["input_shape"]), "output_shape": list(last["output_shape"]),
            "device_resident_input": False, "linked_blocks": len(stage["bindings"]), "elapsed_ms": elapsed_ms,
        })
        self._executed["fused_stage"] = self._executed.get("fused_stage", 0) + 1
        self._executed["fused_bottleneck"] += len(stage["bindings"])
        self._executed["layer_engine"] = self._executed.get("layer_engine", 0) + 1

    def _run_device_network(self, values: dict[str, Any], stage: dict[str, Any]) -> None:
        """Image in -> (stem Conv, MaxPool, all bottleneck stages) on the NPU -> last block output."""
        network = stage["network"]
        try:
            import stem_pool
        except ImportError:
            from . import stem_pool
        started = time.perf_counter()
        image = np.asarray(values[network["stem"]["input_name"]], dtype=np.float32)
        data = stem_pool.im2col_chunks(image, network["stem"])
        with stage["input"].overwrite() as host_input:
            np.copyto(host_input, data.view(np.int8))
        self._profile["device_network_prepare_ms"] = self._profile.get("device_network_prepare_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        launch = time.perf_counter()
        stage["kernel"](stage["input"], stage["parameters"], stage["output"], stage["extra"])
        elapsed_ms = (time.perf_counter() - launch) * 1000.0
        self._profile["fused_stage_kernel_call_ms"] = self._profile.get("fused_stage_kernel_call_ms", 0.0) + elapsed_ms
        last = stage["bindings"][-1]
        _, channels, height, width = last["output_shape"]
        raw = (
            stage["output"].numpy().view(np.uint8)[: channels * height * width]
            .reshape(channels // 8, height * width, 8).transpose(0, 2, 1).reshape(1, channels, height, width).copy()
        )
        values[last["output_raw_name"]] = raw
        values[last["output_dequant_name"]] = (
            (raw.astype(np.float32) - float(last["output_zero_point"])) * np.float32(last["output_scale"])
        )
        self._fused_times.append({
            "prefix": "device_network", "input_shape": list(image.shape), "output_shape": list(last["output_shape"]),
            "device_resident_input": False, "linked_blocks": len(stage["bindings"]), "elapsed_ms": elapsed_ms,
        })
        self._executed["fused_stage"] = self._executed.get("fused_stage", 0) + 1
        self._executed["fused_bottleneck"] += len(stage["bindings"])
        self._executed["device_network"] = self._executed.get("device_network", 0) + 1

    def _run_fused_stage(self, values: dict[str, np.ndarray], stage: dict[str, Any]) -> None:
        if stage.get("network") is not None:
            if stage["network"].get("engine") is not None:
                self._run_layer_engine(values, stage)
            else:
                self._run_device_network(values, stage)
            return
        """Launch a linked multi-block stage once and expose only its final edge."""
        bindings = stage["bindings"]
        first, last = bindings[0], bindings[-1]
        activation = values[first["input_raw_name"]]
        if isinstance(activation, _DeviceValue):
            if tuple(activation.shape) != tuple(first["input_shape"]):
                raise ValueError("linked stage input shape does not match its first block")
            stage_input = activation.tensor
            resident_input = True
            self._executed["device_resident_handoffs"] += 1
        else:
            raw = np.asarray(activation).reshape(first["input_shape"])
            channel_last = raw.transpose(0, 2, 3, 1).copy().view(np.int8).reshape(-1)
            with stage["input"].overwrite() as host_input:
                np.copyto(host_input, channel_last)
            stage_input = stage["input"]
            resident_input = False

        started = time.perf_counter()
        if stage.get("tap") is not None:
            stage["kernel"](stage_input, stage["parameters"], stage["output"], stage["tap"])
            if self._capture_enabled:
                shape = bindings[0]["output_shape"]
                self._capture_outputs["stage_tap:block0"] = stage["tap"].numpy().view(np.uint8).reshape(
                    shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).copy()
        elif stage.get("extra") is not None:
            stage["kernel"](stage_input, stage["parameters"], stage["output"], stage["extra"])
        else:
            stage["kernel"](stage_input, stage["parameters"], stage["output"])
        producer = "+".join(binding["block"].prefix for binding in bindings)
        if self._capture_enabled:
            self._capture_outputs[f"fused_stage:{producer}"] = (
                stage["output"].numpy().view(np.uint8).reshape(
                    last["output_shape"][0], last["output_shape"][2],
                    last["output_shape"][3], last["output_shape"][1],
                ).transpose(0, 3, 1, 2).copy()
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._profile["fused_stage_kernel_call_ms"] = (
            self._profile.get("fused_stage_kernel_call_ms", 0.0) + elapsed_ms
        )
        output_shape = tuple(last["output_shape"])
        device_output = _DeviceValue(
            stage["output"], output_shape, float(last["output_scale"]),
            int(last["output_zero_point"]), producer=producer,
        )
        values[last["output_raw_name"]] = device_output
        values[last["output_dequant_name"]] = _DeviceValue(
            stage["output"], output_shape, device_output.scale, device_output.zero_point,
            as_real=True, producer=producer,
        )
        self._fused_times.append({
            "prefix": producer, "input_shape": list(first["input_shape"]),
            "output_shape": list(output_shape), "device_resident_input": resident_input,
            "linked_blocks": len(bindings), "elapsed_ms": elapsed_ms,
        })
        self._executed["fused_stage"] = self._executed.get("fused_stage", 0) + 1
        self._executed["fused_bottleneck"] += len(bindings)

    def _run_host_qadd_fusion(self, add_index: int, values: dict[str, Any]) -> None:
        """Execute DQ+Add+Relu+Q+DQ as one host-side graph step."""
        spec = self._host_qadd_fusions[add_index]
        started = time.perf_counter()
        inputs = []
        for dq_index, (scale, zero) in zip(spec["input_dq_indices"], spec["input_qparams"]):
            raw_name = self.nodes[dq_index].input[0]
            raw = values[raw_name]
            if isinstance(raw, _DeviceValue):
                raw = self._host_value(raw)
            real = np.asarray(raw).astype(np.float32)
            np.subtract(real, np.float32(zero), out=real)
            np.multiply(real, np.float32(scale), out=real)
            inputs.append(real)
        summed = inputs[0]
        np.add(summed, inputs[1], out=summed)
        np.maximum(summed, np.float32(0), out=summed)
        q_scale, q_zero = spec["quantize_params"]
        np.divide(summed, np.float32(q_scale), out=summed)
        np.rint(summed, out=summed)
        np.add(summed, np.float32(q_zero), out=summed)
        quantized = np.clip(summed, 0, 255).astype(np.uint8)
        dq_scale, dq_zero = spec["dequantize_params"]
        dequantized = _dequantize(
            quantized,
            np.asarray([dq_scale], dtype=np.float32),
            np.asarray([dq_zero], dtype=np.uint8),
            1,
        )
        values[spec["quantize_output"]] = quantized
        values[spec["dequantize_output"]] = dequantized
        self._precomputed_host_nodes.update((
            spec["relu_index"], spec["quantize_index"], spec["dequantize_index"],
        ))
        self._profile["host_fused_qadd_ms"] = (
            self._profile.get("host_fused_qadd_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        )
        self._executed["host_fused_qadd"] += 1

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._executed = {
            "xdna_conv": 0,
            "xdna_maxpool": 0,
            "qdq_maxpool_fusions": 0,
            "xdna_quantized_add_relu": 0,
            "device_resident_pool_outputs": 0,
            "device_view_ops": 0,
            "cpu_conv": 0,
            "host_ops": 0,
            "skipped_conv_dq": 0,
            "fused_relu": 0,
            "fused_bottleneck": 0,
            "device_resident_handoffs": 0,
            "device_edge_readbacks": 0,
            "host_fused_qadd": 0,
        }
        self._profile = {}
        self._device_readback_cache = {}
        self._conv_times = []
        self._fused_times = []
        self._precomputed_relu_nodes: set[int] = set()
        self._precomputed_native_nodes: set[int] = set()
        self._precomputed_host_nodes: set[int] = set()
        values = dict(self.arrays)
        values.update(inputs)
        for index, node in enumerate(self.nodes):
            if index in self._static_nodes:
                continue
            if index in self._fused_stage_nodes:
                stage_prefix, is_start = self._fused_stage_nodes[index]
                if is_start:
                    self._run_fused_stage(values, self._fused_stages[stage_prefix])
                continue
            if index in self._fused_nodes:
                prefix, is_start = self._fused_nodes[index]
                if is_start:
                    self._run_fused_bottleneck(values, self._fused_blocks[prefix])
                continue
            if index in self._precomputed_relu_nodes:
                continue
            if index in self._precomputed_native_nodes:
                continue
            if index in self._precomputed_host_nodes or index in self._host_qadd_input_dqs:
                continue
            if index in self._maxpool_input_dqs:
                continue
            if index in self._fused_input_handoffs:
                raw = values.get(str(node.input[0]))
                if (not isinstance(raw, _DeviceValue)
                        or raw.producer != self._fused_input_handoffs[index]):
                    raise RuntimeError("planned device handoff input is not resident on device")
                values[str(node.output[0])] = raw
                continue
            if index in self._dq_only_used_by_conv:
                self._executed["skipped_conv_dq"] += 1
                continue
            node_start = time.perf_counter()
            op = node.op_type
            if op == "Add" and index in self._host_qadd_fusions:
                self._run_host_qadd_fusion(index, values)
                continue
            offloaded = False
            attrs = _attrs(node)
            result = self._device_view(index, node, values)
            native_qadd = (
                op == "Add"
                and index in self.operation_specs
                and self.operation_specs[index].get("compiled_artifact")
                and self.operation_specs[index].get("quantization")
            )
            if result is not None:
                offloaded = True
                self._executed["device_view_ops"] += 1
                args = []
            elif op == "MaxPool" and index in self._maxpool_qdq_fusions:
                args = []
            else:
                args = [] if op == "Conv" or native_qadd else [
                    self._host_value(values[name]) for name in node.input if name
                ]
            if result is not None:
                pass
            elif op == "Constant":
                result = numpy_helper.to_array(attrs["value"])
            elif native_qadd:
                result = self._run_quantized_add_relu(index, values)
                quant = self.operation_specs[index]["quantization"]
                values[str(quant["raw_output"])] = result
                for fused_index in quant.get("fused_node_indices", ())[1:]:
                    self._precomputed_native_nodes.add(int(fused_index))
                offloaded = True
            elif op == "QuantizeLinear":
                result = _quantize(args[0], args[1], args[2], int(attrs.get("axis", 1)))
            elif op == "DequantizeLinear":
                if all(name in self.arrays for name in node.input if name):
                    # Every input is a constant (weights/bias): dequantize once, not per inference.
                    result = self._constant_dequant.get(index)
                    if result is None:
                        result = _dequantize(args[0], args[1], args[2], int(attrs.get("axis", 1)))
                        self._constant_dequant[index] = result
                else:
                    result = _dequantize(args[0], args[1], args[2], int(attrs.get("axis", 1)))
            elif op == "Conv":
                result = self._run_conv(index, node, values)
            elif op == "Relu":
                result = np.maximum(args[0], 0)
            elif op == "Add":
                result = args[0] + args[1]
            elif op == "Mul":
                result = args[0] * args[1]
            elif op == "MaxPool":
                if index in self.operation_specs:
                    if index in self._maxpool_qdq_fusions:
                        raw = values[self._maxpool_qdq_fusions[index]["input_raw"]]
                        result = self._run_quantized_maxpool_kernel(index, raw)
                    else:
                        result = self._run_maxpool_kernel(index, args[0])
                    offloaded = True
                else:
                    result = _max_pool(args[0], attrs)
            elif op == "GlobalAveragePool":
                result = np.mean(args[0], axis=tuple(range(2, args[0].ndim)), keepdims=True)
            elif op == "Flatten":
                axis = int(attrs.get("axis", 1))
                result = args[0].reshape(math.prod(args[0].shape[:axis]), math.prod(args[0].shape[axis:]))
            elif op == "Gemm":
                a = args[0].T if int(attrs.get("transA", 0)) else args[0]
                b = args[1].T if int(attrs.get("transB", 0)) else args[1]
                result = float(attrs.get("alpha", 1.0)) * (a @ b)
                if len(args) > 2:
                    result = result + float(attrs.get("beta", 1.0)) * args[2]
            else:
                raise NotImplementedError(f"node {index} {op} has no XDNA graph-runner implementation")
            for output_name in node.output:
                values[output_name] = result if isinstance(result, _DeviceValue) else np.asarray(result)
            if op == "Conv":
                tail_start = time.perf_counter()
                self._fuse_conv_tail(index, result, values)
                tail_ms = (time.perf_counter() - tail_start) * 1000.0
                self._profile["conv_fused_relu_ms"] = self._profile.get("conv_fused_relu_ms", 0.0) + tail_ms
            if op not in {"Conv", "Constant"} and not offloaded:
                self._executed["host_ops"] += 1
            if op != "Conv" and not offloaded:
                key = f"host_{op}_ms"
                self._profile[key] = self._profile.get(key, 0.0) + (time.perf_counter() - node_start) * 1000.0
        return {value.name: self._host_value(values[value.name]) for value in self.model.graph.output}

    def _fuse_conv_tail(self, conv_index: int, output: np.ndarray, values: dict[str, np.ndarray]) -> None:
        """Materialize a fused Conv+Relu output once instead of repeating Relu."""
        relu = self._fused_relu_for_conv.get(conv_index)
        if relu is not None:
            relu_index, relu_output = relu
            values[relu_output] = np.asarray(output)
            self._precomputed_relu_nodes.add(relu_index)
            self._executed["fused_relu"] += 1


def main(argv: list[str] | None = None, emit_json: bool = True) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-output", type=Path, help="save the first output tensor (.npy) for an external comparison")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--cpu-small-m", type=int, default=0,
        help="run batch-1 Conv layers with at most this many output pixels on CPU",
    )
    parser.add_argument(
        "--cpu-backend", choices=("numpy", "torch", "torch-int8"), default="numpy",
        help="CPU implementation for --cpu-small-m (torch-int8 uses exact integer GEMM)",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=2,
        help="PyTorch intra-op CPU threads for --cpu-backend torch",
    )
    parser.add_argument("--fused-block-prefix", help="fuse one supported identity bottleneck, e.g. /layer1/layer1.1")
    parser.add_argument("--fused-block-xclbin", type=Path)
    parser.add_argument("--fused-block-insts", type=Path)
    parser.add_argument(
        "--fused-block", nargs=3, action="append", metavar=("PREFIX", "XCLBIN", "INSTS"),
        help="add a fused bottleneck specialization; repeat to fuse multiple blocks",
    )
    parser.add_argument(
        "--fused-stage", nargs="+", action="append",
        metavar="BLOCK... XCLBIN INSTS",
        help="run one to three adjacent bottlenecks as one device-linked IRON stage",
    )
    parser.add_argument(
        "--fused-body", nargs=3, metavar=("XCLBIN", "INSTS", "GROUPS_JSON"),
        help="run every bottleneck as ONE resnet_body_design.py artifact; GROUPS_JSON lists the "
             "block-prefix groups in execution order, e.g. '[[\"/layer1/layer1.0\"],[...]]'",
    )
    parser.add_argument(
        "--device-network", nargs=3, metavar=("XCLBIN", "INSTS", "STAGES_JSON"),
        help="run stem Conv + MaxPool + every bottleneck stage as ONE resnet_stage_design.py --stem artifact; STAGES_JSON lists the block prefixes of each stage",
    )
    parser.add_argument(
        "--layer-engine", nargs=3, metavar=("XCLBIN", "INSTS", "STAGES_JSON"),
        help="run every bottleneck block through ONE layer-sequential engine artifact (layer_engine_design.py --net bodyr --looped); stem/MaxPool stay on the host",
    )
    parser.add_argument("--fused-body-rt", action="store_true", help="the --fused-body artifact was compiled with resnet_body_design.py --rt (runtime-shaped kernels)")
    parser.add_argument("--host-maxpool", action="store_true", help="run MaxPool on the host instead of an XDNA artifact")
    parser.add_argument("--fused-stage-blocked", action="store_true", help="fused stages were compiled with --blocked (vectorized layout)")
    parser.add_argument(
        "--parallel-projection-block", nargs=3, action="append", metavar=("PREFIX", "XCLBIN", "INSTS"),
        help="run a projection bottleneck with main and skip branches on separate NPU columns",
    )
    parser.add_argument("--maxpool-uint8-xclbin", type=Path, help="use a uint8 MaxPool artifact to fuse matching QDQ edges")
    parser.add_argument("--maxpool-uint8-insts", type=Path)
    parser.add_argument("--maxpool-runtime", choices=("iron", "xrt"), default="iron", help="runtime used for MaxPool artifacts")
    parser.add_argument("--runtime-backend", choices=("iron", "xrt"), help="runtime for fused bottleneck and MaxPool artifacts")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--capture-npz", type=Path, help="save fused-block/stage device outputs for debugging")
    args = parser.parse_args(argv)
    model = onnx.load(args.model)
    manifest = json.loads(args.manifest.read_text())
    if bool(args.maxpool_uint8_xclbin) != bool(args.maxpool_uint8_insts):
        raise ValueError("--maxpool-uint8-xclbin and --maxpool-uint8-insts must be supplied together")
    body_groups = None
    body_caps: dict[str, int | None] = {}
    body_stages = []
    network_stages = None
    if args.fused_body:
        import json as _json
        raw_groups = _json.loads(args.fused_body[2])
        # Entries are prefix lists or {"blocks": [...], "chunk_cap": N, "depth": D}.
        body_groups = [g["blocks"] if isinstance(g, dict) else g for g in raw_groups]
        body_caps = {
            prefix: g.get("chunk_cap")
            for g in raw_groups if isinstance(g, dict) for prefix in g["blocks"]
        }
        body_stages = [(tuple(prefix for group in body_groups for prefix in group), args.fused_body[0], args.fused_body[1])]

    if args.device_network:
        import json as _json
        network_stages = _json.loads(args.device_network[2])
        body_stages = [(tuple(p for stage in network_stages for p in stage), args.device_network[0], args.device_network[1])]

    if args.layer_engine:
        import json as _json
        network_stages = _json.loads(args.layer_engine[2])
        body_stages = [(tuple(p for stage in network_stages for p in stage), args.layer_engine[0], args.layer_engine[1])]

    runner = XDNAResNetRunner(
        model,
        manifest,
        cpu_small_m=args.cpu_small_m,
        cpu_backend=args.cpu_backend,
        cpu_threads=args.cpu_threads,
        fused_block_prefix=args.fused_block_prefix,
        fused_block_xclbin=str(args.fused_block_xclbin) if args.fused_block_xclbin else None,
        fused_block_insts=str(args.fused_block_insts) if args.fused_block_insts else None,
        fused_blocks=[(prefix, xclbin, insts) for prefix, xclbin, insts in (args.fused_block or [])],
        fused_stages=[
            (tuple(items[:-2]), items[-2], items[-1])
            for items in (args.fused_stage or [])
        ] + body_stages,
        fused_stage_blocked=args.fused_stage_blocked,
        fused_body_groups=body_groups,
        fused_body_chunk_caps=body_caps,
        host_maxpool=args.host_maxpool,
        fused_body_rt=args.fused_body_rt,
        device_network_stages=network_stages,
        layer_engine=bool(args.layer_engine),
        parallel_projection_blocks=[
            (prefix, xclbin, insts) for prefix, xclbin, insts in (args.parallel_projection_block or [])
        ],
        maxpool_uint8_artifact=(str(args.maxpool_uint8_xclbin), str(args.maxpool_uint8_insts))
        if args.maxpool_uint8_xclbin else None,
        maxpool_runtime=args.maxpool_runtime,
        runtime_backend=args.runtime_backend,
        capture_outputs=bool(args.capture_npz),
    )
    input_info = model.graph.input[0]
    shape = [int(dim.dim_value) or 1 for dim in input_info.type.tensor_type.shape.dim]
    sample = np.random.default_rng(args.seed).random(shape, dtype=np.float32)
    feed = {input_info.name: sample}
    start = time.perf_counter()
    runner.run(feed)
    cold_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(args.warmup):
        runner.run(feed)
    # Runner-local timers reset per graph execution; aggregate the timed runs
    # here so JSON profiles report steady-state per-inference means.
    profile_totals: dict[str, float] = {}
    conv_elapsed_totals: dict[int, float] = {}
    conv_records: dict[int, dict[str, Any]] = {}
    fused_elapsed_totals: dict[str, float] = {}
    fused_records: dict[str, dict[str, Any]] = {}
    start = time.perf_counter()
    for _ in range(args.iters):
        outputs = runner.run(feed)
        for name, elapsed in runner._profile.items():
            profile_totals[name] = profile_totals.get(name, 0.0) + elapsed
        for item in runner._conv_times:
            index = int(item["node_index"])
            conv_records[index] = {
                key: value for key, value in item.items() if key != "elapsed_ms"
            }
            conv_elapsed_totals[index] = conv_elapsed_totals.get(index, 0.0) + float(
                item["elapsed_ms"]
            )
        for item in runner._fused_times:
            key = str(item.get("prefix", item.get("node_name", len(fused_records))))
            fused_records[key] = {
                name: value for name, value in item.items() if name != "elapsed_ms"
            }
            fused_elapsed_totals[key] = fused_elapsed_totals.get(key, 0.0) + float(
                item.get("elapsed_ms", 0.0)
            )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = elapsed_ms / args.iters
    if args.capture_npz:
        captured = dict(runner._capture_outputs)
        captured.update({f"graph_output:{name}": value for name, value in outputs.items()})
        args.capture_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.capture_npz, **captured)
    conv_timings = [
        {**conv_records[index], "elapsed_ms": elapsed / args.iters}
        for index, elapsed in conv_elapsed_totals.items()
    ]
    fused_timings = [
        {**fused_records[key], "elapsed_ms": elapsed / args.iters}
        for key, elapsed in fused_elapsed_totals.items()
    ]
    runtime_subgraphs = runner.runtime_subgraph_report()
    runtime_subgraph_summary: dict[str, int] = {"total": len(runtime_subgraphs)}
    for item in runtime_subgraphs:
        status = str(item["status"])
        runtime_subgraph_summary[status] = runtime_subgraph_summary.get(status, 0) + 1
    runtime_subgraph_summary["native_dispatches"] = sum(
        int(item["native_dispatch_count"]) for item in runtime_subgraphs
    )
    result = {
        "backend": "amd_xdna_iron_xrt_resnet_graph",
        "runtime_backend": args.runtime_backend or args.maxpool_runtime,
        "execution": "full_graph_with_fused_bottleneck" if (args.fused_block_prefix or args.fused_block or args.parallel_projection_block or args.fused_stage) else (
            "full_graph_xdna_conv_host_ops" if not args.cpu_small_m else "full_graph_hybrid_conv_host_ops"
        ),
        "fused_block_prefix": args.fused_block_prefix,
        "fused_blocks": list(runner._fused_blocks),
        "fused_stages": [
            {"blocks": [block.prefix for block, _binding in stage["blocks"]], "xclbin": stage["xclbin"]}
            for stage in runner._fused_stages.values()
        ],
        "context_cache_limit": runner.context_cache_limit,
        "context_budget_fallback_blocks": runner.context_budget_fallback_blocks,
        "cpu_small_m_threshold": args.cpu_small_m,
        "cpu_backend": args.cpu_backend,
        "cpu_threads": args.cpu_threads if args.cpu_backend.startswith("torch") else None,
        "model": str(args.model),
        "input_seed": args.seed,
        "input_shape": list(shape),
        "graph_dispatches": runner.codegen.estimated_dispatches,
        "runtime_subgraph_summary": runtime_subgraph_summary,
        "runtime_subgraphs": runtime_subgraphs,
        "execution_counts": runner._executed,
        "unique_fused_xclbins_used": len({item["xclbin"] for item in runner._fused_blocks.values()}),
        "fused_block_timings": fused_timings,
        "profile_samples": args.iters,
        "profile_ms": {key: value / args.iters for key, value in profile_totals.items()},
        "conv_timings": conv_timings,
        "slowest_conv_nodes": sorted(
            conv_timings, key=lambda item: item["elapsed_ms"], reverse=True
        )[:8],
        "unique_xclbins_used": len({item["artifact"] for item in conv_timings if "artifact" in item}),
        "warmup": args.warmup,
        "iters": args.iters,
        "cold_ms": cold_ms,
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms,
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
    }
    if args.dump_output:
        np.save(args.dump_output, np.asarray(next(iter(outputs.values()))))
    if args.capture_npz:
        result["capture_npz"] = str(args.capture_npz)
    try:
        import onnxruntime as ort

        reference = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
        expected = reference.run(None, {input_info.name: sample})
        actual = next(iter(outputs.values()))
        delta = np.abs(actual.astype(np.float64) - expected[0].astype(np.float64))
        result["cpu_reference"] = {
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
            "argmax_match": int(np.argmax(actual)) == int(np.argmax(expected[0])),
        }
    except Exception as exc:
        result["cpu_reference"] = {"available": False, "reason": str(exc)}
    encoded = json.dumps(result, indent=2)
    if emit_json:
        print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
