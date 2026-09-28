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
    output = np.empty((n, c, out_h, out_w), dtype=x.dtype)
    for oh in range(out_h):
        for ow in range(out_w):
            window = padded[
                :, :,
                oh * strides[0] : oh * strides[0] + dilations[0] * (kernel[0] - 1) + 1 : dilations[0],
                ow * strides[1] : ow * strides[1] + dilations[1] * (kernel[1] - 1) + 1 : dilations[1],
            ]
            output[:, :, oh, ow] = np.max(window, axis=(2, 3))
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
    ):
        self.model = model
        self.nodes = list(model.graph.node)
        self.arrays = _values(model)
        self.optimize_small_m = bool(manifest.get("optimize_small_m", False))
        self.cpu_small_m = max(0, int(cpu_small_m))
        self.cpu_backend = cpu_backend
        self.cpu_threads = max(1, int(cpu_threads))
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
        self._executed = {"xdna_conv": 0, "xdna_maxpool": 0, "xdna_quantized_add_relu": 0, "device_resident_pool_outputs": 0, "device_view_ops": 0, "cpu_conv": 0, "host_ops": 0, "skipped_conv_dq": 0, "fused_relu": 0, "fused_bottleneck": 0, "device_resident_handoffs": 0, "device_edge_readbacks": 0}
        self._profile: dict[str, float] = {}
        self._device_readback_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._conv_times: list[dict[str, Any]] = []
        self._fused_times: list[dict[str, Any]] = []
        self._workspace_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._maxpool_workspace_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._qadd_workspace_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        # ONNX weights are constants. Keep their padded GEMM layout so steady
        # state inference only packs the changing activation matrix.
        self._packed_weight_cache: dict[tuple[Any, ...], tuple[np.ndarray, ...]] = {}
        self._cpu_weight_cache: dict[int, np.ndarray] = {}
        self._torch_weight_cache: dict[int, Any] = {}
        fused_specs = list(fused_blocks or ())
        if any((fused_block_prefix, fused_block_xclbin, fused_block_insts)):
            if not all((fused_block_prefix, fused_block_xclbin, fused_block_insts)):
                raise ValueError("fused block requires its node prefix, xclbin, and instruction stream")
            fused_specs.append((fused_block_prefix, fused_block_xclbin, fused_block_insts))
        bottleneck_plans = {block.prefix: block for block in plan_bottleneck_blocks(model)}
        prepared_blocks: dict[str, tuple[Any, dict[str, Any], set[int], str, str]] = {}
        for prefix, xclbin, insts in fused_specs:
            if prefix in prepared_blocks:
                raise ValueError(f"fused block {prefix!r} was specified more than once")
            block = bottleneck_plans.get(prefix)
            if block is None:
                raise ValueError(f"no bottleneck block found for prefix {prefix!r}")
            if not Path(xclbin).is_file() or not Path(insts).is_file():
                raise ValueError("fused block xclbin and instruction stream must exist")
            binding = bind_fused_bottleneck(model, block)
            prepared_blocks[prefix] = (block, binding, set(binding["covered_nodes"]), str(xclbin), str(insts))

        # Strix Halo supports 16 simultaneous hardware contexts. Reserve those
        # slots across pooling, fused blocks, and any remaining XDNA Conv shapes.
        self.context_cache_limit = min(16, max(1, int(os.environ.get("XRT_CONTEXT_CACHE_SIZE", "16"))))
        self.context_budget_fallback_blocks: list[str] = []
        self._forced_cpu_convs: set[int] = set()
        active_prefixes = set(prepared_blocks)
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
            } | set(remaining_conv_contexts) | remaining_qadd_contexts
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
                raise RuntimeError("compiled operator kernels exceed the XRT context limit")
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
        self._fused_nodes: dict[int, tuple[str, bool]] = {}
        self._fused_input_handoffs: dict[int, str] = {}
        if fused_specs:
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
            block_input = iron.tensor(
                np.zeros(input_count, dtype=np.int8), dtype=np.int8, device="npu"
            )
            block_parameters = iron.tensor(
                binding["params"], dtype=np.uint8, device="npu"
            )
            block_output = iron.zeros(output_count, dtype=np.int8, device="npu")
            self._fused_blocks[prefix] = {
                "binding": binding,
                "start": start_index,
                "input": block_input,
                "parameters": block_parameters,
                "output": block_output,
                "kernel": self._kernel(str(xclbin), str(insts)),
                "xclbin": str(xclbin),
            }
            for index in covered:
                self._fused_nodes[index] = (prefix, index == start_index)
        self._plan_fused_input_handoffs()

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

    def _run_maxpool_kernel(self, index: int, x: np.ndarray) -> _DeviceValue:
        """Upload padded NCHW input and retain the pooling result in XRT memory."""
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
                iron.tensor((math.prod(expected),), dtype=np.float32, device="npu"),
                iron.tensor(np.zeros(math.prod(output_shape), dtype=np.float32), dtype=np.float32, device="npu"),
            )
            self._maxpool_workspace_cache[workspace_key] = workspaces
        input_tensor, output_tensor = workspaces
        with input_tensor.overwrite() as host_input:
            np.copyto(host_input, padded.reshape(-1))
        self._profile["maxpool_pad_upload_ms"] = self._profile.get("maxpool_pad_upload_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        launch_start = time.perf_counter()
        self._kernel(str(artifact["xclbin"]), str(artifact["insts"]))(input_tensor, output_tensor)
        self._profile["maxpool_kernel_ms"] = self._profile.get("maxpool_kernel_ms", 0.0) + (time.perf_counter() - launch_start) * 1000.0
        self._executed["xdna_maxpool"] += 1
        self._executed["device_resident_pool_outputs"] += 1
        return _DeviceValue(output_tensor, output_shape, 1.0, 0, producer=f"maxpool:{index}", layout="nchw")

    def _run_quantized_add_relu(self, index: int, values: dict[str, Any]) -> _DeviceValue:
        """Run the compiled residual Add+ReLU+Quantize kernel on XDNA."""
        import aie.iron as iron
        from aie.iron.device import from_name

        iron.set_current_device(from_name("npu2", n_cols=None))
        spec = self.operation_specs[index]
        quant = spec["quantization"]
        artifact = spec["compiled_artifact"]
        shape = tuple(int(v) for v in spec["output_shapes"][0])
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"quantized Add node {index} requires batch-one NCHW tensors")
        elements = math.prod(shape)
        key = (str(artifact["xclbin"]), elements)
        cached = self._qadd_workspace_cache.get(key)
        if cached is None:
            cached = tuple(
                iron.tensor(np.zeros(elements, dtype=np.uint8), dtype=np.uint8, device="npu")
                for _ in range(3)
            )
            self._qadd_workspace_cache[key] = cached
        lhs_workspace, rhs_workspace, output_tensor = cached

        def input_tensor(name: str, workspace: Any) -> Any:
            value = values.get(name)
            if isinstance(value, _DeviceValue) and value.layout == "nhwc" and value.shape == shape:
                # XRT buffers are byte-compatible; the kernel consumes uint8 bit patterns.
                return value.tensor
            raw = self._host_value(value) if isinstance(value, _DeviceValue) else np.asarray(value)
            nhwc = np.asarray(raw, dtype=np.uint8).reshape(shape).transpose(0, 2, 3, 1).copy()
            with workspace.overwrite() as host:
                np.copyto(host, nhwc.reshape(-1))
            return workspace

        lhs = input_tensor(str(quant["raw_inputs"][0]), lhs_workspace)
        rhs = input_tensor(str(quant["raw_inputs"][1]), rhs_workspace)
        self._kernel(str(artifact["xclbin"]), str(artifact["insts"]))(lhs, rhs, output_tensor)
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
        kernel = self._kernel(kernel_info["xclbin"], kernel_info["insts"])
        import aie.iron as iron

        launch_start = time.perf_counter()
        for group in range(plan.groups):
            matrix_start = time.perf_counter()
            a = np.zeros((cm, ck), dtype=np.int8)
            a[:m_rows, :khkwc] = panels[group]
            b = packed_weights[group]
            matrix_ms = (time.perf_counter() - matrix_start) * 1000.0
            self._profile["conv_matrix_padding_ms"] = self._profile.get("conv_matrix_padding_ms", 0.0) + matrix_ms
            workspace_key = (kernel_info["xclbin"], cm, ck, cn)
            workspaces = self._workspace_cache.get(workspace_key)
            if workspaces is None:
                alloc_start = time.perf_counter()
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
        panels = im2col_nchw(x, plan) if self.cpu_backend == "numpy" else None
        bias = values[node.input[2]].astype(np.float32).reshape(-1) if len(node.input) > 2 else None
        self._profile["cpu_conv_prepare_ms"] = self._profile.get("cpu_conv_prepare_ms", 0.0) + (time.perf_counter() - stage_start) * 1000.0
        # The configured threshold is intended for batch-1 tiny feature maps,
        # where NumPy's integer GEMM avoids a device launch per Conv.
        batch, out_channels, out_h, out_w = plan.output_shape
        out_per_group = out_channels // plan.groups
        if self.cpu_backend == "torch":
            # oneDNN's CPU convolution avoids materializing and multiplying a
            # large int32 im2col matrix. Each centered int8 value is exactly
            # representable in float32; keep this experimental backend
            # opt-in because long reductions can round the integer accumulator.
            import torch
            import torch.nn.functional as torch_f

            if not self._torch_initialized:
                torch.set_num_threads(self.cpu_threads)
                self._torch_initialized = True

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
                w = weights[group * out_per_group : (group + 1) * out_per_group]
                matrix = w.reshape(out_per_group, -1).T.astype(np.int32)
                acc = panels[group].astype(np.int32) @ matrix
                raw[:, group * out_per_group : (group + 1) * out_per_group] = acc.reshape(
                    batch, out_h, out_w, out_per_group
                ).transpose(0, 3, 1, 2)
        raw = raw.astype(np.float32)
        raw *= in_scale * wt_scale
        if bias is not None:
            raw += bias.reshape(1, -1, 1, 1)
        output = np.maximum(raw, 0) if plan.fused_relu else raw
        elapsed = (time.perf_counter() - total_start) * 1000.0
        self._profile["cpu_conv_execute_ms"] = self._profile.get("cpu_conv_execute_ms", 0.0) + elapsed
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
        block["kernel"](block_input, block["parameters"], block["output"])
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

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._executed = {
            "xdna_conv": 0,
            "xdna_maxpool": 0,
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
        }
        self._profile = {}
        self._device_readback_cache = {}
        self._conv_times = []
        self._fused_times = []
        self._precomputed_relu_nodes: set[int] = set()
        self._precomputed_native_nodes: set[int] = set()
        values = dict(self.arrays)
        values.update(inputs)
        for index, node in enumerate(self.nodes):
            if index in self._fused_nodes:
                prefix, is_start = self._fused_nodes[index]
                if is_start:
                    self._run_fused_bottleneck(values, self._fused_blocks[prefix])
                continue
            if index in self._precomputed_relu_nodes:
                continue
            if index in self._precomputed_native_nodes:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--cpu-small-m", type=int, default=0,
        help="run batch-1 Conv layers with at most this many output pixels on CPU",
    )
    parser.add_argument(
        "--cpu-backend", choices=("numpy", "torch"), default="numpy",
        help="CPU implementation for --cpu-small-m (torch uses optimized float32 Conv2d)",
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
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.model)
    manifest = json.loads(args.manifest.read_text())
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
    start = time.perf_counter()
    for _ in range(args.iters):
        outputs = runner.run(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_ms = elapsed_ms / args.iters
    result = {
        "backend": "amd_xdna_iron_xrt_resnet_graph",
        "execution": "full_graph_with_fused_bottleneck" if (args.fused_block_prefix or args.fused_block) else (
            "full_graph_xdna_conv_host_ops" if not args.cpu_small_m else "full_graph_hybrid_conv_host_ops"
        ),
        "fused_block_prefix": args.fused_block_prefix,
        "fused_blocks": list(runner._fused_blocks),
        "context_cache_limit": runner.context_cache_limit,
        "context_budget_fallback_blocks": runner.context_budget_fallback_blocks,
        "cpu_small_m_threshold": args.cpu_small_m,
        "cpu_backend": args.cpu_backend,
        "cpu_threads": args.cpu_threads if args.cpu_backend == "torch" else None,
        "model": str(args.model),
        "graph_dispatches": runner.codegen.estimated_dispatches,
        "execution_counts": runner._executed,
        "unique_fused_xclbins_used": len({item["xclbin"] for item in runner._fused_blocks.values()}),
        "fused_block_timings": runner._fused_times,
        "profile_ms": runner._profile,
        "slowest_conv_nodes": sorted(
            runner._conv_times, key=lambda item: item["elapsed_ms"], reverse=True
        )[:8],
        "unique_xclbins_used": len({item["artifact"] for item in runner._conv_times if "artifact" in item}),
        "warmup": args.warmup,
        "iters": args.iters,
        "cold_ms": cold_ms,
        "avg_ms": avg_ms,
        "fps": 1000.0 / avg_ms,
        "output_shapes": {name: list(value.shape) for name, value in outputs.items()},
    }
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
    print(encoded)
    if args.json:
        args.json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
