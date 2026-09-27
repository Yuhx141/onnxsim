#!/usr/bin/env python3
"""Execute a static QDQ ResNet graph with Conv GEMMs on XDNA.

Convolutions run through compiled IRON/XRT artifacts. The remaining ONNX ops
run in NumPy on the host, so this is full graph execution with XDNA Conv
offload, not an all-operators-on-NPU claim.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from conv_lowering import plan_all_convs
from conv_reference import im2col_nchw
from resnet_codegen import build_codegen_plan


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
    ):
        self.model = model
        self.nodes = list(model.graph.node)
        self.arrays = _values(model)
        self.optimize_small_m = bool(manifest.get("optimize_small_m", False))
        self.cpu_small_m = max(0, int(cpu_small_m))
        self.cpu_backend = cpu_backend
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
        self._executed = {"xdna_conv": 0, "cpu_conv": 0, "host_ops": 0, "skipped_conv_dq": 0, "fused_relu": 0}
        self._profile: dict[str, float] = {}
        self._conv_times: list[dict[str, Any]] = []
        self._workspace_cache: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        # ONNX weights are constants. Keep their padded GEMM layout so steady
        # state inference only packs the changing activation matrix.
        self._packed_weight_cache: dict[tuple[Any, ...], tuple[np.ndarray, ...]] = {}
        self._cpu_weight_cache: dict[int, np.ndarray] = {}

    def _quant_source(self, value_name: str, values: dict[str, np.ndarray]) -> tuple[np.ndarray, float, int]:
        dq = self.nodes_by_output.get(value_name)
        if dq is None or dq.op_type != "DequantizeLinear":
            raise ValueError(f"Conv tensor {value_name!r} is not produced by DequantizeLinear")
        raw_name, scale_name, zero_name = dq.input[:3]
        raw = values[raw_name]
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

    def _run_conv(self, index: int, node: Any, values: dict[str, np.ndarray]) -> np.ndarray:
        total_start = time.perf_counter()
        plan = self.conv_plans[index]
        if self.cpu_small_m and plan.gemm_shape[0] <= self.cpu_small_m:
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

            if torch.get_num_threads() > 1:
                # Small batch-1 feature maps lose more to thread-pool
                # coordination than they gain from CPU parallelism.
                torch.set_num_threads(1)

            attrs = _attrs(node)
            pads = tuple(int(v) for v in attrs.get("pads", (0, 0, 0, 0)))
            strides = tuple(int(v) for v in attrs.get("strides", (1, 1)))
            dilations = tuple(int(v) for v in attrs.get("dilations", (1, 1)))
            tx = torch.from_numpy(np.array(x, copy=True, order="C")).to(torch.float32)
            tw = torch.from_numpy(np.array(weights, copy=True, order="C")).to(torch.float32)
            if any(pads):
                tx = torch_f.pad(tx, (pads[1], pads[3], pads[0], pads[2]))
            raw = torch_f.conv2d(
                tx,
                tw,
                bias=None,
                stride=strides,
                padding=0,
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

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._executed = {"xdna_conv": 0, "cpu_conv": 0, "host_ops": 0, "skipped_conv_dq": 0, "fused_relu": 0}
        self._profile = {}
        self._conv_times = []
        self._precomputed_relu_nodes: set[int] = set()
        values = dict(self.arrays)
        values.update(inputs)
        for index, node in enumerate(self.nodes):
            if index in self._precomputed_relu_nodes:
                continue
            if index in self._dq_only_used_by_conv:
                self._executed["skipped_conv_dq"] += 1
                continue
            node_start = time.perf_counter()
            op = node.op_type
            attrs = _attrs(node)
            args = [] if op == "Conv" else [values[name] for name in node.input if name]
            if op == "Constant":
                result = numpy_helper.to_array(attrs["value"])
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
                values[output_name] = np.asarray(result)
            if op == "Conv":
                tail_start = time.perf_counter()
                self._fuse_conv_tail(index, result, values)
                tail_ms = (time.perf_counter() - tail_start) * 1000.0
                self._profile["conv_fused_relu_ms"] = self._profile.get("conv_fused_relu_ms", 0.0) + tail_ms
            if op not in {"Conv", "Constant"}:
                self._executed["host_ops"] += 1
            if op != "Conv":
                key = f"host_{op}_ms"
                self._profile[key] = self._profile.get(key, 0.0) + (time.perf_counter() - node_start) * 1000.0
        return {value.name: values[value.name] for value in self.model.graph.output}

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
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.model)
    manifest = json.loads(args.manifest.read_text())
    runner = XDNAResNetRunner(
        model,
        manifest,
        cpu_small_m=args.cpu_small_m,
        cpu_backend=args.cpu_backend,
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
        "execution": "full_graph_xdna_conv_host_ops" if not args.cpu_small_m else "full_graph_hybrid_conv_host_ops",
        "cpu_small_m_threshold": args.cpu_small_m,
        "cpu_backend": args.cpu_backend,
        "model": str(args.model),
        "graph_dispatches": runner.codegen.estimated_dispatches,
        "execution_counts": runner._executed,
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
