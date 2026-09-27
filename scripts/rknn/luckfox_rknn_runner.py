#!/usr/bin/env python3
"""Minimal RKNN Runtime benchmark runner for a Luckfox/RV1106 board.

This file is intentionally dependency-free and is copied to the target.  The
Luckfox Buildroot image ships ``librknnmrt.so`` but normally does not ship the
Python RKNN Lite package, so ctypes is used for the small stable runtime API.

Example on the board::

    python3 luckfox_rknn_runner.py model.rknn --input-shape 1,3,224,224

The model must already have been compiled on the host with
``rknn.config(target_platform="rv1106")`` and ``rknn.export_rknn()``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time


RKNN_QUERY_IN_OUT_NUM = 0
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_TENSOR_NC1HWC2 = 3


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("_padding", ctypes.c_uint8 * 3),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through_attr", ctypes.c_uint8),
        ("_padding2", ctypes.c_uint8 * 3),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnTensorMem(ctypes.Structure):
    # ARMv7 EABI aligns uint64_t to 4 bytes; ctypes otherwise uses the host
    # ABI's 8-byte alignment and shifts every field after phys_addr.
    _pack_ = 4
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("_padding", ctypes.c_uint8 * 2),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


def _fail(api: str, ret: int) -> None:
    if ret:
        raise RuntimeError(f"{api} failed with {ret}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="compiled .rknn model")
    ap.add_argument("--input-shape", default="1,3,224,224")
    ap.add_argument("--input-type", choices=("int8", "float32"), default="int8")
    ap.add_argument("--input-format", choices=("nchw", "nhwc"), default="nhwc")
    ap.add_argument("--output-format", choices=("native", "nchw", "nhwc"), default="nhwc")
    ap.add_argument("--inputs", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--library", default="/oem/usr/lib/librknnmrt.so")
    args = ap.parse_args()
    shape = tuple(int(x) for x in args.input_shape.split(","))
    if not shape or any(x <= 0 for x in shape):
        ap.error("--input-shape must be comma-separated positive dimensions")
    if args.inputs != 1:
        ap.error("the first runner supports one input; pass --inputs 1")

    with open(args.model, "rb") as f:
        model = f.read()
    # Keep the backing buffers alive for the entire runtime lifetime.
    model_buf = ctypes.create_string_buffer(model)
    input_type = RKNN_TENSOR_INT8 if args.input_type == "int8" else RKNN_TENSOR_FLOAT32
    input_size = (1 if input_type == RKNN_TENSOR_INT8 else 4) * _product(shape)
    input_buf = ctypes.create_string_buffer(input_size)

    lib = ctypes.CDLL(args.library)
    lib.rknn_init.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_init.restype = ctypes.c_int
    lib.rknn_query.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                               ctypes.c_uint32]
    lib.rknn_query.restype = ctypes.c_int
    lib.rknn_create_mem.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.rknn_create_mem.restype = ctypes.POINTER(RknnTensorMem)
    lib.rknn_set_io_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(RknnTensorMem),
                                    ctypes.POINTER(RknnTensorAttr)]
    lib.rknn_set_io_mem.restype = ctypes.c_int
    lib.rknn_destroy_mem.argtypes = [ctypes.c_uint32, ctypes.POINTER(RknnTensorMem)]
    lib.rknn_destroy_mem.restype = ctypes.c_int
    lib.rknn_run.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    lib.rknn_run.restype = ctypes.c_int
    lib.rknn_destroy.argtypes = [ctypes.c_uint32]
    lib.rknn_destroy.restype = ctypes.c_int

    ctx = ctypes.c_uint32(0)
    _fail("rknn_init", lib.rknn_init(ctypes.byref(ctx), model_buf, len(model), 0, None))
    try:
        io_num = RknnInputOutputNum()
        _fail("rknn_query", lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM,
                                             ctypes.byref(io_num), ctypes.sizeof(io_num)))
        if io_num.n_input != 1:
            raise RuntimeError(f"expected one model input, found {io_num.n_input}")
        input_attr = RknnTensorAttr(index=0)
        _fail("rknn_query(input_attr)", lib.rknn_query(
            ctx, 1, ctypes.byref(input_attr), ctypes.sizeof(input_attr)))
        input_format = RKNN_TENSOR_NHWC if args.input_format == "nhwc" else RKNN_TENSOR_NCHW
        input_mem_size = input_attr.size_with_stride or input_attr.size
        input_mem = lib.rknn_create_mem(ctx, input_mem_size)
        if not input_mem:
            raise RuntimeError("rknn_create_mem returned null for input")
        ctypes.memset(input_mem.contents.virt_addr, 0, input_mem_size)
        input_attr.type = input_type
        input_attr.fmt = input_format
        _fail("rknn_set_io_mem(input)", lib.rknn_set_io_mem(
            ctx, input_mem, ctypes.byref(input_attr)))
        output_attr = RknnTensorAttr(index=0)
        _fail("rknn_query(output_attr)", lib.rknn_query(
            ctx, 2, ctypes.byref(output_attr), ctypes.sizeof(output_attr)))
        output_mem = lib.rknn_create_mem(ctx, output_attr.size_with_stride or output_attr.size)
        if not output_mem:
            raise RuntimeError("rknn_create_mem returned null for output")
        output_attr.type = RKNN_TENSOR_INT8
        if args.output_format == "native":
            pass
        else:
            output_attr.fmt = RKNN_TENSOR_NHWC if args.output_format == "nhwc" else RKNN_TENSOR_NCHW
        _fail("rknn_set_io_mem(output)", lib.rknn_set_io_mem(
            ctx, output_mem, ctypes.byref(output_attr)))
        for _ in range(args.warmup):
            _fail("rknn_run", lib.rknn_run(ctx, None))
        samples = []
        for _ in range(args.iterations):
            t0 = time.perf_counter_ns()
            _fail("rknn_run", lib.rknn_run(ctx, None))
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        print(json.dumps({
            "model": os.path.basename(args.model),
            "soc": "rv1106",
            "iterations": len(samples),
            "latency_ms": {
                "min": min(samples),
                "mean": statistics.mean(samples),
                "p50": statistics.median(samples),
                "p95": _percentile(samples, 0.95),
                "max": max(samples),
            },
        }, sort_keys=True))
    finally:
        if "input_mem" in locals():
            lib.rknn_destroy_mem(ctx, input_mem)
        lib.rknn_destroy(ctx)
    return 0


def _product(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]


if __name__ == "__main__":
    raise SystemExit(main())
