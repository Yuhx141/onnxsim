# Luckfox RV1106 NPU benchmark

Date: 2026-09-27

Board: Luckfox `rv1106g3` / `rv1103g-38x38-ipc-v10`, Buildroot Linux
5.10.160, using `/oem/usr/lib/librknnmrt.so`. Models were generated as static
NCHW ONNX graphs, simplified with onnxsim, compiled with RKNN-Toolkit2 2.3.2
for `target_platform="rv1106"`, and calibrated to INT8 with eight synthetic
images. The board runtime uses zero-copy INT8/NHWC input and output memory.

Each result is 20 warmups followed by 100 timed `rknn_run` calls. Timing is
on-device runtime time and excludes SSH/model upload.

| model | ONNX nodes | simplified nodes | RKNN size | input | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| conv_bn_relu_224 | 5 | 4 | 29 KiB | 224x224x3 | 2.381 ms | 2.316 ms | 2.525 ms | 4.397 ms |
| depthwise_pointwise_112 | 5 | 5 | 24 KiB | 112x112x3 | 0.485 ms | 0.410 ms | 1.100 ms | 2.141 ms |

For a float ONNX model, the RV1106 compiler rejects
`do_quantization=False`, so the direct path requires INT8 calibration. A
pre-quantized QDQ model is handled separately as a QAT model below.

## onnxsim pre-quantization

The same models were quantized with `onnxsim.quantize_static(...,
full_graph=True, method="minmax", op_types_to_exclude=["Relu",
"GlobalAveragePool"])`, then loaded by RKNN as pre-quantized QDQ/QAT models
and built with `do_quantization=False` and optimization level 3:

| model | RKNN size | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv_bn_relu_224 | 30 KiB | 2.355 ms | 2.304 ms | 2.535 ms | 3.756 ms |
| depthwise_pointwise_112 | 25 KiB | 0.500 ms | 0.414 ms | 1.161 ms | 2.705 ms |

Compared with the earlier unoptimized QDQ path, this reduces the 224x224
model from 2.721 ms to 2.355 ms and the depthwise model from 0.544 ms to
0.500 ms. The 224x224 result is now marginally faster than the direct RKNN
calibration result; the depthwise result remains about 3% slower.
