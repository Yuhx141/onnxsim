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

## Dense throughput stress

`build_rv1106_peak.py` generates dense 3x3 INT8 Conv chains to reduce launch
and small-tensor overhead. These runs used direct RKNN calibration, zero-copy
INT8/NHWC I/O, and fixed-size inputs:

| Workload | Operations | Mean ms | Effective INT8 throughput |
| --- | ---: | ---: | ---: |
| 224x224, 64 channels, 8 Conv layers | 26.07 GOP | 109.80 | 237.4 GOPS |
| 224x224, 128 channels, 4 Conv layers | 44.74 GOP | 190.27 | 235.1 GOPS |

Widening the model did not increase throughput, so this is the current
measured ceiling for the flashed image/toolchain configuration, not a claim of
the silicon's theoretical peak. The board reports RV1106G3 compatibility and
the public product specification advertises up to 1 TOPS INT8 for G3; reaching
that figure will require confirming NPU clock/voltage state and the exact
Rockchip benchmark methodology.
## ImageNet reference models

ResNet-18 and MobileNetV2 were downloaded from the ONNX Model Zoo, simplified,
INT8-calibrated with RKNN-Toolkit2 2.3.2, and executed with zero-copy INT8/NHWC
input. Output memory used the model's native layout because the ResNet output
is packed as NC1HWC2 by the RV1106 runtime.

| Model | Approx. operations | Mean ms | Effective throughput | RKNN size |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18, 224x224 | 3.63 GOP | 18.73 | 193.8 GOPS | 11.3 MiB |
| MobileNetV2, 224x224 | 0.86 GOP | 12.94 | 66.3 GOPS | 3.75 MiB |

The dense stress graphs reach about 235 GOPS, so these realistic models leave
substantial utilization on the table. MobileNetV2 is especially limited by its
depthwise and pointwise workload mix plus small feature maps, rather than by
the peak dense-convolution rate.

## INT4 and clock-control checks

RKNN-Toolkit2 2.3.2 rejects `quantized_dtype="w4a16"` during
`rknn.config(target_platform="rv1106")`: `w4a16` is not supported for RV1106.
Therefore no INT4 RKNN model can be generated through this target/toolchain.
The board image also exposes only `/dev/rknpu` (driver v0.9.2); no NPU
devfreq, clock, or voltage control nodes were available through sysfs. The
235 GOPS stress result is consequently the ceiling of the current image and
clock configuration, not a frequency-tuned silicon peak.

The alternative `onnxsim.quantize_weight_only_int4` route was probed with a
64x64 MatMul. It generated a standard block-wise INT4 QDQ graph and ONNX
Runtime executed it, but RKNN-Toolkit2 2.3.2 treated it as QAT: loading
succeeded, `do_quantization=False` was rejected for RV1106, and calibrated
builds are ordinary INT8 models. Manually changing the opset to 19 only
caused the expected checker warning for the opset-21 `block_size` attribute;
it did not provide an INT4 kernel. `com.microsoft::MatMulNBits` is not a
portable RV1106 fallback either.

The public host SDK is already 2.3.2, the latest published version. Its
W4A16 support is target-specific (documented for RK3576), so an SDK update
cannot enable W4A16 on this RV1106 image without a vendor compiler/runtime
release.
