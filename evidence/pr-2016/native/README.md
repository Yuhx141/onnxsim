# Custom C++ IR round-trip

`roundtrip.cpp` uses onnxsim's pinned ONNX fork directly. It imports a `ModelProto` into the custom C++ IR, prints public graph outputs and use counts, exports the graph, and runs the native basic and full checkers before and after the round-trip. It does not implement or simulate any optimizer pass.

The recorded driver was linked against the exact-base build and custom ONNX revision `281e5e6accfd6113fd57d819ce47f32a937bb106`. Its SHA-256 was `cb3b89762abf6625b7d5f263100123b4c11c099f2d829b5203c9cee9f2eafae9`; the binary is not committed.

## Recorded outcomes

| Models | Result |
|---|---|
| 11 sources | 11 import/export and full-checker passes |
| 9 serializable pre-fix targets | 8 passes; #1996 imports/exports but the full checker rejects it |
| 11 pass-disabled targets | 11 passes |
| 11 PR-head targets | 11 passes |

There is no pre-fix target for #1998 or #1999: the actual simplification path fails before serialization. For #1995, the pre-fix target passes this native round-trip and full checking, while ONNX Runtime rejects its Conv bias. These boundaries are visible in `recorded/` and in the runtime results one directory above.

## Build

After building onnxsim, set `REPO` to that source tree. Normal FetchContent builds place the dependency sources under `$BUILD/_deps`; when an existing source cache is used, point `PROTOBUF_SOURCE` and `ABSL_SOURCE` to those source directories.

```bash
REPO=/path/to/onnxsim
BUILD="$REPO/.setuptools-cmake-build"
PROTOBUF_SOURCE="${PROTOBUF_SOURCE:-$BUILD/_deps/protobuf-src}"
ABSL_SOURCE="${ABSL_SOURCE:-$BUILD/_deps/absl-src}"

c++ -std=c++20 -O2 \
  -DONNX_ML=1 -DONNX_NAMESPACE=onnx -DONNX_USE_LITE_PROTO=1 \
  -I"$REPO/third_party/onnx" \
  -I"$BUILD/third_party/onnx" \
  -I"$PROTOBUF_SOURCE/src" \
  -I"$PROTOBUF_SOURCE/third_party/utf8_range" \
  -I"$ABSL_SOURCE" \
  roundtrip.cpp -o roundtrip \
  -Wl,--start-group \
  "$BUILD/third_party/onnx/libonnx.a" \
  "$BUILD/_deps/protobuf-build/libprotobuf-lite.a" \
  $(find "$BUILD/_deps/absl-build" -name '*.a' -print) \
  "$BUILD/_deps/protobuf-build/third_party/utf8_range/libutf8_validity.a" \
  -Wl,--end-group -lrt -pthread

./roundtrip ../cases/1995/case-01-source.onnx /tmp/source-roundtrip.onnx
```
