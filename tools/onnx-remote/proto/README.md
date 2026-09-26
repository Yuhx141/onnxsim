# Optional gRPC schema

`onnxsim_remote.proto` is an optional host-side API for the remote executor.
It is not needed to build or run the dependency-free C++ transport, and it is
not intended to be linked into constrained workers by default.

The schema deliberately uses a small raw tensor message instead of importing
ONNX `TensorProto` or the full KServe protocol. This keeps generated clients
usable for compile hosts, ROS2/DORA gateways, and simple test runners while
preserving the v5 transport's tensor ABI:

* `dtype` is the numeric ONNX `TensorProto.DataType` value.
* `shape` is a positive row-major shape.
* `raw_data` is little-endian flattened tensor storage.
* profile timestamps are worker-relative microseconds.

The intended deployment is:

```text
KServe/Triton client (optional)
          |
   gRPC gateway / host
          |
 OnnxSimExecutor RPC
          |
 native v5 binary transport
          |
 constrained runner
```

`Compile` and `LoadArtifact` are separate from `Execute`, so compilation and
execution hosts can be different. The `parameters` maps are for small options
such as shape/profile settings; model and artifact bytes must use their
dedicated fields.

Generation is intentionally left to the consumer's toolchain because gRPC
runtime availability varies by host. For example:

```sh
protoc -I tools/onnx-remote/proto \
  --cpp_out=/tmp/onnxsim-grpc \
  tools/onnx-remote/proto/onnxsim_remote.proto
```

The standalone CMake project can generate a reusable static binding target when
protobuf and gRPC development packages are installed:

```sh
cmake -S tools/onnx-remote -B build/onnx-remote-grpc \
  -DONNXSIM_REMOTE_GRPC=ON
cmake --build build/onnx-remote-grpc --target onnx_remote_grpc
```

This option is deliberately off by default and is intended for a host-side
gateway, not an embedded runner.

The generated build also provides `onnx-remote-grpc-gateway`. It accepts the
same native worker endpoint used by the C++ client:

```sh
build/onnx-remote-grpc/onnx-remote-grpc-gateway \
  --listen 0.0.0.0:50051 --worker-host 127.0.0.1 --worker-port 39501
```

The initial gateway uses insecure gRPC credentials for trusted local networks.
Put it behind a TLS/authenticated reverse proxy or add gRPC credentials before
exposing it beyond a trusted ROS2/DORA domain.

An actual gRPC service should translate `Execute` to
`onnx_remote::Request`/`Response` and retain the native transport as the
canonical embedded path. A KServe V2 adapter can map `ModelInfer` to
`Execute`; KServe's model metadata and readiness methods remain gateway
features rather than requirements for the worker.
