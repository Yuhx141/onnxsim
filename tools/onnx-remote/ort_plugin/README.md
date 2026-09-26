# ONNX Runtime EP package metadata

`onnxsim_remote_ep.manifest.json` describes the currently shipped remote EP
adapter and its install-time contract. The legacy adapter is compiled into the
`onnxsim` library when `ONNXSIM_REMOTE_ORT_EP=ON`; the public adapter is built
as the standalone `onnxsim_remote_ep` library with
`ONNXSIM_REMOTE_ORT_PUBLIC_PLUGIN=ON`.

The implementation uses ORT's legacy internal C++ `IExecutionProvider` interface
and therefore must be built against a matching ORT source checkout that still
contains `core/framework/execution_provider.h`. Current public ORT source
releases such as 1.29 do not expose that header; CMake rejects them before
compilation. Consumers should use the manifest to reject incompatible ORT
versions before loading the onnxsim library. Provider options are documented in
`tools/onnx-remote/README.md` and include `host`, `port`, timeout values,
`profiling`, and `supported_ops`.

The standalone public plugin exports `CreateEpFactories` and
`ReleaseEpFactory` and implements the public `OrtEpFactory`/`OrtEp` C API. It
advertises CPU-backed remote execution for `Identity`, `Relu`, `Add`, and
`Mul`, marshaling each fused node through `onnx-remote-v5`. Set
`ONNXSIM_REMOTE_EP_HOST`, `ONNXSIM_REMOTE_EP_PORT`, and optionally
`ONNXSIM_REMOTE_EP_PROFILING` or `ONNXSIM_REMOTE_EP_PROFILE_FILE` before
creating the ORT session. ORT session/run profiling receives the remote events
through the public `OrtEpProfilerImpl` bridge; the profile-file option also
keeps a transport-native JSON copy for ROS2, DORA, or browser tooling.

When a public ORT SDK library is supplied, the CMake target also builds
`onnxsim_remote_ep_load_test` and `onnxsim_remote_ep_run_test` for plugin
discovery and real remote execution coverage.
