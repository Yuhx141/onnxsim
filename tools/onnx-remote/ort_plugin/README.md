# ONNX Runtime EP package metadata

`onnxsim_remote_ep.manifest.json` describes the currently shipped remote EP
adapter and its install-time contract. The adapter is compiled into the
`onnxsim` library when `ONNXSIM_REMOTE_ORT_EP=ON`; it is not currently a
standalone public ORT plugin shared library.

The implementation uses ORT's internal C++ `IExecutionProvider` interface and
therefore must be built against the matching ORT source checkout. Consumers
should use the manifest to reject incompatible ORT versions before loading the
onnxsim library. Provider options are documented in
`tools/onnx-remote/README.md` and include `host`, `port`, timeout values,
`profiling`, and `supported_ops`.

The public plugin ABI is intentionally represented as an explicit migration
contract rather than a fake entry point. A loadable public ORT plugin must
export `CreateEpFactories` and `ReleaseEpFactory` and implement the public
`OrtEpFactory`/`OrtEp` C API. Until that adapter exists, use ORT's internal C++
registration path shown in the main README.
