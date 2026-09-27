#pragma once

// This adapter intentionally uses ONNX Runtime's internal EP interface. ORT
// does not expose IExecutionProvider in its public release headers, so the
// target must be built against the matching ORT source tree (currently 1.29).
#ifdef ONNXSIM_REMOTE_ORT_EP

#include <memory>
#include <string>
#include <unordered_set>

#include "core/framework/execution_provider.h"
#include "remote_transport.h"

namespace onnxsim::ort_remote {

struct Options {
  std::string host = "127.0.0.1";
  uint16_t port = 39501;
  int connect_timeout_ms = 5000;
  int io_timeout_ms = 0;
  onnx_remote::ProfilingLevel profiling = onnx_remote::ProfilingLevel::Off;
  std::unordered_set<std::string> supported_ops{"Identity", "Relu", "Add",
                                                "Mul", "Sub", "Div", "Max",
                                                "Min", "Abs", "Neg", "Sqrt"};
};

// The caller owns the returned provider and registers it with its
// InferenceSession/SessionOptions integration. Keeping construction here makes
// the transport adapter usable by applications that already have an ORT
// internal-provider registration path without changing onnxsim's default EP.
std::unique_ptr<onnxruntime::IExecutionProvider>
CreateRemoteExecutionProvider(const Options &options = {});

// Parse the string map used by ORT provider registration. Unknown keys are
// ignored so the same map can carry application-level settings.
Options OptionsFromProviderOptions(const onnxruntime::ProviderOptions &options);

std::unique_ptr<onnxruntime::IExecutionProvider>
CreateRemoteExecutionProviderFromOptions(
    const onnxruntime::ProviderOptions &options);

} // namespace onnxsim::ort_remote

#endif // ONNXSIM_REMOTE_ORT_EP
