// Public ONNX Runtime EP plugin for the dependency-free remote transport.
//
// This file deliberately uses only ORT's public OrtEp C API plus the public
// C++ graph/tensor wrappers.  It is built as a standalone plugin and does not
// depend on ORT's internal C++ execution-provider ABI.

#include <onnxruntime_cxx_api.h>

#include "remote_profile.h"
#include "remote_transport.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using onnx_remote::ProfilingLevel;
using Clock = std::chrono::high_resolution_clock;

constexpr const char* kName = "onnxsim_remote";
constexpr const char* kVendor = "onnxsim";
constexpr const char* kVersion = "1.0.0";

bool SupportedOp(const std::string& op) {
  return op == "Identity" || op == "Relu" || op == "Add" || op == "Mul";
}

bool IsFloatTensor(const Ort::ConstValueInfo& value_info) {
  const auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) return false;
  return type_info.GetTensorTypeAndShapeInfo().GetElementType() ==
         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

bool SupportedNode(const Ort::ConstNode& node) {
  const std::string op = node.GetOperatorType();
  if (!SupportedOp(op)) return false;
  // The dependency-free reference worker currently implements arithmetic and
  // ReLU for float32. Identity can safely preserve the supported raw dtypes.
  if (op == "Identity") return true;
  for (const auto& value : node.GetInputs()) {
    if (!IsFloatTensor(value)) return false;
  }
  for (const auto& value : node.GetOutputs()) {
    if (!IsFloatTensor(value)) return false;
  }
  return true;
}

const char* RemoteOperation(const std::string& op) {
  if (op == "Identity") return "identity";
  if (op == "Relu") return "relu";
  if (op == "Add") return "add";
  if (op == "Mul") return "mul";
  return nullptr;
}

size_t ElementBytes(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return 8;
    default:
      return 0;
  }
}

OrtStatus* Error(const OrtApi* api, OrtErrorCode code, const std::string& text) {
  return api->CreateStatus(code, text.c_str());
}

std::string Environment(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value == nullptr || *value == '\0' ? fallback : value;
}

int EnvironmentInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  return end == value || *end != '\0' || parsed < 0 || parsed > 65535
             ? fallback
             : static_cast<int>(parsed);
}

ProfilingLevel EnvironmentProfile() {
  const std::string value = Environment("ONNXSIM_REMOTE_EP_PROFILING", "off");
  if (value == "detailed") return ProfilingLevel::Detailed;
  if (value == "summary") return ProfilingLevel::Summary;
  return ProfilingLevel::Off;
}

struct RemoteOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 39501;
  int connect_timeout_ms = 5000;
  int io_timeout_ms = 0;
  ProfilingLevel profiling = ProfilingLevel::Off;
  std::string profile_file;
};

class RemoteEp;

struct RemoteProfiler final : OrtEpProfilerImpl {
  explicit RemoteProfiler(const OrtEpApi* api) : ep_api(api) {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    StartProfiling = StartProfilingImpl;
    StartEvent = StartEventImpl;
    StopEvent = StopEventImpl;
    EndProfiling = EndProfilingImpl;
  }

  struct Event {
    std::string name;
    Clock::time_point start;
    int64_t duration_us;
  };

  const OrtEpApi* ep_api;
  int64_t start_offset_ns = 0;
  Clock::time_point start_time;
  std::mutex mutex;
  std::vector<Event> events;

  void Record(const std::vector<onnx_remote::ProfileEvent>& remote_events,
              Clock::time_point local_start) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& event : remote_events) {
      events.push_back({event.name, local_start,
                        static_cast<int64_t>(std::min<uint64_t>(
                            event.duration_us,
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())))});
    }
  }

  static thread_local RemoteProfiler* active;

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* value) noexcept {
    delete static_cast<RemoteProfiler*>(value);
  }

  static OrtStatus* ORT_API_CALL StartProfilingImpl(
      OrtEpProfilerImpl* value, int64_t offset_ns) noexcept {
    auto* profiler = static_cast<RemoteProfiler*>(value);
    profiler->start_offset_ns = offset_ns;
    profiler->start_time = Clock::now();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL StartEventImpl(
      OrtEpProfilerImpl* value, uint64_t) noexcept {
    active = static_cast<RemoteProfiler*>(value);
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL StopEventImpl(
      OrtEpProfilerImpl*, uint64_t, const OrtProfilingEvent*) noexcept {
    active = nullptr;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL EndProfilingImpl(
      OrtEpProfilerImpl* value,
      OrtProfilingEventsContainer* container) noexcept {
    auto* profiler = static_cast<RemoteProfiler*>(value);
    try {
      std::vector<Event> events;
      {
        std::lock_guard<std::mutex> lock(profiler->mutex);
        events.swap(profiler->events);
      }
      for (const auto& event : events) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            event.start - profiler->start_time).count();
        const int64_t timestamp_us =
            (profiler->start_offset_ns + elapsed) / 1000;
        OrtProfilingEvent* ort_event = nullptr;
        const char* key = "remote_profile";
        const char* value_text = "onnx-remote-v5";
        if (OrtStatus* status = profiler->ep_api->CreateProfilingEvent(
                OrtProfilingEventCategory_KERNEL, -1, -1, event.name.c_str(),
                timestamp_us, event.duration_us, &key, &value_text, 1,
                &ort_event)) {
          return status;
        }
        const OrtProfilingEvent* events_to_add[] = {ort_event};
        if (OrtStatus* status = profiler->ep_api->ProfilingEventsContainer_AddEvents(
                container, events_to_add, 1)) {
          profiler->ep_api->ReleaseProfilingEvent(ort_event);
          return status;
        }
        profiler->ep_api->ReleaseProfilingEvent(ort_event);
      }
      return nullptr;
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, error.what());
    }
  }
};

thread_local RemoteProfiler* RemoteProfiler::active = nullptr;

struct RemoteNodeComputeInfo final : OrtNodeComputeInfo {
  RemoteNodeComputeInfo(RemoteEp& ep, std::string op);

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo*, OrtNodeComputeContext*, void** state) noexcept {
    *state = nullptr;
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* self, void*, OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo*, void*) noexcept {}

  RemoteEp& ep;
  std::string op;
};

class RemoteEp final : public OrtEp {
 public:
  RemoteEp(const OrtApi* api, const OrtEpApi* ep_api, RemoteOptions options)
      : api_(api), ep_api_(ep_api), options_(std::move(options)) {
    ort_version_supported = ORT_API_VERSION;
    GetName = [](const OrtEp* value) noexcept -> const char* {
      return static_cast<const RemoteEp*>(value)->Name();
    };
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
    CreateProfiler = CreateProfilerImpl;
  }

  const char* Name() const { return kName; }
  const OrtApi* api() const { return api_; }
  const OrtEpApi* ep_api() const { return ep_api_; }
  const RemoteOptions& options() const { return options_; }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* self, const OrtGraph* graph,
      OrtEpGraphSupportInfo* support) noexcept {
    auto* ep = static_cast<RemoteEp*>(self);
    try {
      Ort::ConstGraph view{graph};
      for (const auto& node : view.GetNodes()) {
        if (!node.GetDomain().empty() || !SupportedNode(node)) {
          continue;
        }
        const OrtNode* raw = node;
        OrtNodeFusionOptions options{};
        options.ort_version_supported = ORT_API_VERSION;
        options.drop_constant_initializers = false;
        if (OrtStatus* status = ep->ep_api_->EpGraphSupportInfo_AddNodesToFuse(
                support, &raw, 1, &options)) {
          return status;
        }
      }
      return nullptr;
    } catch (const Ort::Exception& error) {
      return Error(ep->api_, error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      return Error(ep->api_, ORT_EP_FAIL, error.what());
    }
  }

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* self, const OrtGraph** graphs, const OrtNode**, size_t count,
      OrtNodeComputeInfo** infos, OrtNode**) noexcept {
    auto* ep = static_cast<RemoteEp*>(self);
    for (size_t i = 0; i < count; ++i) infos[i] = nullptr;
    try {
      for (size_t i = 0; i < count; ++i) {
        Ort::ConstGraph view{graphs[i]};
        const auto nodes = view.GetNodes();
        if (nodes.size() != 1 || !SupportedNode(nodes[0]) ||
            !nodes[0].GetDomain().empty()) {
          return Error(ep->api_, ORT_EP_FAIL,
                       "onnxsim_remote received an unsupported fused graph");
        }
        infos[i] = new RemoteNodeComputeInfo(*ep, nodes[0].GetOperatorType());
      }
      return nullptr;
    } catch (const Ort::Exception& error) {
      for (size_t i = 0; i < count; ++i) delete infos[i];
      return Error(ep->api_, error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      for (size_t i = 0; i < count; ++i) delete infos[i];
      return Error(ep->api_, ORT_EP_FAIL, error.what());
    }
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* self, OrtNodeComputeInfo** infos, size_t count) noexcept {
    (void)self;
    for (size_t i = 0; i < count; ++i) delete static_cast<RemoteNodeComputeInfo*>(infos[i]);
  }

  static OrtStatus* ORT_API_CALL CreateProfilerImpl(
      OrtEp* self, OrtEpProfilerImpl** profiler) noexcept {
    auto* ep = static_cast<RemoteEp*>(self);
    *profiler = new RemoteProfiler(ep->ep_api_);
    return nullptr;
  }

 private:
  const OrtApi* api_;
  const OrtEpApi* ep_api_;
  RemoteOptions options_;
};

RemoteNodeComputeInfo::RemoteNodeComputeInfo(RemoteEp& owner, std::string operation)
    : ep(owner), op(std::move(operation)) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

bool ReadTensor(const OrtApi* api, const OrtValue* value, onnx_remote::Tensor& output,
                std::string& error) {
  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (OrtStatus* status = api->GetTensorTypeAndShape(value, &info)) {
    api->ReleaseStatus(status);
    error = "cannot inspect ORT input tensor";
    return false;
  }
  ONNXTensorElementDataType type;
  size_t rank = 0;
  size_t elements = 0;
  if (api->GetTensorElementType(info, &type) ||
      api->GetDimensionsCount(info, &rank) ||
      api->GetTensorShapeElementCount(info, &elements)) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    error = "cannot inspect ORT tensor metadata";
    return false;
  }
  const size_t bytes = ElementBytes(type);
  if (bytes == 0) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    error = "onnxsim_remote received an unsupported tensor dtype";
    return false;
  }
  output.dtype = static_cast<uint8_t>(type);
  output.shape.resize(rank);
  if (api->GetDimensions(info, output.shape.data(), rank)) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    error = "cannot read ORT tensor shape";
    return false;
  }
  const void* data = nullptr;
  if (api->GetTensorData(value, &data)) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    error = "cannot read ORT tensor data";
    return false;
  }
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const auto* floats = static_cast<const float*>(data);
    output.data.assign(floats, floats + elements);
  } else {
    const auto* bytes_data = static_cast<const uint8_t*>(data);
    output.raw_data.assign(bytes_data, bytes_data + elements * bytes);
  }
  api->ReleaseTensorTypeAndShapeInfo(info);
  return true;
}

OrtStatus* ORT_API_CALL RemoteNodeComputeInfo::ComputeImpl(
    OrtNodeComputeInfo* self, void*, OrtKernelContext* context) noexcept {
  auto* info = static_cast<RemoteNodeComputeInfo*>(self);
  const OrtApi* api = info->ep.api();
  try {
    size_t input_count = 0;
    size_t output_count = 0;
    if (api->KernelContext_GetInputCount(context, &input_count) ||
        api->KernelContext_GetOutputCount(context, &output_count) ||
        output_count != 1) {
      return Error(api, ORT_INVALID_ARGUMENT, "onnxsim_remote expects one output");
    }
    onnx_remote::Request request;
    request.op = RemoteOperation(info->op);
    request.profiling = info->ep.options().profiling;
    request.inputs.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      const OrtValue* input = nullptr;
      if (OrtStatus* status = api->KernelContext_GetInput(context, i, &input)) {
        api->ReleaseStatus(status);
        return Error(api, ORT_FAIL, "cannot read onnxsim_remote input");
      }
      onnx_remote::Tensor tensor;
      std::string error;
      if (input == nullptr || !ReadTensor(api, input, tensor, error)) {
        return Error(api, ORT_INVALID_ARGUMENT, error);
      }
      request.inputs.push_back(std::move(tensor));
    }
    const int fd = onnx_remote::connect_tcp_timeout(
        info->ep.options().host, info->ep.options().port,
        info->ep.options().connect_timeout_ms);
    if (fd < 0) return Error(api, ORT_FAIL, "onnxsim_remote connection failed");
    if (!onnx_remote::set_socket_io_timeout(fd, info->ep.options().io_timeout_ms)) {
      onnx_remote::close_socket(fd);
      return Error(api, ORT_FAIL, "onnxsim_remote socket timeout setup failed");
    }
    std::string transport_error;
    onnx_remote::Response response;
    const auto local_start = Clock::now();
    const bool ok = onnx_remote::send_request(fd, request, transport_error) &&
                    onnx_remote::receive_response(fd, response, transport_error);
    onnx_remote::close_socket(fd);
    if (!ok || !response.ok) {
      return Error(api, ORT_FAIL,
                   transport_error.empty() ? response.error : transport_error);
    }
    if (auto* profiler = RemoteProfiler::active) {
      profiler->Record(response.profile, local_start);
    }
    if (response.outputs.size() != 1) {
      return Error(api, ORT_FAIL, "onnxsim_remote returned an unexpected output count");
    }
    const auto& result = response.outputs[0];
    const size_t element_bytes = ElementBytes(
        static_cast<ONNXTensorElementDataType>(result.dtype));
    if (element_bytes == 0) {
      return Error(api, ORT_FAIL, "onnxsim_remote returned an unsupported output dtype");
    }
    size_t element_count = 1;
    for (int64_t dimension : result.shape) {
      if (dimension < 0 ||
          static_cast<uint64_t>(dimension) >
              std::numeric_limits<size_t>::max() / element_count) {
        return Error(api, ORT_FAIL, "onnxsim_remote returned an invalid output shape");
      }
      element_count *= static_cast<size_t>(dimension);
    }
    const size_t result_bytes = result.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                                    ? result.data.size() * sizeof(float)
                                    : result.raw_data.size();
    if (element_count > std::numeric_limits<size_t>::max() / element_bytes ||
        result_bytes != element_count * element_bytes) {
      return Error(api, ORT_FAIL,
                   "onnxsim_remote output shape and payload size mismatch");
    }
    OrtValue* output = nullptr;
    if (OrtStatus* status = api->KernelContext_GetOutput(
            context, 0, result.shape.data(), result.shape.size(), &output)) {
      api->ReleaseStatus(status);
      return Error(api, ORT_FAIL, "cannot allocate onnxsim_remote output");
    }
    void* destination = nullptr;
    if (OrtStatus* status = api->GetTensorMutableData(output, &destination)) {
      api->ReleaseStatus(status);
      return Error(api, ORT_FAIL, "cannot access onnxsim_remote output");
    }
    const size_t byte_count = result_bytes;
    if (result.dtype != 1) {
      std::memcpy(destination, result.raw_data.data(), byte_count);
    } else {
      std::memcpy(destination, result.data.data(), byte_count);
    }
    if (!info->ep.options().profile_file.empty() && !response.profile.empty()) {
      static std::mutex profile_mutex;
      std::lock_guard<std::mutex> lock(profile_mutex);
      std::ofstream profile(info->ep.options().profile_file,
                            std::ios::app | std::ios::binary);
      if (profile) profile << onnx_remote::profile_json(response) << '\n';
    }
    return nullptr;
  } catch (const std::exception& error) {
    return Error(api, ORT_EP_FAIL, error.what());
  }
}

class RemoteFactory final : public OrtEpFactory {
 public:
  RemoteFactory(const OrtApi* api, const OrtEpApi* ep_api,
                RemoteOptions options)
      : api_(api), ep_api_(ep_api), options_(std::move(options)) {
    ort_version_supported = ORT_API_VERSION;
    GetName = [](const OrtEpFactory*) noexcept -> const char* { return kName; };
    GetVendor = [](const OrtEpFactory*) noexcept -> const char* { return kVendor; };
    GetVersion = [](const OrtEpFactory*) noexcept -> const char* { return kVersion; };
    GetVendorId = [](const OrtEpFactory*) noexcept -> uint32_t { return 0; };
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
    GetCustomOpDomains = GetCustomOpDomainsImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
    GetHardwareDeviceIncompatibilityDetails =
        GetHardwareDeviceIncompatibilityDetailsImpl;
    if (OrtStatus* status = api_->CreateCpuMemoryInfo(
            OrtDeviceAllocator, OrtMemTypeDefault, &memory_info_)) {
      api_->ReleaseStatus(status);
      memory_info_ = nullptr;
    }
  }

  ~RemoteFactory() {
    if (memory_info_ != nullptr) api_->ReleaseMemoryInfo(memory_info_);
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* self, const OrtHardwareDevice* const* devices,
      size_t device_count, OrtEpDevice** ep_devices, size_t max_devices,
      size_t* result_count) noexcept {
    auto* factory = static_cast<RemoteFactory*>(self);
    *result_count = 0;
    for (size_t i = 0; i < device_count && *result_count < max_devices; ++i) {
      if (factory->api_->HardwareDevice_Type(devices[i]) !=
          OrtHardwareDeviceType_CPU) {
        continue;
      }
      OrtKeyValuePairs* metadata = nullptr;
      OrtKeyValuePairs* options = nullptr;
      factory->api_->CreateKeyValuePairs(&metadata);
      factory->api_->CreateKeyValuePairs(&options);
      factory->api_->AddKeyValuePair(metadata, "transport", "onnx-remote-v5");
      OrtEpDevice* ep_device = nullptr;
      if (OrtStatus* status = factory->ep_api_->CreateEpDevice(
              self, devices[i], metadata, options, &ep_device)) {
        factory->api_->ReleaseKeyValuePairs(metadata);
        factory->api_->ReleaseKeyValuePairs(options);
        return status;
      }
      factory->api_->ReleaseKeyValuePairs(metadata);
      factory->api_->ReleaseKeyValuePairs(options);
      if (factory->memory_info_ != nullptr) {
        if (OrtStatus* status = factory->ep_api_->EpDevice_AddAllocatorInfo(
                ep_device, factory->memory_info_)) {
          return status;
        }
      }
      ep_devices[(*result_count)++] = ep_device;
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* self, const OrtHardwareDevice* const*,
      const OrtKeyValuePairs* const*, size_t device_count,
      const OrtSessionOptions*, const OrtLogger*, OrtEp** output) noexcept {
    auto* factory = static_cast<RemoteFactory*>(self);
    if (device_count != 1) {
      return Error(factory->api_, ORT_INVALID_ARGUMENT,
                   "onnxsim_remote expects one selected device");
    }
    *output = new RemoteEp(factory->api_, factory->ep_api_, factory->options_);
    return nullptr;
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory*, OrtEp* ep) noexcept {
    delete static_cast<RemoteEp*>(ep);
  }

  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(
      OrtEpFactory*, size_t* count) noexcept {
    *count = 0;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(
      OrtEpFactory*, OrtCustomOpDomain**, size_t) noexcept {
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEpFactory*, const OrtMemoryInfo*, const OrtKeyValuePairs*,
      OrtAllocator** allocator) noexcept {
    return Ort::GetApi().GetAllocatorWithDefaultOptions(allocator);
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory*, OrtAllocator*) noexcept {}

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory*, OrtDataTransferImpl** transfer) noexcept {
    *transfer = nullptr;
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory*) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      OrtEpFactory*, const OrtHardwareDevice*,
      OrtDeviceEpIncompatibilityDetails*) noexcept {
    return nullptr;
  }

 private:
  const OrtApi* api_;
  const OrtEpApi* ep_api_;
  RemoteOptions options_;
  OrtMemoryInfo* memory_info_ = nullptr;
};

RemoteOptions OptionsFromEnvironment() {
  RemoteOptions options;
  options.host = Environment("ONNXSIM_REMOTE_EP_HOST", "127.0.0.1");
  options.port = static_cast<uint16_t>(EnvironmentInt("ONNXSIM_REMOTE_EP_PORT", 39501));
  options.connect_timeout_ms = EnvironmentInt("ONNXSIM_REMOTE_EP_CONNECT_TIMEOUT_MS", 5000);
  options.io_timeout_ms = EnvironmentInt("ONNXSIM_REMOTE_EP_IO_TIMEOUT_MS", 0);
  options.profiling = EnvironmentProfile();
  options.profile_file = Environment("ONNXSIM_REMOTE_EP_PROFILE_FILE", "");
  return options;
}

}  // namespace

extern "C" OrtStatus* ORT_API_CALL CreateEpFactories(
    const char*, const OrtApiBase* ort_api_base, const OrtLogger*,
    OrtEpFactory** factories, size_t max_factories,
    size_t* num_factories) {
  if (factories == nullptr || num_factories == nullptr || max_factories == 0) {
    return nullptr;
  }
  Ort::InitApi(ort_api_base->GetApi(ORT_API_VERSION));
  const OrtApi* api = &Ort::GetApi();
  const OrtEpApi* ep_api = &Ort::GetEpApi();
  factories[0] = new RemoteFactory(api, ep_api, OptionsFromEnvironment());
  *num_factories = 1;
  return nullptr;
}

extern "C" OrtStatus* ORT_API_CALL ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<RemoteFactory*>(factory);
  return nullptr;
}
