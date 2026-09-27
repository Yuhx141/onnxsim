#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static size_t ElementBytes(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
    default: return 0;
  }
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: onnxsim_remote_ep_run_test PLUGIN MODEL.onnx [ORT_PROFILE_PREFIX]\n";
    return 2;
  }
  const char* verbose = std::getenv("ONNXSIM_REMOTE_EP_TEST_VERBOSE");
  try {
    Ort::Env env{verbose != nullptr && std::string(verbose) != "0"
                     ? ORT_LOGGING_LEVEL_VERBOSE
                     : ORT_LOGGING_LEVEL_WARNING,
                "onnxsim-remote-ep-run-test"};
    env.RegisterExecutionProviderLibrary("onnxsim_remote", argv[1]);
    // ORT's device discovery runs asynchronously after registration. On
    // low-core CI runners the refresh thread races session creation and
    // ORT frees EpDevice-related objects underneath CreateSession (an
    // indirect call through recycled memory inside libonnxruntime). Poll
    // the device list until two consecutive snapshots match and our EP
    // is present, so discovery has demonstrably quiesced before the
    // session is created; then read the authoritative list.
    std::string previous_snapshot;
    std::vector<Ort::ConstEpDevice> devices;
    for (int attempt = 0; attempt < 50; ++attempt) {
      devices = env.GetEpDevices();
      std::string snapshot;
      for (const auto& value : devices) {
        snapshot += value.EpName();
        snapshot += ';';
      }
      if (snapshot == previous_snapshot &&
          snapshot.find("onnxsim_remote;") != std::string::npos)
        break;
      previous_snapshot = std::move(snapshot);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (verbose != nullptr && std::string(verbose) != "0") {
      for (const auto& value : devices)
        std::cerr << "[devices] ep=" << value.EpName() << '\n';
    }
    const char* append_name = std::getenv("ONNXSIM_REMOTE_EP_TEST_APPEND");
    const std::string target = append_name != nullptr ? append_name
                                                      : "onnxsim_remote";
    auto device = std::find_if(devices.begin(), devices.end(),
                               [&target](const auto& value) {
                                 return std::string(value.EpName()) == target;
                               });
    if (device == devices.end()) {
      std::cerr << target << " device was not discovered\n";
      return 1;
    }
    Ort::SessionOptions options;
    if (argc == 4) options.EnableProfiling(argv[3]);
    options.AppendExecutionProvider_V2(env, {*device}, Ort::KeyValuePairs{});
    // The session and everything referencing it must be released before
    // UnregisterExecutionProviderLibrary dlcloses the plugin: ORT releases
    // EP kernels during session teardown, and those vtables live in the
    // library being closed.
    {
    Ort::Session session{env, argv[2], options};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session.GetInputCount();
    std::vector<std::string> input_name_storage;
    std::vector<const char*> input_names;
    // Reserve before taking c_str() pointers: pushing more names must not
    // reallocate and dangle the pointers handed to Session::Run.
    input_name_storage.reserve(input_count);
    input_names.reserve(input_count);
    std::vector<std::vector<float>> input_storage;
    std::vector<std::vector<uint8_t>> raw_input_storage;
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < input_count; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      input_name_storage.emplace_back(name.get());
      input_names.push_back(input_name_storage.back().c_str());
      auto type_info = session.GetInputTypeInfo(i);
      auto shape_info = type_info.GetTensorTypeAndShapeInfo();
      const auto type = shape_info.GetElementType();
      const size_t element_bytes = ElementBytes(type);
      if (element_bytes == 0) {
        std::cerr << "runtime test encountered an unsupported input type\n";
        return 1;
      }
      auto shape = shape_info.GetShape();
      size_t elements = 1;
      for (auto& dimension : shape) {
        if (dimension <= 0) dimension = 2;
        elements *= static_cast<size_t>(dimension);
      }
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        input_storage.emplace_back(elements, static_cast<float>(i + 1));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory, input_storage.back().data(), elements, shape.data(),
            shape.size()));
      } else {
        raw_input_storage.emplace_back(elements * element_bytes,
                                       static_cast<uint8_t>(i + 1));
        inputs.push_back(Ort::Value::CreateTensor(
            memory.GetConst(), raw_input_storage.back().data(),
            raw_input_storage.back().size(), shape.data(), shape.size(), type));
      }
    }
    std::vector<std::string> output_name_storage;
    std::vector<const char*> output_names;
    const size_t output_count = session.GetOutputCount();
    output_name_storage.reserve(output_count);
    output_names.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      output_name_storage.emplace_back(name.get());
      output_names.push_back(output_name_storage.back().c_str());
    }
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                               inputs.data(), inputs.size(), output_names.data(),
                               output_names.size());
    if (outputs.empty()) {
      std::cerr << "model returned no outputs\n";
      return 1;
    }
    const void* result = outputs[0].GetTensorRawData();
    if (result == nullptr) {
      std::cerr << "remote EP returned no output data\n";
      return 1;
    }
    if (argc == 4) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto profile_path_owner = session.EndProfilingAllocated(allocator);
      const std::string profile_path = profile_path_owner.get();
      std::ifstream profile(profile_path);
      const std::string contents((std::istreambuf_iterator<char>(profile)),
                                 std::istreambuf_iterator<char>());
      const char* expect_remote = std::getenv("ONNXSIM_REMOTE_EP_EXPECT_REMOTE");
      const bool remote_expected = expect_remote == nullptr ||
                                    std::string(expect_remote) != "0";
      const bool remote_present = contents.find("onnxsim_remote") !=
                                   std::string::npos;
      if (remote_present != remote_expected) {
        std::cerr << (remote_expected
                          ? "ORT profile did not contain the remote EP event\n"
                          : "ORT profile unexpectedly contained the remote EP event\n");
        return 1;
      }
    }
    }
    env.UnregisterExecutionProviderLibrary("onnxsim_remote");
    std::cout << "public OrtEp remote execution passed\n";
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
