#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: onnxsim_remote_ep_run_test PLUGIN MODEL.onnx [ORT_PROFILE_PREFIX]\n";
    return 2;
  }
  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnxsim-remote-ep-run-test"};
    env.RegisterExecutionProviderLibrary("onnxsim_remote", argv[1]);
    const auto devices = env.GetEpDevices();
    auto device = std::find_if(devices.begin(), devices.end(), [](const auto& value) {
      return std::string(value.EpName()) == "onnxsim_remote";
    });
    if (device == devices.end()) {
      std::cerr << "onnxsim_remote device was not discovered\n";
      return 1;
    }
    Ort::SessionOptions options;
    if (argc == 4) options.EnableProfiling(argv[3]);
    options.AppendExecutionProvider_V2(env, {*device}, Ort::KeyValuePairs{});
    Ort::Session session{env, argv[2], options};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_name_storage;
    std::vector<const char*> input_names;
    std::vector<std::vector<float>> input_storage;
    std::vector<Ort::Value> inputs;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      input_name_storage.emplace_back(name.get());
      input_names.push_back(input_name_storage.back().c_str());
      auto type_info = session.GetInputTypeInfo(i);
      auto shape_info = type_info.GetTensorTypeAndShapeInfo();
      if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::cerr << "runtime test requires float32 inputs\n";
        return 1;
      }
      auto shape = shape_info.GetShape();
      size_t elements = 1;
      for (auto& dimension : shape) {
        if (dimension <= 0) dimension = 2;
        elements *= static_cast<size_t>(dimension);
      }
      input_storage.emplace_back(elements, static_cast<float>(i + 1));
      inputs.push_back(Ort::Value::CreateTensor<float>(
          memory, input_storage.back().data(), elements, shape.data(),
          shape.size()));
    }
    std::vector<std::string> output_name_storage;
    std::vector<const char*> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
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
    const float* result = outputs[0].GetTensorData<float>();
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
    env.UnregisterExecutionProviderLibrary("onnxsim_remote");
    std::cout << "public OrtEp remote execution passed\n";
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
