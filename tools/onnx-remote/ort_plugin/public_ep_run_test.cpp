#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
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
    const std::array<int64_t, 1> shape{256};
    std::array<float, 256> left{};
    std::array<float, 256> right{};
    left.fill(1.0f);
    right.fill(3.0f);
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    auto lhs = Ort::Value::CreateTensor<float>(memory, left.data(), left.size(),
                                                shape.data(), shape.size());
    auto rhs = Ort::Value::CreateTensor<float>(memory, right.data(), right.size(),
                                                shape.data(), shape.size());
    const char* input_names[] = {"a", "b"};
    const char* output_names[] = {"c"};
    std::array<Ort::Value, 2> inputs{std::move(lhs), std::move(rhs)};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                               inputs.data(), 2, output_names, 1);
    const float* result = outputs[0].GetTensorData<float>();
    if (std::fabs(result[0] - 4.0f) > 1e-6f ||
        std::fabs(result[255] - 4.0f) > 1e-6f) {
      std::cerr << "unexpected remote EP result\n";
      return 1;
    }
    if (argc == 4) {
      Ort::AllocatorWithDefaultOptions allocator;
      auto profile_path_owner = session.EndProfilingAllocated(allocator);
      const std::string profile_path = profile_path_owner.get();
      std::ifstream profile(profile_path);
      const std::string contents((std::istreambuf_iterator<char>(profile)),
                                 std::istreambuf_iterator<char>());
      if (contents.find("onnxsim_remote") == std::string::npos) {
        std::cerr << "ORT profile did not contain the remote EP event\n";
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
