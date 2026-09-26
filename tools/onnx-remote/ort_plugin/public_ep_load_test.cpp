#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: onnxsim_remote_ep_load_test PLUGIN\n";
    return 2;
  }
  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnxsim-remote-ep-test"};
    env.RegisterExecutionProviderLibrary("onnxsim_remote", argv[1]);
    bool found = false;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "onnxsim_remote") {
        found = true;
        break;
      }
    }
    env.UnregisterExecutionProviderLibrary("onnxsim_remote");
    if (!found) {
      std::cerr << "registered plugin did not advertise an EP device\n";
      return 1;
    }
    std::cout << "public OrtEp plugin load/discovery passed\n";
    return 0;
  } catch (const Ort::Exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
