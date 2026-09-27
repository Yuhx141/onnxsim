#include "remote_transport.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace onnx_remote;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: onnx-remote-ort-worker-test MODEL.onnx PORT\n";
    return 2;
  }
  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnxsim-remote-ort-worker-test"};
    Ort::SessionOptions options;
    Ort::Session local(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    {
      Request capabilities;
      capabilities.request_id = 7000;
      capabilities.op = "capabilities";
      const int capability_fd = connect_tcp_timeout(
          "127.0.0.1", static_cast<uint16_t>(std::stoul(argv[2])), 2000);
      if (capability_fd < 0) throw std::runtime_error("cannot connect to ORT worker");
      std::string capability_error;
      if (!send_request(capability_fd, capabilities, capability_error))
        throw std::runtime_error(capability_error);
      Response capability_response;
      if (!receive_response(capability_fd, capability_response,
                             capability_error))
        throw std::runtime_error(capability_error);
      close_socket(capability_fd);
      if (!capability_response.ok ||
          capability_response.manifest.find("ort-cpu-worker") ==
              std::string::npos)
        throw std::runtime_error("ORT worker capability response mismatch");
    }
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<std::vector<float>> input_storage;
    std::vector<Ort::Value> local_inputs;
    Request request;
    request.request_id = 7001;
    request.op = "subgraph";
    request.profiling = ProfilingLevel::Detailed;
    std::ifstream model_file(argv[1], std::ios::binary);
    if (!model_file) throw std::runtime_error("cannot open test model");
    request.model.assign(std::istreambuf_iterator<char>(model_file), {});
    for (size_t i = 0; i < local.GetInputCount(); ++i) {
      auto name = local.GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
      input_name_ptrs.push_back(input_names.back().c_str());
      auto type_info = local.GetInputTypeInfo(i);
      auto info = type_info.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("test model must use float32 inputs (dtype=" +
                                 std::to_string(info.GetElementType()) + ")");
      auto shape = info.GetShape();
      size_t elements = 1;
      for (auto& dimension : shape) {
        if (dimension <= 0) dimension = 2;
        elements *= static_cast<size_t>(dimension);
      }
      input_storage.emplace_back(elements, static_cast<float>(i + 1));
      local_inputs.push_back(Ort::Value::CreateTensor<float>(
          memory, input_storage.back().data(), elements, shape.data(),
          shape.size()));
      request.inputs.push_back(Tensor{shape, input_storage.back()});
    }
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    for (size_t i = 0; i < local.GetOutputCount(); ++i) {
      auto name = local.GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
      output_name_ptrs.push_back(output_names.back().c_str());
    }
    auto expected = local.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(),
                              local_inputs.data(), local_inputs.size(),
                              output_name_ptrs.data(), output_name_ptrs.size());

    const int fd = connect_tcp_timeout("127.0.0.1",
                                       static_cast<uint16_t>(std::stoul(argv[2])),
                                       2000);
    if (fd < 0) throw std::runtime_error("cannot connect to ORT worker");
    std::string error;
    if (!send_request(fd, request, error)) throw std::runtime_error(error);
    Response response;
    if (!receive_response(fd, response, error)) throw std::runtime_error(error);
    close_socket(fd);
    if (!response.ok || response.outputs.size() != expected.size() ||
        response.profile.empty())
      throw std::runtime_error(response.error.empty()
                                   ? "remote ORT response mismatch"
                                   : response.error);
    for (size_t i = 0; i < expected.size(); ++i) {
      auto info = expected[i].GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("test model must use float32 outputs");
      const size_t count = info.GetElementCount();
      const float* values = expected[i].GetTensorData<float>();
      if (response.outputs[i].data.size() != count)
        throw std::runtime_error("remote output size mismatch");
      for (size_t j = 0; j < count; ++j) {
        if (std::fabs(values[j] - response.outputs[i].data[j]) > 1e-5f)
          throw std::runtime_error("remote output value mismatch");
      }
    }
    request.request_id = 7002;
    const int cached_fd = connect_tcp_timeout(
        "127.0.0.1", static_cast<uint16_t>(std::stoul(argv[2])), 2000);
    if (cached_fd < 0) throw std::runtime_error("cannot reconnect to ORT worker");
    if (!send_request(cached_fd, request, error)) throw std::runtime_error(error);
    Response cached_response;
    if (!receive_response(cached_fd, cached_response, error))
      throw std::runtime_error(error);
    close_socket(cached_fd);
    if (!cached_response.ok || cached_response.profile.size() < 2 ||
        cached_response.profile.front().name != "ort_session_cache_hit")
      throw std::runtime_error("ORT worker session cache was not reused");
    std::cout << "ORT graph worker integration passed (profile events: "
              << cached_response.profile.size() << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
