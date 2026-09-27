#include "remote_transport.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>

using namespace onnx_remote;

namespace {

uint64_t elapsed_us(const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

size_t element_bytes(ONNXTensorElementDataType type) {
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

Response execute(const Request& request, Ort::Env& env, int threads) {
  Response response;
  response.request_id = request.request_id;
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "ort-cpu-worker";
    response.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        "\"runner_id\":\"ort-cpu-worker\",\"ready\":true,"
        "\"supported_ops\":[\"subgraph\",\"onnx\"],"
        "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\","
        "\"INT8\",\"UINT8\",\"INT16\",\"UINT16\",\"INT32\","
        "\"UINT32\",\"INT64\",\"UINT64\",\"DOUBLE\",\"BOOL\"],"
        "\"profiling\":true}";
    return response;
  }
  if (request.op != "subgraph" && request.op != "onnx") {
    response.error = "ORT worker expects subgraph or onnx operation";
    return response;
  }
  if (request.model.empty()) {
    response.error = "ORT worker requires serialized ONNX graph bytes";
    return response;
  }
  try {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    Ort::Session session(env, request.model.data(), request.model.size(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
    if (session.GetInputCount() != request.inputs.size()) {
      response.error = "ORT worker input count mismatch";
      return response;
    }
    std::vector<std::string> input_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<Ort::Value> inputs;
    input_names.reserve(request.inputs.size());
    input_name_ptrs.reserve(request.inputs.size());
    inputs.reserve(request.inputs.size());
    for (size_t i = 0; i < request.inputs.size(); ++i) {
      const auto input_type = static_cast<ONNXTensorElementDataType>(
          request.inputs[i].dtype);
      const size_t input_bytes = element_bytes(input_type);
      if (input_bytes == 0) {
        response.error = "ORT worker received an unsupported input dtype";
        return response;
      }
      auto name = session.GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
      input_name_ptrs.push_back(input_names.back().c_str());
      const size_t input_elements = request.inputs[i].dtype == 1
                                        ? request.inputs[i].data.size()
                                        : request.inputs[i].raw_data.size() /
                                              input_bytes;
      const size_t input_size = input_elements * input_bytes;
      if (input_size == 0 && input_elements != 0) {
        response.error = "ORT worker input tensor size overflow";
        return response;
      }
      if (request.inputs[i].dtype == 1) {
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(request.inputs[i].data.data()),
            input_elements, request.inputs[i].shape.data(),
            request.inputs[i].shape.size()));
      } else {
        if (request.inputs[i].raw_data.size() != input_size) {
          response.error = "ORT worker input payload does not match dtype";
          return response;
        }
        inputs.push_back(Ort::Value::CreateTensor(
            memory.GetConst(),
            const_cast<uint8_t*>(request.inputs[i].raw_data.data()),
            input_size, request.inputs[i].shape.data(),
            request.inputs[i].shape.size(), input_type));
      }
    }
    std::vector<std::string> output_names;
    std::vector<const char*> output_name_ptrs;
    output_names.reserve(session.GetOutputCount());
    output_name_ptrs.reserve(session.GetOutputCount());
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
      output_name_ptrs.push_back(output_names.back().c_str());
    }
    const auto started = std::chrono::steady_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(),
                               inputs.data(), inputs.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    if (request.profiling != ProfilingLevel::Off) {
      response.profile.push_back(ProfileEvent{
          "ort_session_run", "onnxruntime", 0, elapsed_us(started),
          "CPU execution"});
    }
    response.outputs.reserve(outputs.size());
    for (auto& output : outputs) {
      auto info = output.GetTensorTypeAndShapeInfo();
      const auto output_type = info.GetElementType();
      const size_t output_bytes = element_bytes(output_type);
      if (output_bytes == 0) {
        response.error = "ORT worker returned an unsupported output dtype";
        response.profile.clear();
        return response;
      }
      Tensor tensor;
      tensor.shape = info.GetShape();
      const size_t count = info.GetElementCount();
      if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* data = output.GetTensorData<float>();
        tensor.data.assign(data, data + count);
      } else {
        tensor.dtype = static_cast<uint8_t>(output_type);
        const auto* data = static_cast<const uint8_t*>(output.GetTensorRawData());
        tensor.raw_data.assign(data, data + count * output_bytes);
      }
      response.outputs.push_back(std::move(tensor));
    }
    response.ok = true;
    return response;
  } catch (const Ort::Exception& error) {
    response.error = std::string("ORT worker: ") + error.what();
    return response;
  } catch (const std::exception& error) {
    response.error = std::string("ORT worker: ") + error.what();
    return response;
  }
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 39503;
  int threads = 1;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc)
      port = static_cast<uint16_t>(std::stoul(argv[++i]));
    else if (std::string(argv[i]) == "--threads" && i + 1 < argc)
      threads = std::stoi(argv[++i]);
    else if (std::string(argv[i]) == "--help") {
      std::cout << "usage: onnx-remote-ort-worker [--port PORT] [--threads N]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << argv[i] << '\n';
      return 2;
    }
  }
  if (threads <= 0) {
    std::cerr << "--threads must be positive\n";
    return 2;
  }
  std::signal(SIGPIPE, SIG_IGN);
  try {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "onnxsim-remote-ort-worker"};
    const int listener = listen_tcp(port);
    if (listener < 0) {
      std::cerr << "listen failed\n";
      return 1;
    }
    std::cerr << "onnx-remote-ort-worker listening on " << port << '\n';
    for (;;) {
      const int fd = accept_tcp(listener);
      if (fd < 0) continue;
      Request request;
      Response response;
      std::string error;
      if (!receive_request(fd, request, error)) {
        response.error = error;
      } else {
        response = execute(request, env, threads);
      }
      if (!send_response(fd, response, error))
        std::cerr << "response failed: " << error << '\n';
      close_socket(fd);
    }
  } catch (const Ort::Exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
