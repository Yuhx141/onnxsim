#include "remote_transport.h"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>

using namespace onnx_remote;

static uint64_t micros_since(
    const std::chrono::steady_clock::time_point& start) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start).count());
}

static bool broadcast_shape(const Tensor& a, const Tensor& b,
                            std::vector<int64_t>& shape) {
  const size_t rank = std::max(a.shape.size(), b.shape.size());
  shape.assign(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t adim = i < rank - a.shape.size()
                             ? 1
                             : a.shape[i - (rank - a.shape.size())];
    const int64_t bdim = i < rank - b.shape.size()
                             ? 1
                             : b.shape[i - (rank - b.shape.size())];
    if (adim <= 0 || bdim <= 0 || (adim != bdim && adim != 1 && bdim != 1))
      return false;
    shape[i] = std::max(adim, bdim);
  }
  return true;
}

static size_t broadcast_index(size_t output_index,
                              const std::vector<int64_t>& output_shape,
                              const std::vector<int64_t>& input_shape) {
  if (input_shape.empty()) return 0;
  const size_t rank = output_shape.size();
  const size_t input_rank = input_shape.size();
  size_t input_index = 0;
  size_t output_stride = 1;
  size_t input_stride = 1;
  for (size_t axis = rank; axis-- > 0;) {
    const size_t coordinate = (output_index / output_stride) %
                              static_cast<size_t>(output_shape[axis]);
    output_stride *= static_cast<size_t>(output_shape[axis]);
    if (axis < rank - input_rank) continue;
    const int64_t input_dim = input_shape[axis - (rank - input_rank)];
    input_index += (input_dim == 1 ? 0 : coordinate) * input_stride;
    input_stride *= static_cast<size_t>(input_dim);
  }
  return input_index;
}

static Response execute(const Request& r) {
  Response out; out.request_id = r.request_id; out.ok = false;
  const auto started = std::chrono::steady_clock::now();
  auto profile = [&](const char* name, uint64_t begin, uint64_t duration) {
    if (r.profiling != ProfilingLevel::Off) {
      out.profile.push_back(ProfileEvent{name, "remote", begin, duration, "reference-worker"});
    }
  };
  auto execute_op = [&](auto&& fn) {
    const uint64_t begin = micros_since(started);
    fn();
    profile(r.op.c_str(), begin, micros_since(started) - begin);
  };
  if (r.op == "capabilities") {
    out.artifact_id = "reference-worker";
    out.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\","
        "\"runner_id\":\"reference-worker\",\"ready\":true,"
        "\"supported_ops\":[\"identity\",\"relu\",\"add\",\"mul\","
        "\"sub\",\"div\",\"max\",\"min\",\"abs\",\"neg\",\"sqrt\","
        "\"exp\",\"log\",\"tanh\"],"
        "\"supported_dtypes\":[\"FLOAT\",\"FLOAT16\",\"BFLOAT16\","
        "\"INT8\",\"UINT8\",\"INT16\",\"UINT16\",\"INT32\","
        "\"UINT32\",\"INT64\",\"UINT64\",\"DOUBLE\",\"BOOL\"],"
        "\"graph_execution\":false,"
        "\"profiling\":true}";
    out.ok = true;
    return out;
  }
  if (r.op == "identity") {
    if (r.inputs.size() != 1) { out.error = "identity expects one input"; return out; }
    execute_op([&] { out.outputs = r.inputs; }); out.ok = true; return out;
  }
  if (r.op == "relu") {
    if (r.inputs.size() != 1) { out.error = "relu expects one input"; return out; }
    if (r.inputs[0].dtype != 1) {
      out.error = "relu reference implementation supports float32 tensors only";
      return out;
    }
    Tensor y;
    execute_op([&] { y = r.inputs[0]; for (float& x : y.data) if (x < 0.0f) x = 0.0f; });
    out.outputs.push_back(std::move(y)); out.ok = true; return out;
  }
  if (r.op == "abs" || r.op == "neg" || r.op == "sqrt" ||
      r.op == "exp" || r.op == "log" || r.op == "tanh") {
    if (r.inputs.size() != 1) {
      out.error = r.op + " expects one input"; return out;
    }
    if (r.inputs[0].dtype != 1) {
      out.error = r.op + " reference implementation supports float32 tensors only";
      return out;
    }
    Tensor y = r.inputs[0];
    execute_op([&] {
      for (float& x : y.data) {
        if (r.op == "abs") x = std::fabs(x);
        else if (r.op == "neg") x = -x;
        else if (r.op == "sqrt") x = std::sqrt(x);
        else if (r.op == "exp") x = std::exp(x);
        else if (r.op == "log") x = std::log(x);
        else x = std::tanh(x);
      }
    });
    out.outputs.push_back(std::move(y)); out.ok = true; return out;
  }
  if (r.op == "add" || r.op == "mul" || r.op == "sub" ||
      r.op == "div" || r.op == "max" || r.op == "min") {
    std::vector<int64_t> output_shape;
    if (r.inputs.size() != 2 ||
        !broadcast_shape(r.inputs[0], r.inputs[1], output_shape)) {
      out.error = r.op + " expects broadcast-compatible tensors"; return out;
    }
    if (r.inputs[0].dtype != 1 || r.inputs[1].dtype != 1) {
      out.error = r.op + " reference implementation supports float32 tensors only";
      return out;
    }
    Tensor y = r.inputs[0];
    y.shape = output_shape;
    size_t output_elements = 1;
    for (const int64_t dimension : output_shape)
      output_elements *= static_cast<size_t>(dimension);
    y.data.resize(output_elements);
    execute_op([&] {
      for (size_t i = 0; i < y.data.size(); ++i) {
        const float lhs = r.inputs[0].data[broadcast_index(
            i, output_shape, r.inputs[0].shape)];
        const float rhs = r.inputs[1].data[broadcast_index(
            i, output_shape, r.inputs[1].shape)];
        if (r.op == "add") y.data[i] = lhs + rhs;
        else if (r.op == "mul") y.data[i] = lhs * rhs;
        else if (r.op == "sub") y.data[i] = lhs - rhs;
        else if (r.op == "div") y.data[i] = lhs / rhs;
        else if (r.op == "max") y.data[i] = std::max(lhs, rhs);
        else y.data[i] = std::min(lhs, rhs);
      }
    });
    out.outputs.push_back(std::move(y)); out.ok = true; return out;
  }
  out.error = "unsupported operation: " + r.op; return out;
}

int main(int argc, char** argv) {
  uint16_t port = 39501;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (std::string(argv[i]) == "--help") { std::cout << "usage: onnx-remote-worker [--port PORT]\n"; return 0; }
    else { std::cerr << "unknown argument: " << argv[i] << '\n'; return 2; }
  }
  std::signal(SIGPIPE, SIG_IGN);
  int listener = listen_tcp(port);
  if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; return 1; }
  std::cerr << "onnx-remote-worker listening on " << port << '\n';
  for (;;) {
    int fd = accept_tcp(listener); if (fd < 0) continue;
    Request request; Response response; std::string error;
    if (!receive_request(fd, request, error)) { response.error = error; }
    else response = execute(request);
    if (!response.ok && response.error.empty()) response.error = error.empty() ? "request failed" : error;
    if (!send_response(fd, response, error)) std::cerr << "response failed: " << error << '\n';
    close_socket(fd);
  }
}
