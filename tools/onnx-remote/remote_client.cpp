#include "remote_transport.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

using namespace onnx_remote;

static bool parse_floats(const std::string& text, std::vector<float>& values,
                         std::string& error) {
  values.clear();
  std::string item;
  std::istringstream stream(text);
  while (std::getline(stream, item, ',')) {
    char* end = nullptr;
    const float value = std::strtof(item.c_str(), &end);
    if (end == item.c_str() || *end != '\0' || !std::isfinite(value)) {
      error = "invalid float value: " + item;
      return false;
    }
    values.push_back(value);
  }
  if (values.empty()) error = "input values must not be empty";
  return !values.empty();
}

static int subgraph_request(int argc, char** argv) {
  // onnx-remote-client --subgraph HOST PORT MODEL.onnx V0,V1,... [--shape D0,D1]
  // Repeat [--input VALUES [--shape DIMS]] for additional model inputs.
  std::string shape_text;
  int positional = 0;
  std::string host, port_text, model_path, values_text;
  std::vector<std::pair<std::string, std::string>> extra_inputs;
  for (int i = 2; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--shape" && i + 1 < argc) {
      shape_text = argv[++i];
    } else if (argument == "--input" && i + 1 < argc) {
      std::string spec = argv[++i];
      std::string input_shape;
      const size_t at = spec.find('@');
      if (at != std::string::npos) {
        input_shape = spec.substr(at + 1);
        spec.resize(at);
      }
      extra_inputs.emplace_back(spec, input_shape);
    } else if (positional == 0) {
      host = argument;
      ++positional;
    } else if (positional == 1) {
      port_text = argument;
      ++positional;
    } else if (positional == 2) {
      model_path = argument;
      ++positional;
    } else if (positional == 3) {
      values_text = argument;
      ++positional;
    } else {
      std::cerr << "unexpected argument: " << argument << '\n';
      return 2;
    }
  }
  if (positional != 4) {
    std::cerr << "usage: onnx-remote-client --subgraph HOST PORT MODEL.onnx "
                 "V0,V1,... [--shape D0,D1] [--input VALUES[@D0,D1] ...]\n";
    return 2;
  }
  std::ifstream model_file(model_path, std::ios::binary);
  if (!model_file) {
    std::cerr << "cannot open model\n";
    return 1;
  }
  Request request;
  request.op = kSubgraphOperation;
  request.profiling = ProfilingLevel::Detailed;
  request.model.assign(std::istreambuf_iterator<char>(model_file), {});
  auto append_input = [&](const std::string& values_text,
                          const std::string& shape_text) {
    std::string error;
    std::vector<float> values;
    if (!parse_floats(values_text, values, error)) {
      std::cerr << error << '\n';
      return false;
    }
    Tensor input;
    if (shape_text.empty()) {
      input.shape = {static_cast<int64_t>(values.size())};
    } else {
      std::string item;
      std::istringstream stream(shape_text);
      while (std::getline(stream, item, ',')) {
        char* end = nullptr;
        const long dimension = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || dimension <= 0) {
          std::cerr << "invalid shape dimension: " << item << '\n';
          return false;
        }
        input.shape.push_back(dimension);
      }
      size_t elements = 1;
      for (int64_t dimension : input.shape) elements *= static_cast<size_t>(dimension);
      if (elements != values.size()) {
        std::cerr << "values count does not match shape\n";
        return false;
      }
    }
    input.data = values;
    request.inputs.push_back(std::move(input));
    return true;
  };
  if (!append_input(values_text, shape_text)) return 2;
  for (const auto& extra : extra_inputs) {
    if (!append_input(extra.first, extra.second)) return 2;
  }
  const int fd = connect_tcp(host,
                             static_cast<uint16_t>(std::strtoul(port_text.c_str(), nullptr, 10)));
  if (fd < 0) {
    std::cerr << "connect failed\n";
    return 1;
  }
  std::string error;
  Response response;
  const bool ok =
      send_request(fd, request, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) {
    std::cerr << (error.empty() ? response.error : error) << '\n';
    return 1;
  }
  for (size_t i = 0; i < response.outputs.size(); ++i) {
    const Tensor& output = response.outputs[i];
    std::cout << "output[" << i << "] dtype=" << static_cast<int>(output.dtype)
              << " shape=[";
    for (size_t j = 0; j < output.shape.size(); ++j) {
      if (j != 0) std::cout << ',';
      std::cout << output.shape[j];
    }
    std::cout << "]\n";
    for (float value : output.data) std::cout << value << '\n';
  }
  for (const ProfileEvent& event : response.profile) {
    std::cout << "profile: " << event.name << " " << event.duration_us
              << "us " << event.detail << '\n';
  }
  return 0;
}

static int self_test() {
  Request payload_request;
  payload_request.request_id = 7;
  payload_request.op = "relu";
  payload_request.profiling = ProfilingLevel::Detailed;
  payload_request.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::vector<uint8_t> payload;
  std::string error;
  if (!encode_request_payload(payload_request, payload, error)) {
    std::cerr << "request payload encode failed: " << error << '\n';
    return 1;
  }
  Request decoded_request;
  if (!decode_request_payload(payload.data(), payload.size(), decoded_request, error) ||
      decoded_request.op != payload_request.op ||
      decoded_request.profiling != payload_request.profiling) {
    std::cerr << "request payload round-trip failed\n";
    return 1;
  }
  struct Operation {
    const char* name;
    std::vector<float> lhs;
    std::vector<float> rhs;
    std::vector<float> expected;
  };
  const std::vector<Operation> operations = {
      {"relu", {-2, -1, 0, 1, 2}, {}, {0, 0, 0, 1, 2}},
      {"add", {1, 2, 3}, {4, 5, 6}, {5, 7, 9}},
      {"mul", {1, 2, 3}, {4, 5, 6}, {4, 10, 18}},
      {"sub", {1, 2, 3}, {4, 5, 6}, {-3, -3, -3}},
      {"div", {4, 9, 12}, {2, 3, 4}, {2, 3, 3}},
      {"max", {1, 5, 3}, {4, 2, 6}, {4, 5, 6}},
      {"min", {1, 5, 3}, {4, 2, 6}, {1, 2, 3}},
      {"abs", {-2, 0, 3}, {}, {2, 0, 3}},
      {"neg", {-2, 0, 3}, {}, {2, 0, -3}},
      {"sqrt", {0, 1, 4}, {}, {0, 1, 2}},
      {"exp", {0, 1, 2}, {}, {1, std::exp(1.0f), std::exp(2.0f)}},
      {"log", {1, std::exp(1.0f), std::exp(2.0f)}, {}, {0, 1, 2}},
      {"tanh", {-1, 0, 1}, {}, {-0.76159414f, 0, 0.76159414f}},
  };
  size_t profile_events = 0;
  for (const Operation& operation : operations) {
    int fd = connect_tcp("127.0.0.1", 39501);
    if (fd < 0) {
      std::cerr << "connect failed (start onnx-remote-worker --port 39501)\n";
      return 1;
    }
    Request request;
    request.request_id = 7 + profile_events;
    request.op = operation.name;
    request.profiling = ProfilingLevel::Detailed;
    request.inputs.push_back(Tensor{{static_cast<int64_t>(operation.lhs.size())},
                                    operation.lhs});
    if (!operation.rhs.empty())
      request.inputs.push_back(Tensor{{static_cast<int64_t>(operation.rhs.size())},
                                      operation.rhs});
    if (!send_request(fd, request, error)) {
      std::cerr << error << '\n'; close_socket(fd); return 1;
    }
    Response response;
    if (!receive_response(fd, response, error) || !response.ok ||
        response.outputs.size() != 1) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      close_socket(fd); return 1;
    }
    if (response.request_id != request.request_id ||
        response.outputs[0].data.size() != operation.expected.size()) {
      std::cerr << "response mismatch for " << operation.name << '\n';
      close_socket(fd); return 1;
    }
    for (size_t i = 0; i < operation.expected.size(); ++i) {
      if (std::fabs(response.outputs[0].data[i] - operation.expected[i]) >
          1e-6f) {
        std::cerr << "value mismatch for " << operation.name << '\n';
        close_socket(fd); return 1;
      }
    }
    profile_events += response.profile.size();
    close_socket(fd);
  }
  {
    int fd = connect_tcp("127.0.0.1", 39501);
    if (fd < 0) return 1;
    Request request;
    request.request_id = 99;
    request.op = "add";
    request.profiling = ProfilingLevel::Detailed;
    request.inputs.push_back(Tensor{{2, 3}, {1, 2, 3, 1, 2, 3}});
    request.inputs.push_back(Tensor{{3}, {4, 5, 6}});
    if (!send_request(fd, request, error)) {
      close_socket(fd);
      return 1;
    }
    Response response;
    if (!receive_response(fd, response, error) || !response.ok ||
        response.outputs.size() != 1 || response.outputs[0].shape !=
            std::vector<int64_t>({2, 3}) ||
        response.outputs[0].data !=
            std::vector<float>({5, 7, 9, 5, 7, 9})) {
      std::cerr << "broadcast add self-test failed\n";
      close_socket(fd);
      return 1;
    }
    profile_events += response.profile.size();
    close_socket(fd);
  }
  std::cout << "remote transport self-test passed (profile events: "
            << profile_events << ")\n";
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
  if (argc >= 2 && std::string(argv[1]) == "--subgraph")
    return subgraph_request(argc, argv);
  if (argc == 4 && std::string(argv[1]) == "--capabilities") {
    int fd = connect_tcp(argv[2], static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)));
    if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
    Request request;
    request.op = "capabilities";
    std::string error;
    Response response;
    const bool ok = send_request(fd, request, error) &&
                    receive_response(fd, response, error);
    close_socket(fd);
    if (!ok || !response.ok) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      return 1;
    }
    std::cout << response.manifest << '\n';
    return 0;
  }
  if (argc == 5 && std::string(argv[1]) == "--compile") {
    std::ifstream input(argv[4], std::ios::binary);
    if (!input) { std::cerr << "cannot open model\n"; return 1; }
    Request request;
    request.op = "compile";
    request.model.assign(std::istreambuf_iterator<char>(input), {});
    int fd = connect_tcp(argv[2], static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)));
    if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
    std::string error;
    Response response;
    bool ok = send_request(fd, request, error) && receive_response(fd, response, error);
    close_socket(fd);
    if (!ok || !response.ok) {
      std::cerr << (error.empty() ? response.error : error) << '\n';
      return 1;
    }
    std::cout << "artifact_id=" << response.artifact_id
              << " bytes=" << response.artifact.size() << '\n'
              << response.manifest << '\n';
    return 0;
  }
  if (argc != 4) {
    std::cerr << "usage: onnx-remote-client HOST PORT OP\n"
                 "   or: onnx-remote-client --subgraph HOST PORT MODEL.onnx "
                 "V0,V1,... [--shape D0,D1]\n";
    return 2;
  }
  int fd = connect_tcp(argv[1], static_cast<uint16_t>(std::strtoul(argv[2], nullptr, 10)));
  if (fd < 0) { std::cerr << "connect failed\n"; return 1; }
  Request r; r.op = argv[3]; r.inputs.push_back(Tensor{{5}, {-2, -1, 0, 1, 2}});
  std::string error; Response response;
  bool ok = send_request(fd, r, error) && receive_response(fd, response, error);
  close_socket(fd);
  if (!ok || !response.ok) { std::cerr << (error.empty() ? response.error : error) << '\n'; return 1; }
  for (float x : response.outputs[0].data) std::cout << x << '\n';
  return 0;
}
