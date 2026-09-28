#include "remote_profile.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace onnx_remote;

namespace {

bool read_input(const std::string& path, std::vector<uint8_t>& bytes,
                std::string& error) {
  if (path == "-") {
    std::istreambuf_iterator<char> begin(std::cin);
    std::istreambuf_iterator<char> end;
    for (; begin != end; ++begin) bytes.push_back(static_cast<uint8_t>(*begin));
  } else {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      error = "cannot open " + path;
      return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(input), {});
  }
  if (bytes.empty()) {
    error = "profile response payload is empty";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string format = "json";
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--input" && i + 1 < argc) {
      input_path = argv[++i];
    } else if (argument == "--format" && i + 1 < argc) {
      format = argv[++i];
    } else if (argument == "--help") {
      std::cout << "usage: onnx-remote-profile-dump --input FILE|- "
                   "[--format json|chrome]\n";
      return 0;
    } else {
      std::cerr << "unknown or incomplete argument: " << argument << '\n';
      return 2;
    }
  }
  if (input_path.empty() || (format != "json" && format != "chrome")) {
    std::cerr << "usage: onnx-remote-profile-dump --input FILE|- "
                 "[--format json|chrome]\n";
    return 2;
  }

  std::vector<uint8_t> bytes;
  std::string error;
  if (!read_input(input_path, bytes, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  Response response;
  if (!decode_response_payload(bytes.data(), bytes.size(), response, error)) {
    std::cerr << "invalid response payload: " << error << '\n';
    return 1;
  }
  if (!response.ok) {
    std::cerr << (response.error.empty() ? "remote response failed"
                                         : response.error)
              << '\n';
    return 1;
  }
  std::cout << (format == "chrome" ? profile_chrome_trace_json(response)
                                    : profile_json(response))
            << '\n';
  return 0;
}
