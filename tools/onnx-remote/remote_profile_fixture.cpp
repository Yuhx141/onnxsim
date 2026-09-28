#include "remote_profile.h"

#include <iostream>
#include <string>
#include <vector>

using namespace onnx_remote;

int main() {
  Response response;
  response.request_id = 42;
  response.ok = true;
  response.profile.push_back(
      ProfileEvent{"fixture_execute", "fixture", 12, 34, "profile-cli"});
  std::vector<uint8_t> payload;
  std::string error;
  if (!encode_response_payload(response, payload, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
  return std::cout ? 0 : 1;
}
