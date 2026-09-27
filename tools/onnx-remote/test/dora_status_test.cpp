#include "dora_status.h"

#include <cassert>
#include <string>

int main() {
  using onnx_remote::dora::readiness_status;
  using onnx_remote::dora::unavailable_capabilities;

  const std::string ready = readiness_status(true, "100.64.0.7", 39501);
  assert(ready.find("\"schema_version\":1") != std::string::npos);
  assert(ready.find("\"status\":\"ready\"") != std::string::npos);
  assert(ready.find("\"host\":\"100.64.0.7\"") != std::string::npos);
  assert(ready.find("\"port\":39501") != std::string::npos);
  assert(ready.find("\"error\"") == std::string::npos);

  const std::string error =
      readiness_status(false, "runner\"host", 39502, "broken\nworker");
  assert(error.find("runner\\\"host") != std::string::npos);
  assert(error.find("broken\\nworker") != std::string::npos);
  assert(error.find("\"status\":\"unavailable\"") != std::string::npos);

  const std::string capabilities = unavailable_capabilities("bad\tworker");
  assert(capabilities.find("bad\\tworker") != std::string::npos);
  assert(capabilities.find("\"ready\":false") != std::string::npos);
  return 0;
}
