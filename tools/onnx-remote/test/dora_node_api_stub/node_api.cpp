#include "node_api.h"

#include "remote_transport.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct StubEvent {
  DoraEventType type = DoraEventType_Input;
  std::string id;
  std::vector<uint8_t> data;
};

struct StubContext {
  bool runtime = false;
  bool delivered = false;
  bool failed = false;
};

StubContext* context(void* value) {
  return static_cast<StubContext*>(value);
}

}  // namespace

extern "C" {

void* init_dora_context_from_env(void) {
  auto* result = new StubContext;
  result->runtime = std::getenv("ONNXSIM_DORA_STUB_RUNTIME") != nullptr;
  return result;
}

void* dora_next_event(void* value) {
  auto* ctx = context(value);
  if (!ctx->runtime || ctx->delivered) return nullptr;
  ctx->delivered = true;
  onnx_remote::Request request;
  request.request_id = 41;
  request.op = "relu";
  request.profiling = onnx_remote::ProfilingLevel::Detailed;
  request.inputs.push_back(onnx_remote::Tensor{{4}, {-1.0f, 0.0f, 2.0f, 4.0f}});
  std::string error;
  auto* event = new StubEvent;
  event->id = "run";
  if (!onnx_remote::encode_request_payload(request, event->data, error)) {
    ctx->failed = true;
    delete event;
    return nullptr;
  }
  return event;
}

void free_dora_event(void* value) { delete static_cast<StubEvent*>(value); }

void free_dora_context(void* value) {
  auto* ctx = context(value);
  const bool failed = ctx->failed;
  delete ctx;
  if (failed) std::_Exit(1);
}

DoraEventType read_dora_event_type(void* value) {
  return static_cast<StubEvent*>(value)->type;
}

int read_dora_input_id(void* value, char** id, size_t* length) {
  auto* event = static_cast<StubEvent*>(value);
  *id = event->id.data();
  *length = event->id.size();
  return 0;
}

int read_dora_input_data(void* value, char** data, size_t* length) {
  auto* event = static_cast<StubEvent*>(value);
  *data = reinterpret_cast<char*>(event->data.data());
  *length = event->data.size();
  return 0;
}

int dora_send_output(void* value, const char* id, size_t id_length,
                     const char* data, size_t data_length) {
  auto* ctx = context(value);
  if (!ctx->runtime) return 0;
  if (id_length == 13 && std::memcmp(id, "profile_event", 13) == 0) {
    const std::string profile(data, data_length);
    if (profile.find("\"request_id\":41") == std::string::npos ||
        profile.find("\"event\"") == std::string::npos)
      ctx->failed = true;
    return ctx->failed ? -1 : 0;
  }
  if (id_length != 6 || std::memcmp(id, "result", 6) != 0)
    return 0;
  onnx_remote::Response response;
  std::string error;
  if (!onnx_remote::decode_response_payload(
          reinterpret_cast<const uint8_t*>(data), data_length, response,
          error) ||
      !response.ok || response.request_id != 41 || response.outputs.size() != 1 ||
      response.outputs[0].data.size() != 4 || response.outputs[0].data[0] != 0.0f ||
      response.outputs[0].data[1] != 0.0f || response.outputs[0].data[2] != 2.0f ||
      response.outputs[0].data[3] != 4.0f) {
    ctx->failed = true;
    return -1;
  }
  return 0;
}

void dora_log(void*, const char*, size_t, const char*, size_t) {}

}
