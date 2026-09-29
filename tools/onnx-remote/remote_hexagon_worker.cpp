// Hexagon cDSP runner for compiled `tghx-v65` artifacts: tinygrad-generated standalone DSP programs made by
// scripts/android/tinygrad_hexagon_bridge/openpilot_v65/compile_v65.sh behind onnx-remote-compiler.
//
// Runs on the Android side of a Snapdragon (FastRPC to the cDSP, unsigned PD). Operations:
//   capabilities                     the runner manifest
//   load_compiled(id, artifact)      unpack, cache tg_graph_<id>.so, open it on the cDSP and upload the weights once
//   run_compiled(id, tensors)        ONNX inputs in graph order -> ONNX outputs in graph order (in their program.txt dtypes)
//                                    a recurrent-state input (program.txt "state") sent empty takes the previous run's output,
//                                    and that output comes back empty: after the first call the state stays on the phone
// An artifact is TGHXV65\0, u32 file count, then per file u32 name length, name, u64 size, bytes: tg_graph.so (the skel),
// blob.bin (weights) and program.txt (the I/O contract; see tinygrad's examples/openpilot/dsp_graph_v65.py).
#include "remote_transport.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "remote.h"
#include "tg_graph.h"

using namespace onnx_remote;
namespace fs = std::filesystem;

namespace {

struct InputSpec { int onnx_index = 0; uint32_t dtype = 0; uint64_t bytes = 0; int slot = -1; };
struct OutputSpec { uint8_t dtype = 1; uint64_t elements = 0; std::vector<int64_t> shape; };
struct Program {
  int ncalls = 0, threads = 1;
  uint64_t output_bytes = 0, output_align = 1;  // each output starts at a multiple of output_align bytes
  std::vector<InputSpec> inputs;  // ONNX graph order
  std::vector<OutputSpec> outputs;
  std::vector<std::string> names;  // kernel name per call (program.txt "name"), for per-call profile events
  // recurrent state (program.txt "state <output> <input>": the output is the input's next value, e.g. next_state_img_q ->
  // state_img_q). The last run's state outputs stay here, so a client can send such an input empty to mean "the resident
  // value" and then gets that output back empty: the state never crosses the transport after the first call
  std::vector<std::pair<int, int>> states;  // (output index, input ONNX index)
  std::map<int, std::vector<uint8_t>> resident;  // input ONNX index -> bytes
  remote_handle64 handle = 0;
};

fs::path g_cache_dir = "/data/local/tmp/onnx-remote-hexagon";
int g_threads_override = 0;
std::map<std::string, std::unique_ptr<Program>> g_programs;

uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool valid_id(const std::string& id) {
  if (id.empty() || id.size() > kMaxArtifactIdBytes) return false;
  return std::all_of(id.begin(), id.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '.' || c == '_' || c == '-';
  });
}

template <typename T> bool take(const std::vector<uint8_t>& b, size_t& at, T& v) {
  if (at + sizeof(T) > b.size()) return false;
  std::memcpy(&v, b.data() + at, sizeof(T));  // little-endian on every Android target
  at += sizeof(T);
  return true;
}

bool unpack(const std::vector<uint8_t>& artifact, std::map<std::string, std::pair<size_t, size_t>>& files, std::string& error) {
  static const char kMagic[8] = {'T', 'G', 'H', 'X', 'V', '6', '5', '\0'};
  size_t at = 8;
  uint32_t count = 0;
  if (artifact.size() < 12 || std::memcmp(artifact.data(), kMagic, 8) != 0 || !take(artifact, at, count) || count > 16) {
    error = "not a tghx-v65 artifact";
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_len = 0;
    uint64_t size = 0;
    if (!take(artifact, at, name_len) || name_len > 64 || at + name_len > artifact.size()) { error = "truncated artifact"; return false; }
    std::string name(reinterpret_cast<const char*>(artifact.data() + at), name_len);
    at += name_len;
    if (!take(artifact, at, size) || size > artifact.size() - at) { error = "truncated artifact"; return false; }
    files[name] = {at, static_cast<size_t>(size)};
    at += static_cast<size_t>(size);
  }
  for (const char* need : {"tg_graph.so", "blob.bin", "program.txt"})
    if (!files.count(need)) { error = std::string("artifact lacks ") + need; return false; }
  return true;
}

bool parse_program(const std::string& text, Program& p, std::string& error) {
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream f(line);
    std::string key;
    f >> key;
    if (key == "ncalls") f >> p.ncalls;
    else if (key == "threads") f >> p.threads;
    else if (key == "output_bytes") f >> p.output_bytes;
    else if (key == "output_align") f >> p.output_align;
    else if (key == "input") {
      InputSpec s;
      f >> s.onnx_index >> s.dtype >> s.bytes >> s.slot;
      p.inputs.push_back(s);
    } else if (key == "output") {
      OutputSpec o;
      uint32_t dtype = 0;
      f >> dtype >> o.elements;
      for (int64_t d; f >> d;) o.shape.push_back(d);
      if (dtype > 255 || dtype_bytes(static_cast<uint8_t>(dtype)) == 0) { error = "unsupported program output dtype"; return false; }
      o.dtype = static_cast<uint8_t>(dtype);
      p.outputs.push_back(o);
    } else if (key == "state") {
      int o = -1, i = -1;
      f >> o >> i;
      p.states.emplace_back(o, i);
    } else if (key == "name") {
      size_t i = 0;
      std::string name;
      f >> i >> name;
      if (i < 100000) { if (p.names.size() <= i) p.names.resize(i + 1); p.names[i] = name; }
    } else if (!key.empty()) { error = "unknown program record: " + key; return false; }
  }
  uint64_t total = 0;
  if (p.output_align == 0) { error = "inconsistent program.txt"; return false; }
  for (const auto& o : p.outputs) total = (total + o.elements * dtype_bytes(o.dtype) + p.output_align - 1) / p.output_align * p.output_align;
  if (p.ncalls <= 0 || p.outputs.empty() || total != p.output_bytes) { error = "inconsistent program.txt"; return false; }
  for (const auto& [o, idx] : p.states) {
    const int i = idx;  // (a structured binding can't be captured in C++17)
    const auto in = std::find_if(p.inputs.begin(), p.inputs.end(), [&](const InputSpec& s) { return s.onnx_index == i; });
    if (o < 0 || o >= static_cast<int>(p.outputs.size()) || in == p.inputs.end() || in->dtype != p.outputs[o].dtype ||
        in->bytes != p.outputs[o].elements * dtype_bytes(p.outputs[o].dtype)) { error = "inconsistent program.txt state"; return false; }
  }
  return true;
}

// Load (or re-attach) one artifact: the skel goes to the cache dir, which is on ADSP_LIBRARY_PATH, under a per-artifact name so
// several programs can be resident in the PD at once.
bool load(const std::string& id, const std::vector<uint8_t>& artifact, Response& response) {
  if (!valid_id(id)) { response.error = "invalid artifact id"; return false; }
  if (g_programs.count(id)) return true;
  std::map<std::string, std::pair<size_t, size_t>> files;
  if (!unpack(artifact, files, response.error)) return false;
  auto program = std::make_unique<Program>();
  const auto& [ptxt, ntxt] = files["program.txt"];
  if (!parse_program(std::string(reinterpret_cast<const char*>(artifact.data() + ptxt), ntxt), *program, response.error)) return false;
  std::error_code ec;
  fs::create_directories(g_cache_dir, ec);
  const std::string so_name = "tg_graph_" + id + ".so";
  {
    const auto& [off, n] = files["tg_graph.so"];
    std::ofstream so(g_cache_dir / so_name, std::ios::binary | std::ios::trunc);
    so.write(reinterpret_cast<const char*>(artifact.data() + off), static_cast<std::streamsize>(n));
    if (!so) { response.error = "cannot write the skel to the cache dir"; return false; }
  }
  const std::string uri = "file:///" + so_name + "?tg_graph_skel_handle_invoke&_modver=1.0&_dom=cdsp";
  int rc = tg_graph_open(uri.c_str(), &program->handle);
  if (rc) { response.error = "FastRPC open of " + so_name + " failed: " + std::to_string(rc); return false; }
  const auto& [boff, bn] = files["blob.bin"];
  constexpr size_t kChunk = 8u << 20;  // well under FastRPC's per-buffer limit
  for (size_t at = 0; at < bn || at == 0; at += kChunk) {
    const size_t n = std::min(kChunk, bn - at);
    rc = tg_graph_load(program->handle, static_cast<int>(at), static_cast<int>(bn), artifact.data() + boff + at, static_cast<int>(n));
    if (rc) { tg_graph_close(program->handle); response.error = "weight upload failed: " + std::to_string(rc); return false; }
    if (bn == 0) break;
  }
  g_programs[id] = std::move(program);
  return true;
}

void profile(Response& r, const Request& q, const char* name, uint64_t start, uint64_t dur, const std::string& detail) {
  if (q.profiling != ProfilingLevel::Off) r.profile.push_back(ProfileEvent{name, "hexagon", start, dur, detail});
}

Response execute(const Request& request) {
  Response response;
  response.request_id = request.request_id;
  const uint64_t t0 = now_us();
  if (request.op == "capabilities") {
    response.ok = true;
    response.artifact_id = "hexagon-v65-runner";
    response.manifest =
        "{\"schema_version\":1,\"protocol\":\"onnx-remote-v5\",\"runner_id\":\"hexagon-v65-runner\",\"ready\":true,"
        "\"graph_execution\":false,\"supported_ops\":[\"load_compiled\",\"run_compiled\"],"
        "\"supported_dtypes\":[\"FLOAT\",\"UINT8\"],\"profiling\":true,"
        "\"artifact\":{\"format\":\"tghx-v65\",\"abi\":\"tg_graph-idl-1\"}}";
    return response;
  }
  if (request.op != "load_compiled" && request.op != "run_compiled") {
    response.error = "hexagon runner accepts capabilities/load_compiled/run_compiled";
    return response;
  }
  if (!request.artifact.empty() || request.op == "load_compiled") {
    if (!load(request.artifact_id, request.artifact, response)) return response;
    profile(response, request, "hexagon_load", 0, now_us() - t0, request.artifact_id);
  }
  if (request.op == "load_compiled") { response.ok = true; response.artifact_id = request.artifact_id; return response; }
  auto it = g_programs.find(request.artifact_id);
  if (it == g_programs.end()) { response.error = "compiled artifact is not attached: " + request.artifact_id; return response; }
  Program& p = *it->second;
  if (request.inputs.size() != p.inputs.size()) { response.error = "input count differs from the compiled model"; return response; }
  // the program's input buffer: its used inputs in program slot order, each 128-byte aligned
  std::vector<const InputSpec*> by_slot;
  for (const auto& s : p.inputs) if (s.slot >= 0) by_slot.push_back(&s);
  std::sort(by_slot.begin(), by_slot.end(), [](const InputSpec* a, const InputSpec* b) { return a->slot < b->slot; });
  std::vector<uint8_t> packed;
  std::vector<bool> from_resident(request.inputs.size(), false);
  for (const auto& [o, i] : p.states) {
    const Tensor& t = request.inputs[static_cast<size_t>(i)];
    if (!t.data.empty() || !t.raw_data.empty()) continue;
    if (!p.resident.count(i)) {
      response.error = "state input " + std::to_string(i) + " sent empty but has no resident value: send it in full first";
      return response;
    }
    from_resident[static_cast<size_t>(i)] = true;
  }
  for (const InputSpec* s : by_slot) {
    const Tensor& t = request.inputs[static_cast<size_t>(s->onnx_index)];
    const bool res = from_resident[static_cast<size_t>(s->onnx_index)];
    const uint8_t* data = res ? p.resident[s->onnx_index].data()
                              : t.dtype == 1 ? reinterpret_cast<const uint8_t*>(t.data.data()) : t.raw_data.data();
    const size_t n = res ? p.resident[s->onnx_index].size() : t.dtype == 1 ? t.data.size() * 4 : t.raw_data.size();
    if ((!res && t.dtype != s->dtype) || n != s->bytes) {
      response.error = "input " + std::to_string(s->onnx_index) + " dtype/size differs from the compiled model";
      return response;
    }
    packed.insert(packed.end(), data, data + n);
    packed.resize((packed.size() + 127) / 128 * 128);
  }
  std::vector<uint8_t> out(p.output_bytes);
  const bool detailed = request.profiling == ProfilingLevel::Detailed;
  std::vector<uint64_t> times(detailed ? 1 + static_cast<size_t>(p.ncalls) : 1);
  const int threads = g_threads_override > 0 ? g_threads_override : p.threads;
  const uint64_t run0 = now_us();
  int rc = tg_graph_run(p.handle, 0, p.ncalls, threads, packed.data(), static_cast<int>(packed.size()), out.data(),
                        static_cast<int>(out.size()), reinterpret_cast<uint64*>(times.data()), static_cast<int>(times.size()));
  if (rc) { response.error = "tg_graph_run failed: " + std::to_string(rc); return response; }
  profile(response, request, "hexagon_run", run0 - t0, now_us() - run0,
          "dsp_us=" + std::to_string(times[0]) + " threads=" + std::to_string(threads));
  if (detailed) {
    // the transport carries at most kMaxProfileEvents events: send the slowest calls, not the first ones
    std::vector<int> order(p.ncalls);
    for (int i = 0; i < p.ncalls; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return times[1 + a] > times[1 + b]; });
    const size_t room = kMaxProfileEvents - response.profile.size() - 1;
    for (size_t k = 0; k < order.size() && k < room; ++k) {
      const int i = order[k];
      if (times[1 + i] == 0) break;
      response.profile.push_back(ProfileEvent{"call_" + std::to_string(i), "hexagon_call", 0, times[1 + i],
                                              i < static_cast<int>(p.names.size()) ? p.names[i] : ""});
    }
  }
  size_t at = 0;
  std::vector<size_t> offsets;  // the program's output is the ONNX outputs' bytes back to back, each in its own dtype
  for (const auto& o : p.outputs) {
    offsets.push_back(at);
    at = (at + o.elements * dtype_bytes(o.dtype) + p.output_align - 1) / p.output_align * p.output_align;
  }
  std::vector<bool> omit(p.outputs.size(), false);  // state outputs of inputs that came from the resident values
  for (const auto& [o, i] : p.states) {
    const size_t n = p.outputs[o].elements * dtype_bytes(p.outputs[o].dtype);
    p.resident[i].assign(out.data() + offsets[o], out.data() + offsets[o] + n);
    omit[o] = from_resident[static_cast<size_t>(i)];
  }
  for (size_t k = 0; k < p.outputs.size(); ++k) {
    const auto& o = p.outputs[k];
    Tensor t;
    t.shape = o.shape;
    t.dtype = o.dtype;
    const size_t n = o.elements * dtype_bytes(o.dtype);
    if (omit[k]) {
      t.shape = {0};
    } else if (o.dtype == 1) {
      t.data.resize(o.elements);
      std::memcpy(t.data.data(), out.data() + offsets[k], n);
    } else {
      t.raw_data.assign(out.data() + offsets[k], out.data() + offsets[k] + n);
    }
    response.outputs.push_back(std::move(t));
  }
  response.ok = true;
  return response;
}

}  // namespace

int main(int argc, char** argv) {
  uint16_t port = 39520;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (a == "--cache-dir" && i + 1 < argc) g_cache_dir = argv[++i];
    else if (a == "--threads" && i + 1 < argc) g_threads_override = std::atoi(argv[++i]);
    else {
      std::cout << "usage: onnx-remote-hexagon-worker [--port PORT] [--cache-dir DIR] [--threads N]\n";
      return a == "--help" ? 0 : 2;
    }
  }
  std::error_code ec;
  fs::create_directories(g_cache_dir, ec);
  g_cache_dir = fs::absolute(g_cache_dir);
  // the cDSP loader finds tg_graph_<id>.so through ADSP_LIBRARY_PATH; set before the first FastRPC call
  std::string path = g_cache_dir.string();
  if (const char* old = std::getenv("ADSP_LIBRARY_PATH")) path += std::string(";") + old;
  setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
  remote_rpc_control_unsigned_module um = {CDSP_DOMAIN_ID, 1};
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &um, sizeof(um));
  std::signal(SIGPIPE, SIG_IGN);
  const int listener = listen_tcp(port);
  if (listener < 0) { std::cerr << "listen failed: " << std::strerror(errno) << '\n'; return 1; }
  std::cerr << "onnx-remote-hexagon-worker listening on " << port << ", cache " << g_cache_dir << '\n';
  for (;;) {
    const int fd = accept_tcp(listener);
    if (fd < 0) continue;
    Request request;
    Response response;
    std::string error;
    if (!receive_request(fd, request, error)) response.error = error;
    else response = execute(request);
    if (!response.ok && response.error.empty()) response.error = "hexagon runner failed";
    send_response(fd, response, error);
    close_socket(fd);
  }
}
