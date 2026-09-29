// Load-bandwidth microbenchmark: bw KIND WORKING_SET_KB ITERS   (KIND = buf | tex | lds)
// Every thread issues 8 independent vec4<f32> loads per iteration from a small working set, so this
// measures cache/local-memory load throughput, not DRAM.
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using std::string; using std::to_string;
int main(int argc, char** argv) {
  dawnProcSetProcs(&dawn::native::GetProcs());
  string kind = argv[1]; uint32_t kb = atoi(argv[2]), iters = atoi(argv[3]);
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  wgpu::Instance inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DeviceDescriptor dd{};
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
  uint32_t nvec = kb * 1024 / 16;               // vec4 elements in the working set (power of two)
  uint32_t wgs = 8192, WS = 64;
  string hdr = "struct U { iters: u32 }\n@group(0) @binding(1) var<uniform> u : U;\n@group(0) @binding(2) var<storage, read_write> dst : array<f32>;\n";
  string body;
  if (kind == "buf") {
    hdr += "@group(0) @binding(0) var<storage, read> src : array<vec4<f32>>;\n";
    body = "  let m = " + to_string(nvec - 1) + "u;\n  var acc = vec4<f32>(0.0);\n  for (var i = 0u; i < u.iters; i++) {\n    let b = gid.x + i * 8u;\n";
    for (int q = 0; q < 8; q++) body += "    acc += src[(b + " + to_string(q) + "u * 3u) & m];\n";
    body += "  }\n  dst[gid.x] = acc.x + acc.y + acc.z + acc.w;\n";
  } else if (kind == "tex") {
    hdr += "@group(0) @binding(0) var src : texture_2d<f32>;\n";
    uint32_t h = std::max(1u, nvec / 256);
    body = "  var acc = vec4<f32>(0.0);\n  for (var i = 0u; i < u.iters; i++) {\n    let b = gid.x + i * 8u;\n";
    for (int q = 0; q < 8; q++) body += "    { let p = (b + " + to_string(q) + "u * 3u) & " + to_string(nvec - 1) + "u; acc += textureLoad(src, vec2<u32>(p & 255u, (p >> 8u) % " + to_string(h) + "u), 0); }\n";
    body += "  }\n  dst[gid.x] = acc.x + acc.y + acc.z + acc.w;\n";
  } else {  // lds
    hdr += "@group(0) @binding(0) var<storage, read> src : array<vec4<f32>>;\nvar<workgroup> lds : array<vec4<f32>, " + to_string(nvec) + ">;\n";
    body = "  for (var i = lid.x; i < " + to_string(nvec) + "u; i += " + to_string(WS) + "u) { lds[i] = src[i]; }\n  workgroupBarrier();\n"
           "  let m = " + to_string(nvec - 1) + "u;\n  var acc = vec4<f32>(0.0);\n  for (var i = 0u; i < u.iters; i++) {\n    let b = lid.x + i * 8u;\n";
    for (int q = 0; q < 8; q++) body += "    acc += lds[(b + " + to_string(q) + "u * 3u) & m];\n";
    body += "  }\n  dst[gid.x] = acc.x + acc.y + acc.z + acc.w;\n";
  }
  string src = hdr + "@compute @workgroup_size(" + to_string(WS) + ")\nfn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {\n" + body + "}\n";
  wgpu::ShaderSourceWGSL w{}; w.code = {src.data(), src.size()}; wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  wgpu::ComputePipeline pl = dev.CreateComputePipeline(&cpd);
  auto mk = [&](size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = u; return dev.CreateBuffer(&b); };
  wgpu::Buffer bs = mk((size_t)nvec * 16, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst);
  wgpu::Buffer bd = mk((size_t)wgs * WS * 4, wgpu::BufferUsage::Storage);
  wgpu::Buffer bu = mk(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  std::vector<float> host((size_t)nvec * 4); for (size_t i = 0; i < host.size(); i++) host[i] = (i % 13) * 0.01f;
  uint32_t uni[4] = {iters, 0, 0, 0}; dev.GetQueue().WriteBuffer(bu, 0, uni, 16); dev.GetQueue().WriteBuffer(bs, 0, host.data(), host.size() * 4);
  wgpu::Texture tex;
  wgpu::BindGroupEntry e[3]; e[1].binding = 1; e[1].buffer = bu; e[1].size = 16; e[2].binding = 2; e[2].buffer = bd; e[2].size = bd.GetSize(); e[0].binding = 0;
  if (kind == "tex") {
    uint32_t h = std::max(1u, nvec / 256);
    wgpu::TextureDescriptor td{}; td.size = {256, h, 1}; td.format = wgpu::TextureFormat::RGBA32Float; td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    tex = dev.CreateTexture(&td);
    wgpu::TexelCopyTextureInfo dstinfo{}; dstinfo.texture = tex;
    wgpu::TexelCopyBufferLayout lay{}; lay.bytesPerRow = 256 * 16; lay.rowsPerImage = h;
    wgpu::Extent3D ext = {256, h, 1};
    dev.GetQueue().WriteTexture(&dstinfo, host.data(), host.size() * 4, &lay, &ext);
    e[0].textureView = tex.CreateView();
  } else { e[0].buffer = bs; e[0].size = bs.GetSize(); }
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 3; bgd.entries = e; wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);
  auto run = [&]() {
    wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
    { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); p.DispatchWorkgroups(wgs, 1, 1); p.End(); }
    wgpu::CommandBuffer cb = enc.Finish();
    auto t0 = std::chrono::steady_clock::now();
    dev.GetQueue().Submit(1, &cb);
    inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  run(); std::vector<double> t; for (int i = 0; i < 9; i++) t.push_back(run()); std::sort(t.begin(), t.end());
  double bytes = (double)wgs * WS * iters * 8 * 16;
  printf("%-4s working set %5u KB  best=%.2f ms  %.0f GB/s (%.0f G vec4 loads/s)\n", kind.c_str(), kb, t[0], bytes / t[0] / 1e6, bytes / 16 / t[0] / 1e6);
  return 0;
}
