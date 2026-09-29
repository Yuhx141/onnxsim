// Measures achievable FMA throughput of the WebGPU/Vulkan device: peak <f32|f16> WORKGROUPS ITERS
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  dawnProcSetProcs(&dawn::native::GetProcs());
  std::string ty = argc > 1 ? argv[1] : "f32";
  uint32_t wgs = argc > 2 ? atoi(argv[2]) : 4096, iters = argc > 3 ? atoi(argv[3]) : 2048;
  int VW = argc > 4 ? atoi(argv[4]) : 2, CH = argc > 5 ? atoi(argv[5]) : 8, WS = argc > 6 ? atoi(argv[6]) : 256;
  wgpu::InstanceDescriptor id{};
  wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny;
  id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  wgpu::Instance inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  if (!ad) { fprintf(stderr, "no adapter\n"); return 2; }
  bool has16 = ad.HasFeature(wgpu::FeatureName::ShaderF16);
  bool hasTS = ad.HasFeature(wgpu::FeatureName::TimestampQuery);
  printf("shader-f16=%d timestamp-query=%d subgroups=%d\n", has16, hasTS, ad.HasFeature(wgpu::FeatureName::Subgroups));
  if (ty == "f16" && !has16) { printf("f16 unsupported by this adapter\n"); return 3; }
  std::vector<wgpu::FeatureName> feats;
  if (has16) feats.push_back(wgpu::FeatureName::ShaderF16);
  wgpu::DeviceDescriptor dd{}; dd.requiredFeatureCount = feats.size(); dd.requiredFeatures = feats.data();
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
  std::string T = ty == "f16" ? "f16" : "f32";
  std::string vec = VW == 1 ? T : ("vec" + std::to_string(VW) + "<" + T + ">");
  auto lit = [&](const char* v) { return VW == 1 ? std::string(T) + "(" + v + ")" : vec + "(" + v + ")"; };
  std::string src = std::string(ty == "f16" ? "enable f16;\n" : "") +
    "@group(0) @binding(0) var<storage, read_write> o : array<" + T + ">;\n"
    "struct U { iters: u32 }\n@group(0) @binding(1) var<uniform> u : U;\n"
    "@compute @workgroup_size(" + std::to_string(WS) + ")\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "  let s = " + T + "(f32(gid.x) * 0.001);\n  let b = " + lit("1.0009765625") + "; let c = " + lit("0.0009765625") + ";\n";
  for (int i = 0; i < CH; i++) src += "  var a" + std::to_string(i) + " = " + lit(("s + " + std::to_string(i * 0.25)).c_str()) + ";\n";
  src += "  for (var i = 0u; i < u.iters; i++) {\n";
  for (int i = 0; i < CH; i++) src += "    a" + std::to_string(i) + " = fma(a" + std::to_string(i) + ", b, c);\n";
  src += "  }\n  var r = a0;\n";
  for (int i = 1; i < CH; i++) src += "  r = r + a" + std::to_string(i) + ";\n";
  src += VW == 1 ? "  o[gid.x] = r;\n}\n" : (VW == 2 ? "  o[gid.x] = r.x + r.y;\n}\n" : "  o[gid.x] = r.x + r.y + r.z + r.w;\n}\n");
  wgpu::ShaderSourceWGSL w{}; w.code = {src.data(), src.size()};
  wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  wgpu::ComputePipeline pl = dev.CreateComputePipeline(&cpd);
  auto mk = [&](size_t n, wgpu::BufferUsage u) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = u; return dev.CreateBuffer(&b); };
  size_t nthreads = (size_t)wgs * WS;
  wgpu::Buffer bo = mk(nthreads * (ty == "f16" ? 2 : 4), wgpu::BufferUsage::Storage);
  wgpu::Buffer bu = mk(16, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  uint32_t uni[4] = {iters, 0, 0, 0};
  dev.GetQueue().WriteBuffer(bu, 0, uni, 16);
  wgpu::BindGroupEntry e[2]; e[0].binding = 0; e[0].buffer = bo; e[0].size = bo.GetSize(); e[1].binding = 1; e[1].buffer = bu; e[1].size = 16;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 2; bgd.entries = e;
  wgpu::BindGroup bg = dev.CreateBindGroup(&bgd);
  auto run = [&]() {
    wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
    { wgpu::ComputePassEncoder p = enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0, bg); p.DispatchWorkgroups(wgs, 1, 1); p.End(); }
    wgpu::CommandBuffer cb = enc.Finish();
    auto t0 = std::chrono::steady_clock::now();
    dev.GetQueue().Submit(1, &cb);
    inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  run();
  std::vector<double> t;
  for (int i = 0; i < 12; i++) t.push_back(run());
  std::sort(t.begin(), t.end());
  double flops = (double)nthreads * iters * CH * VW * 2;  // CH fma/iter, VW lanes each, 2 flop per fma
  double best = t[0], med = t[t.size() / 2];
  printf("%s vw=%d chains=%d wg=%d workgroups=%u iters=%u  best=%.2f ms (%.0f GFLOPS)  median=%.2f ms (%.0f GFLOPS)\n", ty.c_str(), VW, CH, WS, wgs, iters, best, flops / best / 1e6, med, flops / med / 1e6);
  return 0;
}
