// Dispatch-overhead microbenchmark: N tiny elementwise dispatches in one compute pass.
//   chain MODE N ELEMS [PASSES]      MODE: dep = each dispatch reads the previous one's output (RAW chain),
//                                          ind = every dispatch has its own buffers (no hazards)
// Reports CPU time to record+submit, and total time until the GPU is done.
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
  string mode = argv[1]; uint32_t N = atoi(argv[2]), elems = atoi(argv[3]), passes = argc > 4 ? atoi(argv[4]) : 1, submit_every = argc > 5 ? atoi(argv[5]) : 0;
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat = wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount = 1; id.requiredFeatures = &feat;
  wgpu::Instance inst = wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType = wgpu::BackendType::Vulkan;
  std::vector<const char*> dev_en = {"skip_validation", "disable_robustness"};
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView) { if (s == wgpu::RequestAdapterStatus::Success) ad = a; }), UINT64_MAX);
  wgpu::DawnTogglesDescriptor dvt{}; dvt.enabledToggleCount = dev_en.size(); dvt.enabledToggles = dev_en.data();
  wgpu::DeviceDescriptor dd{}; dd.nextInChain = &dvt;
  dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView m) { fprintf(stderr, "DEVICE ERROR: %.*s\n", (int)m.length, m.data); });
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView) { if (s == wgpu::RequestDeviceStatus::Success) dev = d; }), UINT64_MAX);
  string src = "@group(0) @binding(0) var<storage, read> a : array<vec4<f32>>;\n@group(0) @binding(1) var<storage, read_write> o : array<vec4<f32>>;\n"
               "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) g : vec3<u32>) { if (g.x < " + to_string(elems / 4) + "u) { o[g.x] = a[g.x] + vec4<f32>(1.0); } }\n";
  wgpu::ShaderSourceWGSL w{}; w.code = {src.data(), src.size()}; wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain = &w;
  wgpu::ShaderModule sm = dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module = sm; cpd.compute.entryPoint = "main";
  wgpu::ComputePipeline pl = dev.CreateComputePipeline(&cpd);
  auto mk = [&](size_t n) { wgpu::BufferDescriptor b{}; b.size = n; b.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst; return dev.CreateBuffer(&b); };
  size_t bytes = (size_t)elems * 4;
  std::vector<wgpu::BindGroup> groups;
  if (mode == "dep") {   // ping-pong between two buffers: dispatch i reads what i-1 wrote
    wgpu::Buffer x = mk(bytes), y = mk(bytes);
    for (uint32_t i = 0; i < N; i++) {
      wgpu::BindGroupEntry e[2]; e[0].binding = 0; e[0].buffer = (i % 2 ? y : x); e[0].size = bytes; e[1].binding = 1; e[1].buffer = (i % 2 ? x : y); e[1].size = bytes;
      wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 2; bgd.entries = e; groups.push_back(dev.CreateBindGroup(&bgd));
    }
  } else {               // independent: separate in/out buffers per dispatch
    for (uint32_t i = 0; i < N; i++) {
      wgpu::Buffer x = mk(bytes), y = mk(bytes);
      wgpu::BindGroupEntry e[2]; e[0].binding = 0; e[0].buffer = x; e[0].size = bytes; e[1].binding = 1; e[1].buffer = y; e[1].size = bytes;
      wgpu::BindGroupDescriptor bgd{}; bgd.layout = pl.GetBindGroupLayout(0); bgd.entryCount = 2; bgd.entries = e; groups.push_back(dev.CreateBindGroup(&bgd));
    }
  }
  uint32_t wg = (elems / 4 + 63) / 64;
  auto run = [&](double& cpu_ms) {
    auto t0 = std::chrono::steady_clock::now();
    uint32_t per_sub = submit_every ? submit_every : N;
    for (uint32_t base = 0; base < N; base += per_sub) {
      uint32_t cnt = std::min(per_sub, N - base);
      wgpu::CommandEncoder enc = dev.CreateCommandEncoder();
      uint32_t per_pass = std::max(1u, cnt / std::max(1u, passes));
      for (uint32_t i = 0; i < cnt; i += per_pass) {
        wgpu::ComputePassEncoder pass = enc.BeginComputePass(); pass.SetPipeline(pl);
        for (uint32_t j = i; j < std::min(cnt, i + per_pass); j++) { pass.SetBindGroup(0, groups[base + j]); pass.DispatchWorkgroups(wg, 1, 1); }
        pass.End();
      }
      wgpu::CommandBuffer cb = enc.Finish();
      dev.GetQueue().Submit(1, &cb);
    }
    cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    inst.WaitAny(dev.GetQueue().OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly, [](wgpu::QueueWorkDoneStatus, wgpu::StringView) {}), UINT64_MAX);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  double c; run(c); run(c);
  std::vector<double> tot, cpu;
  for (int i = 0; i < 15; i++) { double cc; tot.push_back(run(cc)); cpu.push_back(cc); }
  std::sort(tot.begin(), tot.end()); std::sort(cpu.begin(), cpu.end());
  printf("%-3s N=%u elems=%u passes=%u submit_every=%u  total=%.2f ms (%.1f us/dispatch)  cpu record+submit=%.2f ms (%.1f us/dispatch)\n", mode.c_str(), N, elems, passes, submit_every, tot[0], tot[0] * 1000 / N, cpu[0], cpu[0] * 1000 / N);
  return 0;
}
