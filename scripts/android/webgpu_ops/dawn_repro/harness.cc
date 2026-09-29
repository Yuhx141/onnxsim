// Runs a WGSL transpose-style kernel through Dawn/Vulkan and checks o[c*R+r]==a[r*C+c].
// usage: harness kernel.wgsl R C dispatch_x dispatch_y
#include <webgpu/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/native/DawnNative.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
static wgpu::Buffer mk(wgpu::Device& d, size_t n, wgpu::BufferUsage u){wgpu::BufferDescriptor b{};b.size=n;b.usage=u;return d.CreateBuffer(&b);}
int main(int argc,char**argv){
  dawnProcSetProcs(&dawn::native::GetProcs());
  std::ifstream f(argv[1]); std::stringstream ss; ss<<f.rdbuf(); std::string src=ss.str();
  uint32_t R=atoi(argv[2]),C=atoi(argv[3]),dx=atoi(argv[4]),dy=atoi(argv[5]);
  wgpu::InstanceDescriptor id{}; wgpu::InstanceFeatureName feat=wgpu::InstanceFeatureName::TimedWaitAny; id.requiredFeatureCount=1; id.requiredFeatures=&feat;
  wgpu::Instance inst=wgpu::CreateInstance(&id);
  wgpu::RequestAdapterOptions ro{}; ro.backendType=wgpu::BackendType::Vulkan;
  wgpu::Adapter ad;
  inst.WaitAny(inst.RequestAdapter(&ro,wgpu::CallbackMode::WaitAnyOnly,[&](wgpu::RequestAdapterStatus s,wgpu::Adapter a,wgpu::StringView m){ if(s==wgpu::RequestAdapterStatus::Success) ad=a; else fprintf(stderr,"adapter: %.*s\n",(int)m.length,m.data);}),UINT64_MAX);
  if(!ad){fprintf(stderr,"no adapter\n");return 2;}
  wgpu::AdapterInfo ai{}; ad.GetInfo(&ai); printf("vendor=[%.*s] arch=[%.*s] vid=0x%x adapter: %.*s | %.*s\n",(int)ai.vendor.length,ai.vendor.data,(int)ai.architecture.length,ai.architecture.data,ai.vendorID,(int)ai.device.length,ai.device.data,(int)ai.description.length,ai.description.data);
  wgpu::DeviceDescriptor dd{}; dd.SetUncapturedErrorCallback([](const wgpu::Device&,wgpu::ErrorType,wgpu::StringView m){fprintf(stderr,"DEVICE ERROR: %.*s\n",(int)m.length,m.data);});
  wgpu::Device dev;
  inst.WaitAny(ad.RequestDevice(&dd,wgpu::CallbackMode::WaitAnyOnly,[&](wgpu::RequestDeviceStatus s,wgpu::Device d,wgpu::StringView m){ if(s==wgpu::RequestDeviceStatus::Success) dev=d; else fprintf(stderr,"device: %.*s\n",(int)m.length,m.data);}),UINT64_MAX);
  if(!dev){return 2;}
  wgpu::ShaderSourceWGSL w{}; w.code={src.data(),src.size()}; wgpu::ShaderModuleDescriptor smd{}; smd.nextInChain=&w;
  wgpu::ShaderModule sm=dev.CreateShaderModule(&smd);
  wgpu::ComputePipelineDescriptor cpd{}; cpd.compute.module=sm; cpd.compute.entryPoint="main";
  wgpu::ComputePipeline pl=dev.CreateComputePipeline(&cpd);
  size_t n=(size_t)R*C, bytes=n*4;
  std::vector<float> a(n); for(size_t i=0;i<n;i++) a[i]=(float)i;
  auto St=wgpu::BufferUsage::Storage|wgpu::BufferUsage::CopyDst|wgpu::BufferUsage::CopySrc;
  wgpu::Buffer ba=mk(dev,bytes,St), bo=mk(dev,bytes,St), bu=mk(dev,16,wgpu::BufferUsage::Uniform|wgpu::BufferUsage::CopyDst), rb=mk(dev,bytes,wgpu::BufferUsage::MapRead|wgpu::BufferUsage::CopyDst);
  uint32_t uni[4]={R,C,0,0}; std::vector<float> neg(n,-1.f);
  dev.GetQueue().WriteBuffer(ba,0,a.data(),bytes); dev.GetQueue().WriteBuffer(bo,0,neg.data(),bytes); dev.GetQueue().WriteBuffer(bu,0,uni,16);
  wgpu::BindGroupEntry e[3]; e[0].binding=0;e[0].buffer=ba;e[0].size=bytes; e[1].binding=1;e[1].buffer=bo;e[1].size=bytes; e[2].binding=2;e[2].buffer=bu;e[2].size=16;
  wgpu::BindGroupDescriptor bgd{}; bgd.layout=pl.GetBindGroupLayout(0); bgd.entryCount=3; bgd.entries=e; wgpu::BindGroup bg=dev.CreateBindGroup(&bgd);
  wgpu::CommandEncoder enc=dev.CreateCommandEncoder(); { wgpu::ComputePassEncoder p=enc.BeginComputePass(); p.SetPipeline(pl); p.SetBindGroup(0,bg); p.DispatchWorkgroups(dx,dy,1); p.End(); }
  enc.CopyBufferToBuffer(bo,0,rb,0,bytes); wgpu::CommandBuffer cb=enc.Finish(); dev.GetQueue().Submit(1,&cb);
  bool ok=false; inst.WaitAny(rb.MapAsync(wgpu::MapMode::Read,0,bytes,wgpu::CallbackMode::WaitAnyOnly,[&](wgpu::MapAsyncStatus s,wgpu::StringView){ok=s==wgpu::MapAsyncStatus::Success;}),UINT64_MAX);
  if(!ok){fprintf(stderr,"map failed\n");return 2;}
  const float* o=(const float*)rb.GetConstMappedRange(0,bytes); size_t bad=0; std::vector<int> colbad(R,0);
  // output is C rows x R cols: o[c*R+r] == a[r*C+c]
  for(uint32_t c=0;c<C;c++) for(uint32_t r=0;r<R;r++) if(o[(size_t)c*R+r]!=a[(size_t)r*C+c]){bad++;colbad[r]++;}
  printf("R=%u C=%u bad=%zu/%zu (%.1f%%)\n",R,C,bad,n,100.0*bad/n);
  printf("bad per output column (first 40):"); for(uint32_t r=0;r<R&&r<40;r++) printf(" %d",colbad[r]); printf("\n");
  return bad?1:0;}
