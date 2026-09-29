#include <onnxruntime_c_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do{OrtStatus*st_=(x);if(st_){fprintf(stderr,"ERR %s: %s\n",#x,g->GetErrorMessage(st_));return 2;}}while(0)
int main(int argc,char**argv){
  const OrtApi*g=OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const char*prov=argv[2]; int iters=argc>3?atoi(argv[3]):20;
  OrtEnv*env; CK(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING,"t",&env));
  OrtSessionOptions*so; CK(g->CreateSessionOptions(&so));
  if(!strcmp(prov,"webgpu")){
    std::vector<const char*> k,v;
    for(int i=5;i+1<argc;i+=2){k.push_back(argv[i]);v.push_back(argv[i+1]);}
    CK(g->SessionOptionsAppendExecutionProvider(so,"WebGPU",k.data(),v.data(),k.size()));
  }
  OrtSession*s; CK(g->CreateSession(env,argv[1],so,&s));
  OrtMemoryInfo*mi; CK(g->CreateCpuMemoryInfo(OrtArenaAllocator,OrtMemTypeDefault,&mi));
  std::vector<float> x(3*32*32); for(size_t i=0;i<x.size();i++) x[i]=((i*2654435761u)%1000)/1000.f-.5f;
  int64_t sh[]={1,3,32,32}; OrtValue*in; CK(g->CreateTensorWithDataAsOrtValue(mi,x.data(),x.size()*4,sh,4,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,&in));
  OrtAllocator*al; CK(g->GetAllocatorWithDefaultOptions(&al)); char*on; CK(g->SessionGetOutputName(s,0,al,&on)); const char*inn[]={"X"},*outn[]={on}; OrtValue*out=nullptr;
  CK(g->Run(s,nullptr,inn,&in,1,outn,1,&out));
  float*p; CK(g->GetTensorMutableData(out,(void**)&p)); OrtTensorTypeAndShapeInfo*ti; CK(g->GetTensorTypeAndShape(out,&ti)); size_t n; CK(g->GetTensorShapeElementCount(ti,&n));
  {FILE*f=fopen(argv[4],"wb");fwrite(p,4,n,f);fclose(f);} double sum=0; for(size_t i=0;i<n;i++) sum+=p[i];
  printf("OUT n=%zu sum=%.5f first:",n,sum); for(int i=0;i<6&&i<(int)n;i++) printf(" %.5f",p[i]); printf("\n");
  auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<iters;i++){OrtValue*o=nullptr;CK(g->Run(s,nullptr,inn,&in,1,outn,1,&o));g->ReleaseValue(o);}
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/iters;
  printf("%s avg %.3f ms\n",prov,ms); return 0;}
