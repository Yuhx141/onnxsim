import sys,time,glob,os
import numpy as np
sys.path.insert(0,'.')
import axcl_session
blob=open(sorted(glob.glob('' + (sys.argv[1] if len(sys.argv) > 1 else '/mnt/data/cache/claude-work/u16-seg-cache') + '/resnetv15_stage2_conv2_fwd.*.axmodel'))[0],'rb').read()
print('model bytes',len(blob))
with axcl_session.AXSession() as s:
    T={'load':[], 'write_in':[], 'run_cmd':[], 'read':[], 'unload':[]}
    for it in range(6):
        t=time.perf_counter(); m=s.load(blob); T['load'].append(time.perf_counter()-t)
        ins=[np.random.rand(*sp.shape).astype(np.float32) for sp in m.inputs]
        nb=sum(i.nbytes for i in ins)
        # replicate run() phases
        t=time.perf_counter()
        bufs=[np.ascontiguousarray(x,dtype=sp.dtype).tobytes() for x,sp in zip(ins,m.inputs)]
        for k,b in enumerate(bufs):
            open(os.path.join(s.host_dir,f"t/i{k}.bin"),'wb').write(b)
        T['write_in'].append(time.perf_counter()-t)
        e0=s.exec_us; t=time.perf_counter()
        ys=s.run(m,ins); tot=time.perf_counter()-t
        T['run_cmd'].append(tot); eng=(s.exec_us-e0)/1000
        t=time.perf_counter(); s.unload(m); T['unload'].append(time.perf_counter()-t)
    print('input MB %.1f'%(nb/1e6),'out MB %.1f'%(sum(y.nbytes for y in ys)/1e6),'engine ms %.1f'%eng)
    for k,v in T.items(): print(k.ljust(9),'median ms %.1f'%(1000*sorted(v)[len(v)//2]))
