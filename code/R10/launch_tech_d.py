import os,subprocess,json,time
from pathlib import Path
root=Path('/workspace/r10');ps=[];started=time.time()
for i in range(4):
 env=os.environ.copy();env.update(R3_ROOT=str(root/'shared_data/r3'),X1_R3_ROOT=str(root/'shared_data/r3'),PYTHONPATH=f'{root}/shared_data/r3:{root}/r4_v23_execution',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(i),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 f=open(root/f'ops/D_TECH_V2_worker_{i}.log','x')
 cmd=[str(root/'env_old/bin/python'),str(root/'code/r10/old_history_runner.py'),'--task','reacher','--manifest',str(root/'inputs/A_CASE_WINDOWS.json'),'--assets',str(root/'assets/U1'),'--baseline',str(root/'inputs/REACHER_BASELINE.json'),'--plans',str(root/'inputs/REACHER_FIRST_PLANS.npz'),'--output',str(root/'module_d_tech_v2'),'--prereg',str(root/'inputs/MODULE_D_TECH_PREREG_V2.json'),'--phase','TECH','--worker-index',str(i),'--workers','4','--device',f'cuda:{i}']
 ps.append((subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT),f))
codes=[]
for p,f in ps:codes.append(p.wait());f.close()
receipt=dict(exit_codes=codes,started_unix=started,finished_unix=time.time());(root/'ops/D_TECH_V2_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(int(any(codes)))
