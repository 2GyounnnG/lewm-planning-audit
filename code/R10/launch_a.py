import json, os, pathlib, subprocess, time
root=pathlib.Path('/workspace/r10'); processes=[]; started=time.time()
for i in range(8):
 env=os.environ.copy();env.update(R3_ROOT=str(root/'shared_data/r3'),PYTHONPATH=f'{root}/swm_a:{root}/shared_data/r3:{root}/r4_v23_execution',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(i),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 log=open(root/f'ops/A_worker_{i}.log','x')
 cmd=[str(root/'env_a/bin/python'),str(root/'code/r10/module_a_runner.py'),'--manifest',str(root/'inputs/A_CASE_WINDOWS.json'),'--assets',str(root/'assets/U1'),'--output',str(root/'module_a_formal'),'--phase','FORMAL','--prereg',str(root/'inputs/MODULE_A_FORMAL_PREREG.json'),'--worker-index',str(i),'--workers','8','--device',f'cuda:{i}']
 processes.append((i,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
receipt={'started_unix':started,'workers':[]}
for i,p,log in processes:
 code=p.wait();log.close();receipt['workers'].append({'worker':i,'exit_code':code})
receipt['finished_unix']=time.time();(root/'ops/A_PROCESS_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(int(any(p['exit_code'] for p in receipt['workers'])))
