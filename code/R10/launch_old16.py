import argparse,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--prereg',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path('/workspace/r10');pr=json.loads(pathlib.Path(a.prereg).read_text());assert pr['phase']=='FORMAL' and pr['tech_gate']=='PASS'
module=pr['module'];processes=[];start=time.time()
for i in range(16):
 env=os.environ.copy();env.update(R3_ROOT=str(root/'shared_data/r3'),X1_R3_ROOT=str(root/'shared_data/r3'),PYTHONPATH=f'{root}/shared_data/r3:{root}/r4_v23_execution',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(i%8),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 log=open(root/f'ops/{module}_worker_{i}.log','x')
 cmd=[str(root/'env_old/bin/python'),str(root/'code/r10/old_history_runner.py'),'--task',pr['task'],'--manifest',str(root/'inputs'/pr['manifest_file']),'--assets',str(root/'assets'/pr['asset_subdir']),'--baseline',str(root/'inputs'/pr['baseline_file']),'--plans',str(root/'inputs'/pr['plans_file']),'--output',a.output,'--prereg',a.prereg,'--phase','FORMAL','--worker-index',str(i),'--workers','16','--device',f'cuda:{i%8}']
 processes.append((i,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
receipt={'started_unix':start,'workers':[]}
for i,p,log in processes:
 code=p.wait();log.close();receipt['workers'].append({'worker':i,'exit_code':code})
receipt['finished_unix']=time.time();(root/f'ops/{module}_PROCESS_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(int(any(w['exit_code'] for w in receipt['workers'])))
