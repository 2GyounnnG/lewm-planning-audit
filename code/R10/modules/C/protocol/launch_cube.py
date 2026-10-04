"""Sixteen Cube workers across8 GPUs; stop all on cumulative resolved delivery below90%, no retry."""
import argparse,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--prereg',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path('/workspace/r10');pr=json.loads(pathlib.Path(a.prereg).read_text());assert pr['phase']=='FORMAL' and pr['tech_gate']=='PASS' and pr['module']=='C'
processes=[];start=time.time();out=pathlib.Path(a.output)
for i in range(16):
 env=os.environ.copy();env.update(R3_ROOT=str(root/'shared_data/r3'),X1_R3_ROOT=str(root/'shared_data/r3'),PYTHONPATH=f'{root}/shared_data/r3:{root}/r4_v23_execution',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID=str(i%8),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 log=open(root/f'ops/C_worker_{i}.log','x')
 cmd=[str(root/'env_old/bin/python'),str(root/'code/r10/old_history_runner_c.py'),'--task','cube','--manifest',str(root/'inputs'/pr['manifest_file']),'--assets',str(root/'assets'/pr['asset_subdir']),'--baseline',str(root/'inputs'/pr['baseline_file']),'--plans',str(root/'inputs'/pr['plans_file']),'--output',a.output,'--prereg',a.prereg,'--phase','FORMAL','--worker-index',str(i),'--workers','16','--device',f'cuda:{i%8}']
 processes.append((i,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
stopped=False
while any(p.poll() is None for _,p,_ in processes):
 good=len(list(out.glob('FORMAL/*/*/*/H_REAL3_REPLAN/COMPLETE.json')))
 bad=len(list(out.glob('FORMAL/*/*/*/H_REAL3_REPLAN/FAILURE.json')))+len(list(out.glob('FORMAL/*/*/*/H_POLICY/FAILURE.json')))
 if good+bad and good/(good+bad)<.9:
  stopped=True;receipt={'reason':'CUBE_GLOBAL_DELIVERY_BELOW_90_PERCENT','delivered':good,'resolved_missing_or_invalid':bad,'ratio':good/(good+bad),'unix':time.time(),'no_retry':True}
  (root/'ops/C_STOP_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
  for _,p,_ in processes:
   if p.poll() is None:p.terminate()
  break
 time.sleep(5)
receipt={'started_unix':start,'stopped_on_delivery_gate':stopped,'workers':[]}
for i,p,log in processes:
 try:code=p.wait(timeout=15)
 except subprocess.TimeoutExpired:p.kill();code=p.wait()
 log.close();receipt['workers'].append({'worker':i,'exit_code':code})
receipt['finished_unix']=time.time();(root/'ops/C_PROCESS_COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(int(stopped or any(w['exit_code'] for w in receipt['workers'])))
