"""Exact saved-trajectory distance audit on the two original CPU runtimes."""
import datetime, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_r3 import guard, target, write, SSH, HOST
guard()
PREFIX='recovery_receipts/runtime_diagnosis_001/'
assert not target(PREFIX+'GOAL_NORM_PLATFORM_DIAGNOSIS.json').exists()
CODE=r'''
import hashlib,json,math,platform,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
manifest=json.loads((root/'manifests/RECOVERY_FINAL.json').read_text())
assert hashlib.sha256((root/'manifests/RECOVERY_FINAL.json').read_bytes()).hexdigest()=='f4e0f4c87ae97f3aa57a2ededf97c92f1fe88c1d9bfefb401b508bae0d7d7b4f'
def checked(name):
 p=root/name;b=p.read_bytes();expected=manifest['files'][name]
 assert len(b)==expected['bytes'] and hashlib.sha256(b).hexdigest()==expected['sha256']
 return p
groups={};differences=[]
for name in sorted(manifest['files']):
 if not (name.startswith('artifacts/planning/') and name.endswith('/trajectory.npz')):continue
 result=json.loads(checked(str(Path(name).with_name('result.json'))).read_text())
 assert result['status']=='COMPLETE'
 with np.load(checked(name),allow_pickle=False) as f:
  state=f['physical_state'];goal=f['goal_state'];saved=f['goal_error']
  computed=np.asarray([float(np.linalg.norm(row-goal)) for row in state])
  assert saved.shape==computed.shape
  group=groups.setdefault(result['phase']+'/'+result['task'],{'trajectories':0,'steps':0,'exact_mismatches':0,'max_abs_difference':0.,'max_ULP_distance':0})
  group['trajectories']+=1;group['steps']+=len(saved)
  for i,(a,b) in enumerate(zip(saved,computed)):
   if a!=b:
    diff=abs(float(a)-float(b));ulp=abs(struct.unpack('>Q',struct.pack('>d',a))[0]-struct.unpack('>Q',struct.pack('>d',b))[0])
    group['exact_mismatches']+=1;group['max_abs_difference']=max(group['max_abs_difference'],diff);group['max_ULP_distance']=max(group['max_ULP_distance'],ulp)
    differences.append({'path':name,'step_index':i,'saved':float(a),'computed':float(b),'abs_difference':diff,'ULP_distance':ulp})
print(json.dumps({'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'groups':groups,'differences':differences,
 'formula':'float(np.linalg.norm(row - goal_state)) for each raw saved physical_state row',
 'new_optimizer_updates':0,'new_environment_steps':0,'tolerance_used':None,'diagnostic_not_release':True},allow_nan=False))
'''
venv='/Users/richwang/Documents/ChatGPT/热/jepa_low_label_bridge_gpu_v2/g1/.venv/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
local=subprocess.run([venv,'-B','-',str(target('manifests/RECOVERY_FINAL.json').parents[1])],input=CODE,capture_output=True,text=True,env=env,check=True)
native=json.loads(local.stdout)
remote_root='/workspace/r3_official_lewm_predictor_refit'
remote_prefix="import os,shutil\nassert os.uname().nodename=='6494ba5e1b1a'\nassert shutil.disk_usage('/workspace').free>=30*2**30\n"
command='CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '+shlex.join([remote_root+'/.venv/bin/python','-B','-',remote_root])
remote=subprocess.run(SSH+[HOST,command],input=remote_prefix+CODE,capture_output=True,text=True,check=True)
original=json.loads(remote.stdout)
record={'status':'EXACT_GOAL_NORM_CROSS_PLATFORM_DIAGNOSTIC_NOT_RELEASE',
 'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'local':native,'original_runtime':original,'changes_to_saved_values_or_tolerances':False,
 'source_sha256':hashlib.sha256(CODE.encode()).hexdigest()}
write(PREFIX+'GOAL_NORM_REPLAY_CODE.py',CODE.encode())
write(PREFIX+'GOAL_NORM_PLATFORM_DIAGNOSIS.json',(json.dumps(record,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps({k:record[k]['groups'] for k in ('local','original_runtime')},indent=2))
