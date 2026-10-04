"""Read-only CPU formula audit. References are evidence, never LR replacements."""
import ast, base64, datetime, hashlib, json, math, platform, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_r3 import guard, target, write, remote_python

guard()
prefix = 'recovery_receipts/runtime_diagnosis_001/'
assert not target(prefix+'LR_PLATFORM_DIAGNOSIS.json').exists()
remote_code = r'''
import ast,base64,datetime,hashlib,json,math,os,platform,shutil,struct,subprocess
from pathlib import Path
r=Path('/workspace/r3_official_lewm_predictor_refit')
assert os.uname().nodename=='6494ba5e1b1a'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert uuids==['GPU-75067296-2e84-cc3f-b93d-19a984e0d7eb','GPU-4eaf00b2-ac9c-6e9e-9718-51f1ca58d628']
assert shutil.disk_usage(r).free>=30*2**30
source=(r/'r3/train_worker.py').read_bytes()
sha=hashlib.sha256(source).hexdigest()
assert sha=='c9c6db8d9f1c3a8e572ce5c007cba0fa457de3bdfcb3300fc0dd781d5ebd72ae'
tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('lr_at','scheduler_record')]
namespace={'math':math}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-formula-only>','exec'),namespace)
expected=[namespace['lr_at'](s) for s in range(1,30001)]
raw=struct.pack('<30000d',*expected)
jobs=[]
for p in sorted((r/'artifacts/train').glob('R3_*/updates.jsonl')):
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 values=[x['lr'] for x in rows]
 jobs.append({'job':p.parent.name,'rows':len(rows),'updates_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'steps_contiguous':[x['step'] for x in rows]==list(range(1,30001)),
 'lr_exact_mismatch_count':sum(x!=y for x,y in zip(values,expected)),
 'lr_binary64_little_endian_sha256':hashlib.sha256(struct.pack('<30000d',*values)).hexdigest()})
libm_paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libm.so' in line})
print(json.dumps({'hostname':os.uname().nodename,'gpu_uuids':uuids,'python':platform.python_version(),
 'machine':platform.machine(),'libc':platform.libc_ver(),'libm_files':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libm_paths},
 'source_sha256':sha,'expected_lr_sequence_sha256':hashlib.sha256(raw).hexdigest(),
 'expected_lr_binary64_little_endian_base64':base64.b64encode(raw).decode(),'jobs':jobs,
 'new_optimizer_updates':0,'new_environment_steps':0,'generated_by_formula_not_journal':True}))
'''
remote = remote_python(remote_code)
raw = base64.b64decode(remote.pop('expected_lr_binary64_little_endian_base64'), validate=True)
assert hashlib.sha256(raw).hexdigest() == remote['expected_lr_sequence_sha256']
assert len(remote['jobs']) == 6 and all(x['rows']==30000 and x['steps_contiguous'] and x['lr_exact_mismatch_count']==0 for x in remote['jobs'])
source = target('r3/train_worker.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == remote['source_sha256']
tree = ast.parse(source)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('lr_at','scheduler_record')]
namespace={'math':math}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen-formula-only>','exec'),namespace)
local=[namespace['lr_at'](s) for s in range(1,30001)]
reference=struct.unpack('<30000d',raw)
local_raw=struct.pack('<30000d',*local)
differences=[]
for step,(a,b) in enumerate(zip(local,reference),1):
 if a!=b:
  ai=struct.unpack('>Q',struct.pack('>d',a))[0];bi=struct.unpack('>Q',struct.pack('>d',b))[0]
  differences.append({'step':step,'local':a,'original_runtime':b,'absolute_difference':abs(a-b),'ULP_distance':abs(ai-bi)})
local_jobs=[]
for job in remote['jobs']:
 p=target('artifacts/train/'+job['job']+'/updates.jsonl')
 assert hashlib.sha256(p.read_bytes()).hexdigest()==job['updates_sha256']
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 assert [x['step'] for x in rows]==list(range(1,30001))
 assert all(x['lr']==y for x,y in zip(rows,reference))
 local_jobs.append({'job':job['job'],'rows':len(rows),'exact_vs_original_runtime':True,'mismatches_vs_native_mac_formula':sum(x['lr']!=y for x,y in zip(rows,local))})
record={'status':'CROSS_PLATFORM_LIBM_DIFFERENCE_CONFIRMED_NOT_RELEASE_ACCEPTANCE',
 'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote':remote,
 'local':{'python':sys.version,'machine':platform.machine(),'platform':platform.platform(),
 'lr_binary64_little_endian_sha256':hashlib.sha256(local_raw).hexdigest(),'jobs':local_jobs},
 'difference_count_per_job':len(differences),'maximum_absolute_difference':max(x['absolute_difference'] for x in differences),
 'maximum_ULP_distance':max(x['ULP_distance'] for x in differences),'differences':differences,
 'reference_use':'Diagnostic evidence only. Must not be used as a per-step replacement or lookup to pass acceptance.',
 'new_optimizer_updates':0,'new_environment_steps':0,'existing_result_or_tolerance_changes':False}
write(prefix+'ORIGINAL_RUNTIME_FORMULA_LR_LE_F64.bin',raw)
write(prefix+'NATIVE_MAC_FORMULA_LR_LE_F64.bin',local_raw)
write(prefix+'ORIGINAL_RUNTIME_REPLAY_CODE.py',remote_code.encode())
write(prefix+'LR_PLATFORM_DIAGNOSIS.json',(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps({k:record[k] for k in ('status','difference_count_per_job','maximum_absolute_difference','maximum_ULP_distance')},ensure_ascii=False))
