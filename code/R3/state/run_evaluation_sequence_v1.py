"""Root-operated, single-attempt R3 evaluation sequence; import starts nothing.

CLI on the already authorized host, after the actual model lock is complete:
  .venv/bin/python -u -B state/run_evaluation_sequence_v1.py --run

Writes only state/evaluation_sequence_v1 and its sibling operation-lock file.
The frozen entry points write their normal scientific artifacts. Existing phase
artifacts or a previous sequence attempt require explicit review, never retry.
SIGINT/SIGTERM stop advancement after the current owned child exits; no process
is killed, no source/result is edited, and no resource is destroyed.
"""
from __future__ import annotations
import argparse, contextlib, datetime, fcntl, hashlib, json, os, shutil, signal, stat, subprocess, time
from pathlib import Path, PurePosixPath

ROOT = Path('/workspace/r3_official_lewm_predictor_refit')
HOST = '6494ba5e1b1a'
GPU_UUIDS = ['GPU-75067296-2e84-cc3f-b93d-19a984e0d7eb', 'GPU-4eaf00b2-ac9c-6e9e-9718-51f1ca58d628']
MODELS = 'manifests/MODELS_AND_SELECTION_LOCK.json'
ROUTING = 'manifests/OPEN_LOOP_ROUTING.json'
PREFLIGHT = 'state/environment_preflight/attempt_001/ENVIRONMENT_PREFLIGHT.json'
DIRECTORY = 'state/evaluation_sequence_v1'
CODE = ('r3/open_loop.py','r3/model.py','r3/common.py','r3/data.py','r3/roles.py',
        'r3/planning.py','r3/reset_fallback.py','r3/env_compat.py','scripts/run_planning.py')
ACTIVE_NAMES = {'r3.train_worker','r3.open_loop','lock_models.py','run_formal_training.py',
                'run_training_technical.py','run_planning.py','build_cache.py','check_real_data_model.py',
                'check_cost_wrapper.py','validate_reset_fallback.py','validate_reacher_reset.py'}


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def local(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('Canonical R3-relative path required')
    current=ROOT
    for part in p.parts:
        current/=part
        if current.is_symlink():raise RuntimeError('Symlink in operation path: '+name)
    if not current.resolve().is_relative_to(ROOT):raise RuntimeError('Operation path escapes authorized root')
    return current
def record(path):return {'sha256':sha(path),'bytes':Path(path).stat().st_size}
def write(path,value,create=False):
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if create:
        with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    else:
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
def verify_files(files):
    if not isinstance(files,dict) or not files:raise RuntimeError('Completion receipt has no files')
    for name,expected in files.items():
        p=local(name);before=p.stat()
        if not stat.S_ISREG(before.st_mode) or record(p)!=expected:raise RuntimeError('Completion file identity differs: '+name)
        after=p.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Completion file changed while hashing: '+name)
def space():
    free=shutil.disk_usage(ROOT).free
    if free<30*(1<<30):raise RuntimeError('Less than required30GiB free space')
    return free
def gpu_identity():
    saved=read(local('state/INSTANCE_REUSE.json'))
    if os.uname().nodename!=HOST or saved['hostname']!=HOST or saved['gpu_uuids']!=GPU_UUIDS:raise RuntimeError('Wrong authorized R3 instance')
    actual=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).split()
    if actual!=GPU_UUIDS:raise RuntimeError('Original GPU UUIDs differ')
def no_compute():
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).split()
    if pids:raise RuntimeError('Global GPU computation is active: '+str(pids))
def no_other_r3():
    if not Path('/proc').is_dir():raise RuntimeError('Linux process evidence unavailable')
    found=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            args=(directory/'cmdline').read_bytes().split(b'\0');args=[x.decode(errors='replace') for x in args if x]
            relevant=any(a in ACTIVE_NAMES or Path(a).name in ACTIVE_NAMES for a in args)
            cwd=(directory/'cwd').resolve()
            if relevant and (cwd==ROOT or any(str(ROOT) in a for a in args)):found.append({'pid':int(directory.name),'argv':args})
        except (FileNotFoundError,ProcessLookupError):continue
    if found:raise RuntimeError('Other R3 training/lock/evaluation/planning process is active: '+json.dumps(found))
def phase_idle():
    # Probe, then release: unmodified children acquire this same global EX lock
    # themselves. Holding a separate EX description across exec would deadlock
    # their frozen entry points. The sequence's own inherited lock stays held.
    with contextlib.ExitStack() as stack:
        for name in ('state/execution_phase.lock','state/model_freeze.lock','state/training_technical_pipeline.lock'):
            p=local(name)
            if not p.exists():raise RuntimeError('Expected existing phase-lock file missing: '+name)
            f=stack.enter_context(p.open('r+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
def nonempty(name):
    p=local(name)
    return p.exists() and (not p.is_dir() or next(p.iterdir(),None) is not None)
def clean_start():
    if local(DIRECTORY).exists():raise RuntimeError('A prior sequence attempt exists; no automatic restart')
    for name in ('artifacts/open_loop/pusht','artifacts/open_loop/reacher','artifacts/planning/FORMAL',
                 'state/planning/FORMAL','state/PLANNING_FORMAL_COMPLETE.json','state/FORMAL_TRAJECTORY_LEDGER.json'):
        if nonempty(name):raise RuntimeError('Existing evaluation/planning output requires explicit review: '+name)
def frozen_identity():
    lock=read(local(MODELS))
    if lock.get('status')!='MODELS_AND_SELECTION_LOCKED':raise RuntimeError('Actual completed model lock required')
    if lock.get('budget',{}).get('formal_updates')!=180000 or lock['budget'].get('formal_jobs')!=6:raise RuntimeError('Model lock is not fixed six-job180000 matrix')
    required=CODE+(ROUTING,PREFLIGHT)
    files={}
    for name in required:
        expected=lock['files'].get(name)
        if not expected or record(local(name))!=expected:raise RuntimeError('Frozen execution input differs: '+name)
        files[name]=expected
    for task in ('pusht','reacher'):
        name=f'manifests/{task}_data_roles.json';roles=read(local(name))
        if record(local(name))!=lock['files'][name] or len(roles['cases']['EVAL'])!=100:raise RuntimeError('Expected frozen100 EVAL cases per task')
        files[name]=lock['files'][name]
    return {'models_lock':record(local(MODELS)),'execution_inputs':files,
            'sequence_source':record(Path(__file__).resolve()),'sequence_pid':os.getpid(),'host':HOST,'gpu_uuids':GPU_UUIDS}
def unchanged(identity):
    if record(local(MODELS))!=identity['models_lock'] or record(Path(__file__).resolve())!=identity['sequence_source']:raise RuntimeError('Model lock/operation source changed')
    verify_files(identity['execution_inputs'])
def verify_receipt(stage,identity):
    if stage['task']:
        name='artifacts/open_loop/'+stage['task']+'/OPEN_LOOP_RECEIPT.json';d=read(local(name))
        if d['status']!='COMPLETE' or d['task']!=stage['task'] or d['evaluation_kind']!='OPEN_LOOP' or d['cases']!=100 or d['arms']!=10 or d['optimizer_updates']!=0 or d['decoder_fits']!=0 or d['closed_loop_trajectories']!=0 or d['physical_state_labels_read']!=0:raise RuntimeError('Open-loop completion/count/exposure differs')
        i=d['identity']
        if i['models_lock_sha256']!=identity['models_lock']['sha256'] or i['device']!='cuda:0' or i['batch_size']!=32 or i['routing_sha256']!=identity['execution_inputs'][ROUTING]['sha256']:raise RuntimeError('Open-loop fixed identity differs')
        if not d['all_future_targets_scoring_only']:raise RuntimeError('Forecast target isolation declaration failed')
    else:
        name='state/PLANNING_FORMAL_COMPLETE.json';d=read(local(name));plan=read(local('state/planning/FORMAL/EXECUTION_PLAN.json'))
        if d['status']!='COMPLETE' or d['phase']!='FORMAL' or d['complete_trajectories']!=800 or d['optimizer_updates']!=0:raise RuntimeError('Fixed800-trajectory completion missing')
        if d['execution_plan_sha256']!=digest(plan) or plan['models_lock_sha256']!=identity['models_lock']['sha256'] or plan['expected_complete_trajectories']!=800:raise RuntimeError('Formal planning model/plan identity differs')
        if {t:sum(x['task']==t for x in plan['cases']) for t in ('pusht','reacher')}!={'pusht':100,'reacher':100}:raise RuntimeError('Formal case matrix differs')
    unchanged(identity)
    return {'path':name,**record(local(name)),'status':d['status'],
            'completion_file_count':len(d['files']),'full_artifact_validation_owned_by_frozen_entrypoint':True}


def run():
    if Path(__file__).resolve()!=ROOT/'state/run_evaluation_sequence_v1.py' or ROOT.is_symlink() or ROOT.resolve()!=ROOT:raise RuntimeError('Only the fixed authorized remote root is supported')
    if os.uname().nodename!=HOST:raise RuntimeError('Wrong authorized host; no local execution')
    binary=local('.venv/bin')/'python'
    # A normal venv interpreter can be a symlink; resolve only after checking
    # the venv directory itself. No alternative interpreter is selected.
    if not binary.is_file():raise RuntimeError('Fixed R3 venv interpreter missing')
    directory=local(DIRECTORY);ownpath=local('state/evaluation_sequence_v1.lock')
    with ownpath.open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clean_start();gpu_identity();space();no_other_r3();no_compute();phase_idle();identity=frozen_identity()
        directory.mkdir(exist_ok=False);write(directory/'STARTED.json',{'version':'R3_FIXED_EVALUATION_SEQUENCE_V1','started_at':now(),'identity':identity,'automatic_retry':False,'optimizer_updates':0},create=True)
        stages=[{'name':'01_open_loop_pusht','task':'pusht','gpu':'0'}, {'name':'02_open_loop_reacher','task':'reacher','gpu':'1'}, {'name':'03_formal_planning','task':None,'gpu':'0,1'}]
        stop={'requested':False};completed=[];old_handlers={};child=None
        def request_stop(signum,_frame):
            stop.update(requested=True,signal=signum)
        for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,request_stop)
        try:
            for stage in stages:
                if stop['requested']:raise RuntimeError('Explicit stop received; current child ended, no next stage')
                gpu_identity();space();no_other_r3();no_compute();phase_idle();unchanged(identity)
                command=[str(binary),'-u','-B']
                if stage['task']:command+=['-m','r3.open_loop',stage['task'],'--routing',ROUTING,'--device','cuda:0','--batch-size','32']
                else:command+=['scripts/run_planning.py','--phase','FORMAL','--routing',ROUTING,'--environment-preflight',PREFLIGHT]
                env=dict(os.environ,R3_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES=stage['gpu'],PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',SDL_VIDEODRIVER='dummy')
                started=time.monotonic();log=directory/(stage['name']+'.log')
                write(directory/(stage['name']+'_STARTED.json'),{'stage':stage,'command':command,'started_at':now(),'models_lock_sha256':identity['models_lock']['sha256'],'log':str(log.relative_to(ROOT))},create=True)
                with log.open('xb') as output:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,pass_fds=(own.fileno(),))
                    while child.poll() is None:
                        write(directory/'STATUS.json',{'status':'STOP_PENDING_CHILD_EXIT' if stop['requested'] else 'RUNNING','utc':now(),'sequence_pid':os.getpid(),'child_pid':child.pid,'stage':stage['name'],'elapsed_seconds':time.monotonic()-started,'completed':completed,'models_lock_sha256':identity['models_lock']['sha256']})
                        time.sleep(5)
                    code=child.returncode
                if code!=0:raise RuntimeError('Child failed; no retry: '+stage['name']+' exit='+str(code))
                receipt=verify_receipt(stage,identity)
                info={'stage':stage['name'],'status':'COMPLETE','child_pid':child.pid,'returncode':code,'completed_at':now(),'wall_seconds':time.monotonic()-started,'receipt':receipt,'log':{'path':str(log.relative_to(ROOT)),**record(log)}}
                write(directory/(stage['name']+'_COMPLETE.json'),info,create=True);completed.append(info);child=None
            unchanged(identity);no_other_r3();no_compute();phase_idle()
            result={'status':'FIXED_EVALUATION_SEQUENCE_COMPLETE','completed_at':now(),'sequence_pid':os.getpid(),'identity':identity,'stages':completed,'optimizer_updates':0,'automatic_retries':0,'success_rates_inspected_for_selection':False}
            write(directory/'COMPLETE.json',result,create=True);write(directory/'STATUS.json',result);return result
        except BaseException as error:
            failed={'status':'FAILED_OR_STOPPED_EXPLICIT_REVIEW_REQUIRED','failed_at':now(),'sequence_pid':os.getpid(),'error':{'type':type(error).__name__,'message':str(error)},'completed':completed,
                    'child_pid':None if child is None else child.pid,'child_returncode':None if child is None else child.poll(),'no_automatic_retry':True,'no_child_process_signalled_or_killed':True,'models_lock_sha256':identity['models_lock']['sha256']}
            write(directory/'FAILED.json',failed,create=True);write(directory/'STATUS.json',failed);raise
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required on the original authorized R3 instance')
    result=run();print(json.dumps({'status':result['status'],'stages':len(result['stages']),'optimizer_updates':0}),flush=True)
