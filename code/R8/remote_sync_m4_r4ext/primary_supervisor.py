"""Reacher dependency dispatcher; never trains, deletes logs, or runs batch2."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .common import ARMS,STREAMS,atomic_json,file_record,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--line',required=True);p.add_argument('--unpacked',required=True);a=p.parse_args()
    root=Path(a.r3_root);line=Path(a.line);line.mkdir(parents=True,exist_ok=True)
    lock=(line/'PRIMARY_SUPERVISOR.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (line/'SUPERVISOR_LAUNCH.json').exists():raise RuntimeError('Existing dispatcher journal: inspect before restarting to avoid duplicate workers')
    atomic_json(line/'SUPERVISOR_LAUNCH.json',{'version':'R4_PRIMARY_DISPATCH_001','task':'reacher','pid':os.getpid(),'started':time.time(),'code_sha256':sha256(__file__),'no_second_batch':True})
    env=os.environ.copy();env.update(PYTHONPATH='/workspace/r4_v23_execution',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    jobs={};events=[]
    def event(kind,**payload):
        record={'time':time.time(),'event':kind,**payload};events.append(record);atomic_json(line/'DISPATCH_EVENTS.json',events);print(json.dumps(record),flush=True)
    def launch(name,module,args,gpu=2):
        if name in jobs:raise RuntimeError('Duplicate job '+name)
        e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu);f=(line/(name+'.log')).open('ab')
        cmd=['taskset','-c','24-47','/workspace/env/bin/python','-m',module]+args
        process=subprocess.Popen(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[name]=process;event('STARTED',name=name,pid=process.pid,command=cmd);return process
    def require_done(name):
        process=jobs[name]
        while process.poll() is None:time.sleep(5)
        event('FINISHED',name=name,exit_code=process.returncode)
        if process.returncode:raise RuntimeError('Technical subprocess failure: '+name)
    try:
        routes=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks']['reacher']['arms'];last=None
        while True:
            ready=[]
            for route in routes:
                if route.get('step')!=30000:continue
                rec=route['checkpoint'];path=root/rec['path']
                if path.exists() and path.stat().st_size==rec['bytes'] and sha256(path)==rec['sha256']:ready.append(route['seed'])
            unpacked=Path(a.unpacked)/'UNPACKED.json';boot=line/'boot/BOOT_GATE.json'
            status={'ready_refit_seeds':ready,'unpack_receipt':unpacked.exists(),'boot_pass':boot.exists() and json.loads(boot.read_text())['status']=='PASS'}
            if status!=last:event('WAITING_ASSETS',**status);last=status
            if len(ready)==3 and status['unpack_receipt'] and status['boot_pass']:break
            time.sleep(15)
        records=json.loads(unpacked.read_text())['records'];source=json.loads((root/'manifests/reacher_source_map.json').read_text())['assets']
        for item in source.values():
            record=next(r for r in records if r['bytes']==item['bytes'] and r['sha256']==item['sha256']);target=Path(record['path'])
            if target.stat().st_size!=item['bytes']:raise RuntimeError('Verified H5 source changed')
            destination=root/item['path'];destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists() or destination.is_symlink():
                if destination.resolve()!=target.resolve():raise RuntimeError('Existing source symlink differs')
            else:destination.symlink_to(target)
        launch('export_cases','r4.export_cases',['--r3-root',str(root),'--output',a.assets,'--task','reacher']);require_done('export_cases')
        common=['--r3-root',str(root),'--assets',a.assets,'--task','reacher'];trajectory=str(line/'trajectories')
        launch('s1_offline','r4.s1',['--r3-root',str(root),'--task','reacher','--module','offline','--output',str(line/'s1')])
        for i in range(2):launch('original_worker'+str(i),'r4.runner',common+['--output',trajectory,'--stream','R3_ORIGINAL','--worker-index',str(i),'--workers','2'],2+i)
        launch('s0_a','r4.s0',common+['--output',str(line/'s0'),'--check','a'])
        launch('termination_tech','r4.s0',common+['--output',str(line/'s0'),'--check','termination'])
        launch('tech','r4.runner',common+['--output',trajectory,'--stream','R3_ORIGINAL','--phase','TECH','--arm','H0','--case-limit','4'])
        require_done('s0_a')
        if json.loads((line/'s0/S0_A.json').read_text())['status']!='PASS':raise RuntimeError('S0(a) failed')
        for i in range(2):launch('s2_worker'+str(i),'r4.s2',common+['--output',str(line/'s2'),'--trajectories',trajectory,'--s0-a',str(line/'s0/S0_A.json'),'--worker-index',str(i),'--workers','2'],2+i)
        # Replay check follows TECH directly and does not wait for original400.
        require_done('tech');launch('s0_bcd','r4.s0',common+['--output',str(line/'s0'),'--check','bcd','--trajectories',trajectory])
        for i in range(2):require_done('original_worker'+str(i))
        original=list((line/'trajectories/FORMAL/R3_ORIGINAL').glob('*/*/COMPLETE.json'))
        if len(original)!=400:raise RuntimeError('Original400 not complete')
        for path in original:
            for name,rec in json.loads(path.read_text())['files'].items():
                if file_record(path.parent/name)!=rec:raise RuntimeError('Original output hash mismatch')
        atomic_json(line/'ORIGINAL_400_VERIFIED.json',{'status':'PASS','completed':400,'all_completion_hashes_verified':True})
        launch('s1_cross','r4.s1',['--r3-root',str(root),'--task','reacher','--module','cross','--output',str(line/'s1'),'--trajectories',trajectory])
        for i in range(2):launch('alt1_worker'+str(i),'r4.runner',common+['--output',trajectory,'--stream','R4_ALT_CEM_1','--worker-index',str(i),'--workers','2'],2+i)
        require_done('s0_bcd');require_done('termination_tech')
        receipts=[json.loads(p.read_text()) for p in (line/'s0').glob('*_S0_BCD.json')];term=json.loads((line/'s0/TERMINATION_TECH_RAW.json').read_text())
        sim_legal=bool(receipts) and all(r['status']=='PASS' for r in receipts) and not term['fixed25_unavailable_observed']
        if not sim_legal:
            atomic_json(line/'SIM_TECHNICAL_LIMITATION.json',{'status':'TECHNICALLY_UNEVALUABLE','roles':['SIM_LAT_RERANK','SIM_TASK_RERANK'],'expected_logical_records_per_role':60,'completed':None,'reason':'TRUE_DMC_LAST_PREVENTS_FIXED25' if term['fixed25_unavailable_observed'] else 'S0_REPLAY_NOT_ESTABLISHED','gate_files':{str(p):file_record(p) for p in sorted((line/'s0').glob('*.json'))}})
        roles=['H0_MENU_RERANK']+(['SIM_LAT_RERANK','SIM_TASK_RERANK'] if sim_legal else [])
        # Menu jobs run in parallel to alternate learning streams and initial forks.
        for role in roles:
            for stream in STREAMS:
                for i in range(2):launch(role+'_'+stream+'_'+str(i),'r4.s3',common+['--output',trajectory,'--role',role,'--stream',stream,'--s0-bcd',str(line/'s0'),'--worker-index',str(i),'--workers','2'],2+i)
                for i in range(2):require_done(role+'_'+stream+'_'+str(i))
        for i in range(2):require_done('alt1_worker'+str(i))
        for i in range(2):launch('alt2_worker'+str(i),'r4.runner',common+['--output',trajectory,'--stream','R4_ALT_CEM_2','--worker-index',str(i),'--workers','2'],2+i)
        for name in ['alt2_worker0','alt2_worker1','s1_offline','s1_cross','s2_worker0','s2_worker1']:require_done(name)
        event('PRIMARY_DISPATCH_FINISHED',review_required_before_main_completion_status=True)
    except BaseException as exc:
        event('DISPATCH_TECHNICAL_FAILURE',error_type=type(exc).__name__,error=str(exc),already_running_jobs_preserved=True)
        raise

if __name__=='__main__':main()
