"""Launch only after fixed TECH and all-control/model provenance gates pass."""
import json,os,subprocess,sys,datetime
from pathlib import Path
from r4.common import file_record,sha256,atomic_json

def main():
    root=Path('/workspace/r5/H2');out=root/'FORMAL_LAUNCH_V1.json'
    if out.exists():raise RuntimeError('Formal launch already recorded; no duplicate launch')
    inputs={}
    for task in ['reacher','pusht']:
        gate_path=root/task/'TECH_GATE.json';gate=json.loads(gate_path.read_text());assert gate['status']=='PASS' and gate['TECH_cases']==4
        for name,h in gate['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h
        for name in ['control_audit','model_identity_audit']:
            r=gate[name];assert file_record(r['path'])=={k:r[k] for k in ['bytes','sha256']}
        inputs[task]={'TECH_GATE':{'path':str(gate_path),**file_record(gate_path)},'model_audit':gate['model_identity_audit'],'control_audit':gate['control_audit']}
    workers=[]
    assignments=[('reacher',0,0,'0-19'),('reacher',1,1,'20-39'),('pusht',0,2,'40-59'),('pusht',1,3,'60-79')]
    for task,worker,gpu,cpus in assignments:
        assets='/workspace/shared_data/r4_assets_reacher' if task=='reacher' else '/workspace/shared_data/r4_assets'
        cmd=['taskset','-c',cpus,sys.executable,'-B','-m','h2.run','--r3-root','/workspace/shared_data/r3','--assets',assets,'--task',task,
            '--control-root','/workspace/r4_'+task+'/trajectories','--output',str(root/task),'--phase','FORMAL','--all-streams',
            '--worker-index',str(worker),'--workers','2','--device','cuda','--threads','1','--tech-receipt',str(root/task/'TECH_GATE.json')]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='/workspace/r5/code:/workspace/r4_v23_execution',R3_ROOT='/workspace/shared_data/r3',
            CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=root/task/f'formal_worker_{worker}.log'
        with log.open('xb') as f:process=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'task':task,'worker_index':worker,'workers':2,'gpu':gpu,'cpus':cpus,'pid':process.pid,'command':cmd,'log':str(log),'target_trajectories':600})
    d={'status':'STARTED','module':'R5_H2','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':inputs,'workers':workers,
        'code':{p.name:file_record(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'new_optimizer_updates':0,'formal_target':2400,'R4_writes':False}
    atomic_json(out,d);print(json.dumps(d),flush=True)

if __name__=='__main__':main()
