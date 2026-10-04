"""Three simultaneously started formal seeds; no model/score dependent decisions."""
import json,os,subprocess,time
from pathlib import Path
from . import common

def main():
    gate=common.read_json(common.ROOT/'FORMAL_AUTHORIZATION.json');assert gate['status']=='TECH_PASS_AND_THREE_GPUS_ASSIGNED';jobs=common.jobs();assert len(set(gate['gpu_by_job'].values()))==3
    out=common.ROOT;logs=out/'logs';logs.mkdir(exist_ok=True);workers={};records={};cpus=['112-127','128-143','144-159']
    def launch(job,i,stage):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gate['gpu_by_job'][job['job_id']],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        args=['/workspace/env/bin/python','-B','-m','h3x.train_worker','--job',job['job_id'],'--microbatch','128'] if stage=='TRAIN' else ['/workspace/env/bin/python','-B','-m','h3x.evaluate','--seed',str(job['refit_seed'])]
        log=open(logs/f'{stage}_{job["refit_seed"]}.log','ab',buffering=0)
        proc=subprocess.Popen(['taskset','-c',cpus[i],*args],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd='/workspace/r5/code');workers[i]=proc;records[str(i)]={'job':job,'stage':stage,'pid':proc.pid,'gpu':env['CUDA_VISIBLE_DEVICES'],'cpus':cpus[i],'started_unix':time.time()}
    for i,j in enumerate(jobs):common.require_authorization(j)
    for i,j in enumerate(jobs):launch(j,i,'TRAIN')
    while workers:
        for i,p in list(workers.items()):
            rc=p.poll()
            if rc is None:continue
            previous=records[str(i)]['stage'];records[str(i)]['exit_code']=rc;records[str(i)]['finished_unix']=time.time();del workers[i]
            if rc!=0:records[str(i)]['status']='FAILED_NO_AUTOMATIC_REPLAY'
            elif previous=='TRAIN':
                common.atomic_json(out/f'TRAIN_COMPLETE_{jobs[i]["refit_seed"]}.json',records[str(i)])
                launch(jobs[i],i,'EVALUATE')
            else:records[str(i)]['status']='COMPLETE_TRAIN_AND_MANDATORY_EVALUATION'
        common.atomic_json(out/'SUPERVISOR_STATUS.json',{'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','pid':os.getpid(),'unix':time.time(),'workers':records,'running':len(workers)})
        if workers:time.sleep(10)
    common.atomic_json(out/'SUPERVISOR_COMPLETE.json',{'label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','workers':records,'status':'COMPLETE' if all(r.get('status')=='COMPLETE_TRAIN_AND_MANDATORY_EVALUATION' for r in records.values()) else 'FAILURE_REQUIRES_AUDIT'})
if __name__=='__main__':main()
