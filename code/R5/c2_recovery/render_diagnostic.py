"""Four fixed renderer diagnosis resets; no policy, CEM, or evaluated action."""
import argparse,datetime,fcntl,importlib,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from c2.common import *
from c0.reset import apply_symmetric,snapshot
BASE=ROOT/'diagnostic/DIAGNOSTIC_RENDER'
CID='R3_cube_ed7d40ae89e31ab29e099b34'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def trial(device,repeat):
    c=read(BASE/'CONTRACT.json');folder=BASE/f'EGL{device}_repeat{repeat}';identity=digest({'contract':sha(BASE/'CONTRACT.json'),'device':device,'repeat':repeat})
    if (folder/'COMPLETE.json').exists():
        old=read(folder/'COMPLETE.json')
        if old['identity_sha256']!=identity:raise RuntimeError('Diagnostic identity changed')
        for r in old['files'].values():verify(r)
        return
    if (folder/'STARTED.json').exists():raise RuntimeError('Uncounted diagnostic repeat prohibited')
    verify(c['code'])
    for r in c['locks'].values():verify(r)
    if os.environ['MUJOCO_EGL_DEVICE_ID']!=str(device):raise RuntimeError('Renderer device mismatch')
    if sorted(os.sched_getaffinity(0))!=list(range(80,96)):raise RuntimeError('CPU isolation changed')
    sys.path.insert(0,'/workspace/shared_data/r3/source/swm_compat')
    import mujoco as mj,stable_worldmodel as swm
    official=importlib.import_module('stable_worldmodel.world.world');cfg=read(c['locks']['C0_contract']['path'])['official_configuration']['official']
    with np.load(verify(c['asset'])) as f:values={k:f[k].copy() for k in f.files}
    initial={k.split(':',1)[1]:v[None] for k,v in values.items() if k.startswith('initial:')};goal={('goal' if k.split(':',1)[1]=='pixels' else 'goal_'+k.split(':',1)[1]):v[None] for k,v in values.items() if k.startswith('goal:')}
    atomic(folder/'STARTED.json',{'identity_sha256':identity,'pid':os.getpid(),'utc':utc(),'EGL_device':device,'repeat':repeat})
    original=mj.mj_step;phase='WORLD_CONSTRUCTION';calls=[];world=None;failure=None;state={}
    def counted(model,data,*args,**kwargs):
        calls.append({'phase':phase,'substeps':int(kwargs.get('nstep',args[0] if args else 1))});return original(model,data,*args,**kwargs)
    mj.mj_step=counted
    try:
        world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224))
        phase='OFFICIAL_RESET_AND_C0_CLEAR_BEFORE_SOURCE';env=apply_symmetric(world,official,cfg,initial,goal,mj)
        phase='RENDER_ONLY_SNAPSHOT';state=snapshot(env,mj)
    except BaseException as e:failure={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:
        mj.mj_step=original
        if world is not None:
            renderer=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
            if renderer is not None:renderer.close()
            world.close()
    if state:save_npz(folder/'INITIAL_SNAPSHOT.npz',**state)
    result={'status':'COMPLETE_DIAGNOSTIC' if failure is None else 'TECHNICAL_EXECUTION_FAILURE','case_id':CID,'EGL_device':device,'repeat':repeat,'initialization_mj_step_calls':len(calls),'initialization_physics_substeps':sum(x['substeps'] for x in calls),'initialization_ledger':calls,'evaluated_raw_actions':0,'CEM_calls':0,'new_scientific_case_or_stream':False,'failure':failure,'utc':utc()}
    atomic(folder/'result.json',result)
    atomic(folder/'COMPLETE.json',{'identity_sha256':identity,'files':{p.name:record(p) for p in folder.iterdir() if p.is_file() and p.name!='COMPLETE.json'}})
    print(result,flush=True)
def orchestrate():
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'DIAGNOSTIC.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=read(ROOT/'C2_CONTRACT.json');c0=read(c['c0_contract']['path']);asset=next(x['file'] for x in c['assets'] if x['case_id']==CID)
    sources={'C1_GPU6':c['initial_equivalence_sources'][CID],'C2_GPU0_failed':record(ROOT/'closed_loop/EVAL/R4_ALT_CEM_1/H0'/CID/'states.npz')}
    locks={'C0_contract':c['c0_contract'],'C0_reset':c['c0_reset'],'C0_gate':c['c0_gate'],'C1_contract':c['c1_contract'],'C2_contract':record(ROOT/'C2_CONTRACT.json')}
    for name,item in c0['source_map'].items():locks['source:'+name]=item['file']
    for r in [*locks.values(),asset,*sources.values()]:verify(r)
    design={'version':'R5_C2_FIXED_RENDER_DIAGNOSTIC_V1','case_id':CID,'selection':'sole already failed first-planning-point case; no case substitution','order':[[0,0],[0,1],[7,0],[7,1]],'scope':'engineering diagnosis only; no policy, no CEM, no evaluated raw action. Official world construction/reset initialization steps counted separately.','frozen_tolerance':{'atol':0,'rtol':0,'bitwise':True},'asset':asset,'locks':locks,'source_snapshots':sources,'code':record(__file__),'CPU':list(range(80,96)),'gpu6_never_used':True,'formal_resume_authorized_by_this_diagnostic':False}
    freeze(BASE/'CONTRACT.json',design)
    for device,repeat in design['order']:
        env=dict(os.environ,MUJOCO_EGL_DEVICE_ID=str(device),CUDA_VISIBLE_DEVICES=str(device),MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (BASE/f'EGL{device}_repeat{repeat}.log').open('ab') as log:
            p=subprocess.run(['/workspace/env/bin/python','-B','-m','c2_recovery.render_diagnostic','--device',str(device),'--repeat',str(repeat)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Fixed diagnostic process failed; do not rerun silently')
    arrays={}
    for label,source in sources.items():
        with np.load(verify(source)) as f:arrays[label]={k[4:]:f[k].copy() for k in f.files if k.startswith('000:')}
    for device,repeat in design['order']:
        label=f'EGL{device}_repeat{repeat}'
        with np.load(BASE/label/'INITIAL_SNAPSHOT.npz') as f:arrays[label]={k:f[k].copy() for k in f.files}
    pairs=[('EGL0_repeat0','EGL0_repeat1'),('EGL7_repeat0','EGL7_repeat1')]+[(f'EGL{d}_repeat{r}',reference) for d,r in design['order'] for reference in sources]
    fields=[];summary=[]
    for left,right in pairs:
        rows=comparisons(arrays[left],arrays[right]);fields.extend({'left':left,'right':right,**r} for r in rows);render=next(r for r in rows if r['field']=='render')
        summary.append({'left':left,'right':right,'nonrender_bitwise_equal':all(r['bitwise_equal'] for r in rows if r['field']!='render'),'render_bitwise_equal':render['bitwise_equal'],'render_different_channels':render['different_elements'],'render_max_abs':render['max_abs_difference'],'all_fields_bitwise_equal':all(r['bitwise_equal'] for r in rows)})
    csv_write(BASE/'ALL_FIELDS_RAW.csv',fields);csv_write(BASE/'RENDER_COMPARISON_RAW.csv',summary)
    trialresults=[read(BASE/f'EGL{d}_repeat{r}'/'result.json') for d,r in design['order']]
    atomic(BASE/'DIAGNOSTIC_COMPLETE.json',{'status':'COMPLETE_FIXED_DIAGNOSIS','formal_scoring':False,'contract':record(BASE/'CONTRACT.json'),'trials':4,'evaluated_actions':0,'CEM_calls':0,'initialization_mj_step_calls':sum(r['initialization_mj_step_calls'] for r in trialresults),'initialization_physics_substeps':sum(r['initialization_physics_substeps'] for r in trialresults),'summary':summary,'files':{str(p.relative_to(BASE)):record(p) for p in BASE.rglob('*') if p.is_file() and p.name!='DIAGNOSTIC_COMPLETE.json'},'formal_resume':False})
    print(summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int);p.add_argument('--repeat',type=int);a=p.parse_args()
    orchestrate() if a.device is None else trial(a.device,a.repeat)
