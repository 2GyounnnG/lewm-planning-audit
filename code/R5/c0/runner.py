"""C0 fixed four-case A/A execution. No C1/C2 dispatcher exists here."""
from __future__ import annotations
import datetime, importlib, os, socket, sys, time, traceback
from pathlib import Path
import numpy as np
from .common import *
from .reset import apply_symmetric,snapshot

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def trial(contract,item,repeat):
    import mujoco as mj
    import stable_worldmodel as swm
    official_module=importlib.import_module('stable_worldmodel.world.world')
    cfg=contract['official_configuration']['official'];case=item['case'];cid=case['case_id']
    folder=ROOT/'trials'/cid/str(repeat);identity=digest({'contract':sha(ROOT/'C0_CONTRACT.json'),'case':cid,'repeat':repeat})
    if (folder/'COMPLETE.json').exists():
        saved=read(folder/'COMPLETE.json')
        if saved['identity_sha256']!=identity:raise RuntimeError('Trial identity changed')
        for r in saved['files'].values():verify(r)
        return read(folder/'result.json')
    if (folder/'STARTED.json').exists():raise RuntimeError('Interrupted trial: preserve raw evidence and require explicit technical accounting, not an uncounted repeat')
    with np.load(verify(item['asset'])) as f:values={k:f[k].copy() for k in f.files}
    initial={k.split(':',1)[1]:v[None] for k,v in values.items() if k.startswith('initial:')}
    goal={('goal' if k.split(':',1)[1]=='pixels' else 'goal_'+k.split(':',1)[1]):v[None] for k,v in values.items() if k.startswith('goal:')}
    atomic(folder/'STARTED.json',{'identity_sha256':identity,'case_id':cid,'repeat':repeat,'pid':os.getpid(),'utc':utc(),'asset':item['asset']})
    start=time.monotonic();world=None;arrays={};rows=[];phase='world_construction';calls=[];original_step=mj.mj_step
    def counted(model,data,*args,**kwargs):
        nstep=int(kwargs.get('nstep',args[0] if args else 1))
        calls.append({'phase':phase,'nstep':nstep});return original_step(model,data,*args,**kwargs)
    mj.mj_step=counted
    failure=None;bias=[];sourcepixels=values['initial:pixels']
    try:
        world=swm.World(**cfg['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224))
        phase='official_reset_and_clear'
        env=apply_symmetric(world,official_module,cfg,initial,goal,mj)
        if tuple(world.envs.single_action_space.shape)!=(5,) or env._n_steps!=25:raise RuntimeError('Physical action dimension or substep count differs')
        # The first snapshot follows official reset and source state/goal callables.
        first=snapshot(env,mj)
        for key,value in first.items():arrays[f'000:{key}']=value
        for key in ('qpos','qvel'):
            actual=first['physics:'+key];expected=values['initial:'+key]
            for i,(a,b) in enumerate(zip(actual,expected,strict=True)):
                bias.append({'case_id':cid,'repeat':repeat,'field':key,'component':i,'recorded':float(b),'reset':float(a),'difference':float(a-b),'report_only':True})
        pixel_delta=np.abs(first['render'].astype(np.float64)-sourcepixels.astype(np.float64))
        source_report={'render_sha256':hashlib.sha256(first['render'].tobytes()).hexdigest(),
          'source_pixel_sha256':hashlib.sha256(sourcepixels.tobytes()).hexdigest(),'pixel_mae':float(pixel_delta.mean()),
          'pixel_max_abs':float(pixel_delta.max()),'pixel_bitwise_equal':first['render'].tobytes()==sourcepixels.tobytes(),
          'block_position_max_abs':float(np.max(np.abs(env._data.joint('object_joint_0').qpos[:3]-values['initial:privileged_block_0_pos']))),
          'block_quaternion_max_abs':float(np.max(np.abs(env._data.joint('object_joint_0').qpos[3:]-values['initial:privileged_block_0_quat']))),
          'report_only':True}
        phase='C_A_A_fixed_action_replay'
        for i,action in enumerate(values['actions']):
            if env._reset_next_step:raise RuntimeError('Unexpected automatic reset: no replay step permitted')
            before=len(calls);tick=time.monotonic();out=world.envs.step(action[None]);elapsed=time.monotonic()-tick
            if len(calls)!=before+1:raise RuntimeError('One technical raw action must cause exactly one mj_step call')
            state=snapshot(env,mj)
            for key,value in state.items():arrays[f'{i+1:03d}:{key}']=value
            for key,value in zip(('reward','terminated','truncated'),out[1:4]):arrays[f'{i+1:03d}:step:{key}']=np.asarray(value).copy()
            rows.append({'case_id':cid,'repeat':repeat,'raw_step':i+1,'terminated':bool(out[2][0]),'truncated':bool(out[3][0]),
              'reward':float(out[1][0]),'seconds':elapsed,'physics_substeps':calls[-1]['nstep'],
              'action_sha256':hashlib.sha256(action.tobytes()).hexdigest(),'render_sha256':hashlib.sha256(state['render'].tobytes()).hexdigest()})
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite full-state snapshot')
    except BaseException as e:
        failure={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:
        mj.mj_step=original_step
        if world is not None:
            renderer=getattr(world.envs.envs[0].unwrapped,'_renderer',None)
            if renderer is not None:renderer.close()
            world.close()
    save_npz(folder/'states.npz',**arrays)
    if rows:csv_write(folder/'STEP_RAW.csv',rows)
    if bias:csv_write(folder/'SOURCE_START_COMPONENT_RAW.csv',bias)
    result={'status':'COMPLETE' if failure is None else 'TECHNICAL_EXECUTION_FAILURE','case_id':cid,'repeat':repeat,'identity_sha256':identity,
      'executed_test_raw_steps':len(rows),'initialization_mj_step_calls':sum(x['phase']!='C_A_A_fixed_action_replay' for x in calls),
      'initialization_physics_substeps':sum(x['nstep'] for x in calls if x['phase']!='C_A_A_fixed_action_replay'),
      'test_physics_substeps':sum(x['nstep'] for x in calls if x['phase']=='C_A_A_fixed_action_replay'),
      'mj_step_ledger':calls,'source_start_bias':source_report if 'source_report' in locals() else None,
      'termination_steps':[r['raw_step'] for r in rows if r['terminated']],'truncation_steps':[r['raw_step'] for r in rows if r['truncated']],
      'wall_seconds':time.monotonic()-start,'failure':failure,'optimizer_updates':0,'CEM_calls':0,'utc':utc()}
    atomic(folder/'result.json',result)
    if failure is None:atomic(folder/'COMPLETE.json',{'identity_sha256':identity,'files':{p.name:record(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='COMPLETE.json'}})
    print(json.dumps({'trial':cid,'repeat':repeat,'status':result['status'],'steps':len(rows),'seconds':result['wall_seconds']}),flush=True)
    return result

def summarize(contract,results):
    raw=[];summary=[];bias=[]
    for item in contract['inputs']:
        cid=item['case']['case_id'];base=ROOT/'trials'/cid
        pair=[r for r in results if r['case_id']==cid]
        if len(pair)!=2 or any(r['status']!='COMPLETE' for r in pair):
            summary.append({'case_id':cid,'raw_step':-1,'fields':0,'full_state_equal':False,'render_equal':False,'all_equal':False,'max_abs_difference':None,'reason':'TECHNICAL_EXECUTION_FAILURE'});continue
        with np.load(base/'0/states.npz') as fa,np.load(base/'1/states.npz') as fb:
            for step in range(26):
                prefix=f'{step:03d}:';a={k[len(prefix):]:fa[k] for k in fa.files if k.startswith(prefix)};b={k[len(prefix):]:fb[k] for k in fb.files if k.startswith(prefix)}
                fields=comparisons(a,b)
                raw.extend({'case_id':cid,'raw_step':step,**r} for r in fields)
                state=all(r['bitwise_equal'] for r in fields if r['field']!='render');render=all(r['bitwise_equal'] for r in fields if r['field']=='render')
                summary.append({'case_id':cid,'raw_step':step,'fields':len(fields),'full_state_equal':state,'render_equal':render,'all_equal':state and render,
                   'max_abs_difference':max(r['max_abs_difference'] for r in fields),'reason':'' if state and render else 'BITWISE_MISMATCH'})
        for r in pair:bias.append({'case_id':cid,'repeat':r['repeat'],**r['source_start_bias']})
    if raw:csv_write(ROOT/'C_A_A_FIELD_RAW.csv',raw)
    csv_write(ROOT/'C_A_A_RAW_TABLE.csv',summary)
    if bias:csv_write(ROOT/'SOURCE_START_BIAS_RAW.csv',bias)
    good=len(summary)==104 and all(r['all_equal'] for r in summary)
    failures=[r for r in raw if not r['bitwise_equal']]
    gate={'status':'PASS_C_A_A' if good else 'CUBE_TECHNICALLY_UNEVALUABLE','module':'C0','label':contract['reset_label'],
       'contract':record(ROOT/'C0_CONTRACT.json'),'case_count':4,'independent_resets':len(results),'test_raw_steps':sum(r['executed_test_raw_steps'] for r in results),
       'expected_case_timepoints':104,'completed_case_timepoints':len(summary),'failed_field_comparisons':len(failures),
       'mismatching_fields':sorted({r['field'] for r in failures}),'failed_cases':sorted({r['case_id'] for r in summary if not r['all_equal']}),
       'tables':{p.name:record(p) for p in ROOT.glob('*RAW*.csv')},'results':[record(ROOT/'trials'/r['case_id']/str(r['repeat'])/'result.json') for r in results],
       'initialization_mj_step_calls':sum(r['initialization_mj_step_calls'] for r in results),'initialization_physics_substeps':sum(r['initialization_physics_substeps'] for r in results),
       'test_physics_substeps':sum(r['test_physics_substeps'] for r in results),'optimizer_updates':0,'CEM_calls':0,
       'C1_C2_started':False,'next_action':'C0-only scope: report gate, do not start C1/C2','utc':utc()}
    atomic(ROOT/'C0_GATE.json',gate)
    return gate

def main():
    contract=read(ROOT/'C0_CONTRACT.json')
    if socket.gethostname()!=contract['host']:raise RuntimeError('Host changed')
    if sorted(os.sched_getaffinity(0))!=contract['resource']['cpu_affinity']:raise RuntimeError('CPU isolation mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise RuntimeError('GPU isolation mismatch')
    for r in contract['code'].values():verify(r)
    for entry in contract['source_map'].values():verify(entry['file'])
    sys.path.insert(0,str(R3/'source/swm_compat'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    atomic(ROOT/'JOB_STARTED.json',{'pid':os.getpid(),'utc':utc(),'contract':record(ROOT/'C0_CONTRACT.json'),'affinity':sorted(os.sched_getaffinity(0)),'gpu':6})
    results=[]
    for item in contract['inputs']:
        for repeat in (0,1):results.append(trial(contract,item,repeat))
    gate=summarize(contract,results)
    print(json.dumps({'status':gate['status'],'failed_fields':gate['failed_field_comparisons'],'raw_steps':gate['test_raw_steps']}),flush=True)

if __name__=='__main__':main()
