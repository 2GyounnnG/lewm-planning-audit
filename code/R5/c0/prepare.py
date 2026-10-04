"""Read-only input preparation and pre-execution C0 contract freeze."""
from __future__ import annotations
import importlib, importlib.metadata, inspect, os, platform, socket, sys
from pathlib import Path
import numpy as np
from .common import *
from .reset import MAIN_CLEAR,IK_CLEAR

def main():
    # Import archived X1/R3 only with bytecode writes prohibited.
    sys.path.insert(0,'/workspace/r4_v23_execution')
    from x1 import core,data
    core.ROOT=X1;core.R3_ROOT=R3
    official=core.task_contract('cube');roles_path=X1/'manifests/cube_data_roles.json';roles=read(roles_path)
    cases=roles['cases']['TECH']
    if len(cases)!=4 or any(c['role']!='TECH' for c in cases):raise RuntimeError('Expected the original four TECH cases')
    source=roles['source'];source_path=Path(source['path'])
    if source_path.stat().st_size!=source['bytes']:raise RuntimeError('Source size mismatch')
    receipt_path=X1/'manifests/cube_source_receipt.json';receipt=read(receipt_path)
    if not any(all(entry.get(k)==v for k,v in source.items()) for entry in receipt['files']):raise RuntimeError('Source does not match earlier full SHA receipt')
    unpack=verify(receipt['unpack_receipt'])
    assets=[]
    with data.RawH5(source_path,keys=['pixels','action','qpos','qvel','privileged_block_0_pos','privileged_block_0_quat']) as raw:
        if raw.action_dim!=5:raise RuntimeError('Cube raw dimension must match actual official 5D action space')
        for case in cases:
            ep=case['source_episode_idx'];s=case['start_raw_index'];g=case['goal_raw_index'];values={}
            if g-s!=25:raise RuntimeError('Frozen endpoint offset changed')
            for key in ('pixels','qpos','qvel','privileged_block_0_pos','privileged_block_0_quat'):
                values['initial:'+key]=raw.array(ep,key,s,s+1)[0]
                values['goal:'+key]=raw.array(ep,key,g,g+1)[0]
            values['actions']=raw.array(ep,'action',s,g)
            if values['actions'].shape!=(25,5) or not np.isfinite(values['actions']).all():raise RuntimeError('Invalid fixed source actions')
            for prefix,name in [('initial','start'),('goal','goal')]:
                pixels=values[prefix+':pixels']
                if hashlib.sha256(pixels.tobytes()).hexdigest()!=case['reset_metadata']['source_'+name+'_pixel_sha256']:raise RuntimeError('Endpoint pixel hash differs')
                if pixels.shape==(3,224,224):pixels=np.moveaxis(pixels,0,-1);values[prefix+':pixels']=pixels
                for key in ('qpos','qvel','privileged_block_0_pos','privileged_block_0_quat'):
                    if not np.array_equal(values[prefix+':'+key],np.asarray(case['reset_metadata'][name][key])):raise RuntimeError('Endpoint metadata differs')
            path=ROOT/'assets'/(case['case_id']+'.npz')
            if path.exists():
                with np.load(path) as f:
                    if set(f.files)!=set(values) or not all(np.array_equal(f[k],v) for k,v in values.items()):raise RuntimeError('Prepared asset differs')
            else:save_npz(path,**values)
            assets.append({'case':case,'asset':record(path),'action_sha256':hashlib.sha256(values['actions'].tobytes()).hexdigest()})
    from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
    from ogbench.manipspace.envs.env import CustomMuJoCoEnv
    from ogbench.manipspace.controllers.diff_ik import DiffIKController
    source_map={}
    for label,cls,names in [('ogbench_manipspace',ManipSpaceEnv,['reset','step','set_control','pre_step']),('ogbench_base',CustomMuJoCoEnv,['reset','set_state']),('ogbench_ik',DiffIKController,['solve'])]:
        p=Path(inspect.getfile(cls));source_map[label]={'file':record(p),'methods':{name:inspect.getsourcelines(getattr(cls,name))[1] for name in names}}
    for component,names in {'lewm':['eval.py','config/eval/cube.yaml','config/eval/solver/cem.yaml'],
                            'swm_compat':['stable_worldmodel/world/world.py','stable_worldmodel/world/env_pool.py','stable_worldmodel/envs/ogbench/cube_env.py']}.items():
        for name in names:source_map[component+'/'+name]={'file':record(R3/'source'/component/name)}
    contract={'version':'R5_C0_V1','module':'C0','evidence_label':'POST_R4_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES',
      'reset_label':'CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT','status':'FROZEN_BEFORE_C_A_A','host':socket.gethostname(),
      'scope':'C0 only; no C1/C2, no closed-loop success inference, no neural training',
      'official_configuration':official,'source_map':source_map,'roles':record(roles_path),'inputs':assets,
      'h5_provenance':{'source':source,'full_sha_reused_from_prior_verified_unpack':record(receipt_path),'unpack_receipt':record(unpack),
          'current_validation':'H5 byte size + prior unpack receipt SHA + TECH start/goal pixels and all reset columns; source H5 not rehashed in C0'},
      'reset_sequence':['Fresh official SWM World with original Cube config','official World.reset(seed=None): no seed patch',
        'clear listed internal/control/cache fields before state setting','official _apply_callables: set_state(source qpos,qvel), set_target_pos(source goal xyz,quat)',
        'official initial pixels and goal injection','snapshot without advancing physics','25 fixed source raw actions through EnvPool.step(mask=None)'],
      'cleared_main_fields':list(MAIN_CLEAR),'controller_clear':{'private_ik':'mj_resetData on IK model/data only',
        'scratch_buffers':list(IK_CLEAR),'target_effector':'identity rotation and zero translation',
        'bookkeeping':'zero _prev_qpos/_prev_qvel and all numeric _prev_ob_info arrays'},
      'unmodified':'No official code, physical model, seed implementation, controller gains/limits, success function, input pixels, or CEM budget changed.',
      'state_definition':'MuJoCo mjSTATE_INTEGRATION for main and IK; named qpos/qvel/act/ctrl/warmstart/applied forces/mocap/eq/user/plugin/time; IK workspace/config; effector pose/prev bookkeeping/task flags; derived physical quantities; dynamics/visual model parameters; every render.',
      'excluded_auxiliary':'Source-overridden initial/goal variation values, unused random-reset cached goal observations and RNG are not physical/controller state. Original official seed omission retained.',
      'tolerance':{'rule':'BITWISE_IDENTITY','numeric_atol':0,'numeric_rtol':0,'dtype_shape_required':True,'finite_required':True,'signed_zero_bits_compared':True,'relaxation_allowed':False},
      'design':{'case_count':4,'case_selection':'all original frozen TECH entries in metadata order','independent_worlds_per_case':2,'tested_raw_steps_per_world':25,
        'action_selection':'unaltered recorded source raw action[start_raw_index:goal_raw_index], frozen before simulation',
        'early_termination':'Record real terminated/truncated every step; technical physical continuation uses existing EnvPool.step with no done-mask; verify _reset_next_step remains false. This is not a closed-loop evaluation. Never auto-reset.',
        'gate':'All four cases: complete state and render bitwise at reset and each of 25 physical raw steps; any mismatch fails. No case substitution.',
        'source_bias':'Report reset minus recorded start for qpos/qvel/block pose and pixels; never gate.'},
      'resource':{'gpu_physical':6,'cpu_affinity':list(range(88,100)),'omp_threads':1,'mkl_threads':1,'openblas_threads':1,'torch_threads':1},
      'environment':{'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ('mujoco','ogbench','numpy','gymnasium','dm-control')}},
      'code':{p.name:record(p) for p in sorted(Path(__file__).parent.glob('*.py'))},'optimizer_updates':0}
    freeze(ROOT/'C0_CONTRACT.json',contract)
    atomic(ROOT/'PREPARATION_COMPLETE.json',{'status':'COMPLETE','contract':record(ROOT/'C0_CONTRACT.json'),'inputs':len(assets),'raw_simulation_steps':0})
    print(json.dumps({'status':'FROZEN_BEFORE_C_A_A','contract_sha256':sha(ROOT/'C0_CONTRACT.json'),'inputs':len(assets)}),flush=True)

if __name__=='__main__':main()
