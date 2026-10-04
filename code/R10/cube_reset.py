"""The sole C0 reset intervention. Cleared before official start/goal callables."""
from __future__ import annotations
import numpy as np

MAIN_CLEAR=('act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','cfrc_ext','cfrc_int')
IK_CLEAR=('_err','_site_quat','_site_quat_inv','_err_quat','_jac')

def clear_internal(env,mj):
    for key in MAIN_CLEAR:getattr(env._data,key)[...]=0
    # Reset only the controller's private MjData; preserve its model/gains/limits.
    mj.mj_resetData(env._ik._model,env._ik._data)
    for key in IK_CLEAR:getattr(env._ik,key)[...]=0
    pose=env._target_effector_pose
    env._target_effector_pose=type(pose).from_rotation_and_translation(
        rotation=type(pose.rotation()).identity(),translation=np.zeros(3))
    env._prev_qpos=np.zeros_like(env._prev_qpos)
    env._prev_qvel=np.zeros_like(env._prev_qvel)
    # This cache is overwritten by pre_step before each physics call; clear it
    # too, so complete controller/history bookkeeping has no reset residue.
    for key,value in env._prev_ob_info.items():
        if not isinstance(value,np.ndarray):raise TypeError('Unregistered previous-info field: '+key)
        env._prev_ob_info[key]=np.zeros_like(value)

def apply_symmetric(world,official_module,config,initial,goal,mj):
    world.reset(seed=None)  # Official Cube.reset drops seed; do not patch it.
    env=world.envs.envs[0].unwrapped
    clear_internal(env,mj)
    merged={k:v[0] for k,v in {**initial,**goal}.items()}
    official_module._apply_callables(env,config['callables'],merged)
    # Exact dataset-driven policy input injection used by official World.
    prefix=world.infos['pixels'].shape[:2]
    for source in (initial,goal):
        for key,value in source.items():
            if key in world.infos or key in goal:
                world.infos[key]=np.broadcast_to(value[:,None,...],prefix+value.shape[1:]).copy()
    return env

def snapshot(env,mj):
    out={}
    for prefix,model,data in [('physics',env._model,env._data),('controller_ik',env._ik._model,env._ik._data)]:
        spec=mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mj.mj_stateSize(model,spec),dtype=np.float64)
        mj.mj_getState(model,data,state,spec);out[prefix+':integration']=state
        for key in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','eq_active','userdata','plugin_state'):
            out[prefix+':'+key]=np.asarray(getattr(data,key)).copy()
        out[prefix+':time']=np.asarray(data.time)
    for key in IK_CLEAR:out['controller_ik:'+key]=np.asarray(getattr(env._ik,key)).copy()
    for key in ('_site_ids','_damping','_eye','_max_angle_change'):
        out['controller_ik_config:'+key]=np.asarray(getattr(env._ik,key)).copy()
    if env._ik._qp0 is not None:out['controller_ik_config:_qp0']=np.asarray(env._ik._qp0).copy()
    for name,pose in [('target_effector',env._target_effector_pose),('pinch_to_attach',env._T_pa)]:
        out['controller:'+name]=np.r_[pose.rotation().wxyz,pose.translation()].copy()
    out['controller:prev_qpos']=env._prev_qpos.copy();out['controller:prev_qvel']=env._prev_qvel.copy()
    for key,value in env._prev_ob_info.items():out['controller:prev_info:'+key]=np.asarray(value).copy()
    for key in ('_success','_reset_next_step','_target_block'):
        out['task:'+key]=np.asarray(getattr(env,key))
    # Derived physical quantities and dynamics/visual configuration complement
    # MuJoCo's complete integration state. RNG is not physical/controller state.
    for key in ('qacc','qfrc_actuator','qfrc_constraint','site_xpos','site_xmat','cfrc_ext'):
        out['physical_derived:'+key]=np.asarray(getattr(env._data,key)).copy()
    for key in ('body_mass','body_inertia','dof_damping','geom_friction','actuator_gainprm','actuator_biasprm','geom_rgba','cam_pos','cam_quat'):
        out['model:'+key]=np.asarray(getattr(env._model,key)).copy()
    out['render']=np.asarray(env.render()).copy()
    return out
