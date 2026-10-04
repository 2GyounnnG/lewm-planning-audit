"""Strict actual-author weights and cached/native interface TECH comparison."""
import copy
import numpy as np
from . import core,data

def check(task,device='cpu'):
    import torch
    core.policy();model=core.load_model(task,device);m=core.r3('model');before={k:m.tensor_sha256(v) for k,v in model.state_dict().items()}
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json');case=roles['cases']['TECH'][0];start=case['open_loop_window_start_raw']
    with data.RawH5(core.verify(roles['source']),keys=['pixels','action']) as raw:
        pixels=raw.array(case['source_episode_idx'],'pixels',start,start+11)[[0,5,10]]
        actions=raw.array(case['source_episode_idx'],'action',start,start+35)
    if pixels.shape[-1]==3:pixels=np.moveaxis(pixels,-1,1)
    transform=core.r3('data').image_transform();x=transform(torch.from_numpy(pixels)).to(device)[None]
    a=torch.from_numpy(data.processor(task).transform(actions).astype(np.float32).reshape(1,7,-1)).to(device)
    tolerance_path=core.R3_ROOT/'manifests/NUMERICAL_TOLERANCES.json'
    tolerances=core.read(tolerance_path);tol=tolerances['comparisons']['rollout_wrapper'];checks={}
    with torch.inference_mode():
        z=model.encode({'pixels':x.clone()})['emb'];cached=m.cached_rollout(model,z,a,5)
        native=model.rollout({'pixels':x[:,None].clone()},a[:,None].clone())['predicted_emb'][:,0,3:]
        error=float((cached-native).abs().max());torch.testing.assert_close(cached,native,**tol)
        checks['cached_vs_native_rollout_max_abs']=error
        first=x[:,-1:];aa=a[:,2:7]
        z1=model.encode({'pixels':first.clone()})['emb'];online=m.cached_rollout(model,z1,aa,5)
        official=model.rollout({'pixels':first[:,None].clone()},aa[:,None].clone())['predicted_emb'][:,0,1:]
        error=float((online-official).abs().max());torch.testing.assert_close(online,official,**tol)
        checks['policy_single_frame_history_growth_max_abs']=error
        info={'pixels':first[:,None].clone(),'goal':first[:,None].clone(),'action':aa[:,:1,None].transpose(1,2)}
        cost=model.get_cost(info,aa[:,None].clone());goal=model.encode({'pixels':first.clone()})['emb'][:,-1]
        reference=(online[:,-1]-goal).square().sum(-1)[:,None];torch.testing.assert_close(cost,reference,**tol)
        checks['official_terminal_sum_squared_cost_max_abs']=float((cost-reference).abs().max())
    after={k:m.tensor_sha256(v) for k,v in model.state_dict().items()}
    if before!=after:raise RuntimeError('Zero-update TECH check modified official weights/buffers')
    result={'status':'PASS','task':task,'device':device,'actual_author_model_identity':model.r3_identity,'case_id':case['case_id'],
        'contract':model.r3_contract,'checks':checks,'all_model_state_unchanged':True,'optimizer_updates':0,
        'R3_tolerances_sha256':core.sha(tolerance_path),'applied_tolerance':tol,'scope':'REAL_TECH_INPUT_NATIVE_VS_CACHE_SEMANTICS'}
    core.freeze(core.ROOT/'technical_model/MODEL_CHECK.json',result);return result
