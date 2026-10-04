"""Real TECH-cache CPU proof of full autoregressive gradient and frozen buffers."""
import os,sys,time,json
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,torch
from . import common
from .train_worker import WindowAdapter,WindowSampler,role_ids,lr_at
from r3.model import load_official,refit_mode,trainable_parameters,frozen_hashes,assert_frozen,cached_rollout,delta_state
from r3.data import admissible_windows

def main():
    common.source_hashes();common.fp32_policy();torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);common.seed_all(10103201)
    out=common.ROOT/'technical/cpu_semantics';out.mkdir(parents=True,exist_ok=True)
    if (out/'CPU_GRADIENT_CHECK.json').exists():raise RuntimeError('TECH already completed; no duplicate update')
    roles=common.read_json('manifests/reacher_data_roles.json');cachemanifest=common.read_json('/workspace/r5/H1b/cache/reacher/manifest.json');cache={};sources={}
    for e in roles['episodes']:
        if e['role']!='TECH':continue
        eid=e['episode_id'];r=cachemanifest['episodes'][eid];p=Path(r['path']);assert common.sha256(p)==r['sha256'] and p.stat().st_size==r['bytes']
        with np.load(p,allow_pickle=False) as f:
            assert set(f.files)=={'z','actions','raw_indices','stride'};x={k:f[k].copy() for k in f.files};x['stride']=int(x['stride'])
        x['legal_starts']=admissible_windows(x['actions'],len(x['z']),3,1,5);cache[eid]=x;sources[eid]=r
    m=load_official('reacher','cpu');frozen=frozen_hashes(m);refit_mode(m,True);before=delta_state(m);parameters=trainable_parameters(m)
    adapter=WindowAdapter(cache,m.r3_contract,allowed_ids=role_ids(roles,'TECH'),role='TECH');sampler=WindowSampler(adapter,10103201);_,cursor=sampler.next();cursor=cursor[:8];z,a=adapter.batch(cursor,'cpu',1)
    calls=[];original=m.predict
    def traced(z,act):
        r=original(z,act);calls.append({'context_length':z.shape[1],'requires_grad':z.requires_grad,'prediction':r});return r
    m.predict=traced
    t=time.monotonic();pred=cached_rollout(m,z[:,:1],a,3);loss=(pred-z[:,1:4]).square().mean();last=(pred[:,-1]-z[:,3]).square().mean()
    chain=torch.autograd.grad(last,[calls[0]['prediction'],calls[1]['prediction']],retain_graph=True)
    chain_norm=[float(g.norm()) for g in chain];assert all(np.isfinite(v) and v>0 for v in chain_norm)
    assert [c['context_length'] for c in calls]==[1,2,3] and [c['requires_grad'] for c in calls]==[False,True,True]
    optimizer=torch.optim.AdamW(parameters,lr=lr_at(1),betas=(.9,.999),eps=1e-8,weight_decay=1e-3,foreach=False)
    loss.backward();grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters() if p.requires_grad}
    assert all(v is not None and np.isfinite(v) for v in grads.values());assert sum(v>0 for v in grads.values())>0
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith(('predictor.','pred_proj.')))
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));common.reserve_technical('cpu_semantics_update1','H3X_reacher_s103201',1)
    common.atomic_json(out/'OPTIMIZER_INFLIGHT.json',{'update':1,'classification':'TECHNICAL','formal_initialization_inherits_updates':False})
    optimizer.step();after=assert_frozen(m,frozen);(out/'OPTIMIZER_INFLIGHT.json').unlink();common.finish_technical('cpu_semantics_update1',1,{'status':'COMPLETE','device':'cpu','formal_initialization_inherits_updates':False})
    changed=[n for n,p in m.named_parameters() if n in before and not torch.equal(before[n],p.detach())];assert changed
    np.savez_compressed(out/'CPU_TECH_RAW.npz',z=z.numpy(),actions=a.numpy(),prediction=pred.detach().numpy(),target=z[:,1:4].numpy(),last_horizon_grad_to_step1=chain[0].detach().numpy(),last_horizon_grad_to_step2=chain[1].detach().numpy())
    common.atomic_json(out/'CPU_GRADIENT_CHECK.json',{'status':'PASS','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','source':common.source_hashes(),'model_identity':m.r3_identity,'cache_sources':sources,'cursor':cursor,'technical_batch_size':8,'formal_batch_size':128,'loss_raw_MSE':float(loss.detach()),'last_horizon_gradient_to_steps1_2_norms':chain_norm,'context_lengths':[1,2,3],'context_requires_grad':[False,True,True],'trainable_gradient_norms':grads,'preclip_gradient_norm':norm,'changed_trainable_parameters':changed,'frozen_before':frozen,'frozen_after':after,'frozen_parameters_and_all_buffers_equal':frozen==after,'actual_optimizer_updates':1,'state_labels_read':0,'new_encoder_updates':0,'formal_initialization_inherits_technical_updates':False,'seconds':time.monotonic()-t,'raw_sha256':common.sha256(out/'CPU_TECH_RAW.npz')})
    print(json.dumps({'status':'PASS','loss':float(loss.detach()),'chain_norms':chain_norm,'seconds':time.monotonic()-t}),flush=True)
if __name__=='__main__':main()
