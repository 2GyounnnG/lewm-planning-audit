"""Exact-budget frozen-official-predictor worker; all data roles are explicit.

Importing this module starts nothing. Scientific/technical execution is gated by
common authorization and ledgers. Checkpoints contain no encoder/projector copy.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, contextlib, copy, fcntl, hashlib, json, math, random, re, time
from pathlib import Path
import numpy as np
import torch
from . import common
from .model import (load_official,refit_mode,trainable_parameters,whitelist_manifest,
                    frozen_hashes,assert_frozen,cached_predict,cached_rollout,
                    delta_payload,apply_delta,tensor_sha256)

ROOT=common.ROOT
VERSION='R3_FROZEN_OFFICIAL_PREDICTOR_V1'
MILESTONES=(3000,10000,30000)
BATCH=128

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

def append_json(path,record):
    with Path(path).open('a') as f:f.write(json.dumps(clean(record),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def read_journal(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def atomic_torch(path,payload):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def save_checkpoint_files(folder,payload,technical):
    """One full resumable state; formal public checkpoints contain deltas only."""
    folder=Path(folder);current=int(payload['step'])
    compact_keys=('version','step','identity_sha256','delta','frozen','sampling_summary')
    if not technical:
        compact={key:payload[key] for key in compact_keys}
        compact_name=f'checkpoint_{current}.pt' if current in MILESTONES else 'last_delta.pt'
        public=folder/compact_name;atomic_torch(public,compact)
        common.atomic_json(folder/'last.json',{'path':compact_name,'sha256':common.sha256(public),
            'step':current,'kind':'alias_to_compact_predictor_delta','no_encoder_copy':True})
    resume=folder/'resume.pt';atomic_torch(resume,payload)
    pointer={'path':'resume.pt','sha256':common.sha256(resume),'step':current,
        'kind':'full_resumable_training_state','no_encoder_copy':True}
    common.atomic_json(folder/('last.json' if technical else 'resume.json'),pointer)
    if not technical and current in MILESTONES:
        # This file is our rolling public last delta, superseded by the immutable
        # milestone to which last.json now points; no requested milestone is removed.
        rolling=folder/'last_delta.pt'
        if rolling.exists():rolling.unlink()
    return pointer

def lr_at(step):
    if not 1<=step<=30000:raise ValueError('LR index outside fixed formal horizon')
    return 5e-5*step/500 if step<=500 else 5e-6+.5*(5e-5-5e-6)*(1+math.cos(math.pi*(step-500)/(30000-500)))

def scheduler_record(step):
    return {'kind':'linear_warmup500_cosine30000','completed_updates':step,
            'lr_last':lr_at(step) if step else None,'lr_next':lr_at(step+1) if step<30000 else None,
            'start_lr':5e-5,'end_lr':5e-6,'warmup_updates':500,'formal_horizon':30000}

class WindowAdapter:
    """Raw-frame caches; official span20 windows, with all raw start phases.

    Every cache entry is input-only. Legal starts are supplied by the frozen data
    implementation. Sampling is uniform over windows, not over episodes.
    """
    def __init__(self,cache,contract,*,allowed_ids,role):
        if role not in ('REFIT_TRAIN','TECH','MONITOR'):raise PermissionError('A holdout role cannot enter training/monitor adapter')
        self.cache=cache;self.ids=sorted(cache);self.role=role
        if self.ids!=sorted(allowed_ids) or not self.ids:raise PermissionError('Cache episodes do not exactly match the frozen role')
        self.L=int(contract['history_size']);self.D=int(contract['latent_dim']);self.A=int(contract['macro_action_dim'])
        self.stride=None;self.starts=[];raw_lengths=[]
        for ep in self.ids:
            x=cache[ep]
            if any(k in x for k in ('state','states','reward','rewards','goal','goal_state','qpos','observation','proprio')):
                raise PermissionError('State/reward/goal fields cannot enter predictor cache')
            required={'z','actions','raw_indices','stride','legal_starts'}
            if not required.issubset(x):raise ValueError('Input-only raw cache contract missing fields')
            z=np.asarray(x['z']);a=np.asarray(x['actions']);raw=np.asarray(x['raw_indices']);starts=np.asarray(x['legal_starts'])
            stride=int(x['stride']);self.stride=stride if self.stride is None else self.stride
            if stride!=self.stride or stride<=0 or self.A%stride:raise ValueError('Inconsistent raw action/stride contract')
            if z.ndim!=2 or z.shape[1]!=self.D or z.dtype!=np.float32 or a.shape!=(len(z),self.A//stride) or a.dtype!=np.float32:
                raise ValueError('Raw cache dtype/dimension differs from official config')
            if raw.shape!=(len(z),) or not np.issubdtype(raw.dtype,np.integer) or (len(raw)>1 and not np.all(np.diff(raw)==1)):
                raise ValueError('Contiguous exact raw frame identities required')
            if starts.ndim!=1 or not np.issubdtype(starts.dtype,np.integer) or len(np.unique(starts))!=len(starts):raise ValueError('Unique explicit raw legal starts required')
            span=(self.L+1)*stride
            if np.any(starts<0) or np.any(starts+span>len(z)):raise ValueError('Legal start violates conservative official window span')
            if not np.isfinite(z).all():raise ValueError('Nonfinite frozen observed latent')
            # Only declared legal transitions are checked; source boundary action
            # NaNs outside those windows are not silently altered or sampled.
            for start in starts:
                if not np.isfinite(a[start:start+self.L*stride]).all():raise ValueError('Legal window contains nonfinite action')
            self.starts.append(starts.astype(np.int64,copy=False));raw_lengths.append(len(z))
        self.window_offsets=np.r_[0,np.cumsum([len(x) for x in self.starts])].astype(np.int64)
        self.raw_offsets=np.r_[0,np.cumsum(raw_lengths)].astype(np.int64)
        if not self.window_offsets[-1]:raise ValueError('No authorized legal prediction windows')
    def decode(self,indices):
        indices=np.asarray(indices,dtype=np.int64)
        if np.any(indices<0) or np.any(indices>=self.window_offsets[-1]):raise IndexError('Window index outside explicit legal set')
        episodes=np.searchsorted(self.window_offsets[1:],indices,side='right')
        starts=np.array([self.starts[e][i-self.window_offsets[e]] for i,e in zip(indices,episodes)],dtype=np.int64)
        return episodes,starts
    def cursor(self,indices):
        e,s=self.decode(indices)
        return [[self.ids[int(i)],int(start)] for i,start in zip(e,s)]
    def batch(self,cursor,device,horizon=1):
        z=[];a=[]
        if horizon<1:raise ValueError('Positive forecast horizon required')
        for ep,start in cursor:
            if ep not in self.cache:raise PermissionError('Cursor episode outside authorized data role')
            item=self.cache[ep];start=int(start);required_span=(self.L+horizon)*self.stride
            if start<0 or start+required_span>len(item['z']):raise IndexError('Window violates conservative recorded clip span')
            iz=start+np.arange(self.L+horizon)*self.stride
            window=np.asarray(item['z'])[iz];actions=np.asarray(item['actions'])[start:start+(self.L+horizon-1)*self.stride]
            if not np.isfinite(actions).all():raise ValueError('Requested diagnostic window contains nonfinite actions')
            z.append(window);a.append(actions.reshape(self.L+horizon-1,self.A))
        return torch.as_tensor(np.stack(z),device=device,dtype=torch.float32),torch.as_tensor(np.stack(a),device=device,dtype=torch.float32)

class WindowSampler:
    def __init__(self,adapter,seed):
        self.adapter=adapter;self.rng=np.random.default_rng(int(seed))
        self.window_counts=np.zeros(int(adapter.window_offsets[-1]),dtype=np.int64)
        self.episode_counts=np.zeros(len(adapter.ids),dtype=np.int64)
        self.raw_action_counts=np.zeros(int(adapter.raw_offsets[-1]),dtype=np.int64)
        self.macro_transition_counts=np.zeros_like(self.raw_action_counts)
    def next(self,advance=True):
        rng=self.rng
        if not advance:rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(self.rng.bit_generator.state)
        index=rng.integers(int(self.adapter.window_offsets[-1]),size=BATCH,dtype=np.int64)
        return index,self.adapter.cursor(index)
    def count(self,index):
        e,s=self.adapter.decode(index);origin=self.adapter.raw_offsets[e]+s
        np.add.at(self.window_counts,index,1);np.add.at(self.episode_counts,e,1)
        np.add.at(self.raw_action_counts,(origin[:,None]+np.arange(self.adapter.L*self.adapter.stride)).reshape(-1),1)
        np.add.at(self.macro_transition_counts,(origin[:,None]+np.arange(self.adapter.L)*self.adapter.stride).reshape(-1),1)
    def state(self):
        return {'rng':copy.deepcopy(self.rng.bit_generator.state),'window_counts':torch.from_numpy(self.window_counts.copy()),
                'episode_counts':torch.from_numpy(self.episode_counts.copy()),'raw_action_counts':torch.from_numpy(self.raw_action_counts.copy()),
                'macro_transition_counts':torch.from_numpy(self.macro_transition_counts.copy())}
    def restore(self,state):
        self.rng.bit_generator.state=state['rng']
        for key in ('window_counts','episode_counts','raw_action_counts','macro_transition_counts'):
            value=state[key].cpu().numpy()
            if value.shape!=getattr(self,key).shape or np.any(value<0):raise ValueError('Sampling coverage shape/count mismatch')
            setattr(self,key,value.copy())
    def summary(self):
        return {'sampled_windows':int(self.window_counts.sum()),'sampled_episode_draws':int(self.episode_counts.sum()),
            'predicted_target_tokens':int(self.macro_transition_counts.sum()),'raw_action_exposures':int(self.raw_action_counts.sum()),
            'unique_windows':int(np.count_nonzero(self.window_counts)),'unique_episodes':int(np.count_nonzero(self.episode_counts)),
            'unique_raw_actions':int(np.count_nonzero(self.raw_action_counts)),
            'unique_macro_transitions':int(np.count_nonzero(self.macro_transition_counts))}
    def save_counts(self,path):
        path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,episode_ids=np.asarray(self.adapter.ids),window_offsets=self.adapter.window_offsets,
                legal_starts=np.concatenate(self.adapter.starts),raw_offsets=self.adapter.raw_offsets,
                raw_indices=np.concatenate([self.adapter.cache[e]['raw_indices'] for e in self.adapter.ids]),
                window_counts=self.window_counts,episode_counts=self.episode_counts,
                raw_action_counts=self.raw_action_counts,macro_transition_counts=self.macro_transition_counts)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)

def role_ids(roles,role):return sorted(x['episode_id'] for x in roles['episodes'] if x['role']==role)

def monitor(model,adapter,cursors,device,microbatch):
    was_training=any(p.requires_grad for p in model.parameters());model.eval();one=[];free=[];last=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(cursors),microbatch):
            z,a=adapter.batch(cursors[i:i+microbatch],device,horizon=5);L=adapter.L
            tf=cached_predict(model,z[:,:L],a[:,:L]);pred=cached_rollout(model,z[:,:L],a,5)
            one.extend((tf-z[:,1:L+1]).square().mean((1,2)).cpu().tolist())
            free.extend((pred-z[:,L:]).square().mean((1,2)).cpu().tolist())
            last.extend((pred[:,-1]-z[:,-1]).square().mean(1).cpu().tolist())
    refit_mode(model,was_training)
    return {'teacher_forced_one_step_raw_MSE':float(np.mean(one)),
            'free_H5_mean_raw_MSE':float(np.mean(free)),'free_H5_terminal_raw_MSE':float(np.mean(last)),
            'per_window':{'teacher_forced_one_step_raw_MSE':one,'free_H5_mean_raw_MSE':free,'free_H5_terminal_raw_MSE':last},
            'windows':len(cursors),'window_identity_sha256':digest(cursors),'seconds':time.perf_counter()-start,
            'state_labels_read':0,'checkpoint_selection':False,'diagnostic_only':True}

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'bit_generator':n[0],'keys':n[1].tolist(),'pos':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},
            'torch_cpu':torch.random.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n['bit_generator'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.random.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

def optimizer_to(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            # AdamW non-capturable step scalars remain on CPU, as initialized.
            if torch.is_tensor(value):state[key]=value.cpu() if key=='step' else value.to(device)

def _validate_job(job):
    if job['task'] not in common.TASKS or int(job['refit_seed']) not in common.SEEDS or int(job['updates'])!=30000:
        raise ValueError('Job is outside the fixed two-task six-job contract')
    if not re.fullmatch(r'R3_[A-Za-z0-9_]+',job['job_id']):raise ValueError('Unsafe job ID')

def train_job(job,device='cuda',microbatch=128,technical_steps=None,run_id=None,stop_after=None,slot=0):
    _validate_job(job);technical=technical_steps is not None
    if microbatch not in (32,64,128) or slot not in (0,1):raise ValueError('Only pre-frozen128/64/32 microbatch and up to2 workers/card')
    if not technical and not str(device).startswith('cuda'):raise ValueError('Formal R3 requires the authorized GPU instance')
    if technical and (not run_id or not re.fullmatch(r'[A-Za-z0-9_\-]+',run_id)):raise ValueError('Technical runs need a separate safe run ID')
    endpoint=int(technical_steps) if technical else 30000
    if (technical and not 1<=endpoint<=1024) or (not technical and endpoint!=30000):raise ValueError('Invalid endpoint')
    limit=endpoint if stop_after is None else int(stop_after)
    if not 1<=limit<=endpoint:raise ValueError('stop_after is an absolute step within this run')
    common.require_authorization(job,technical=technical);common.ensure_space()
    folder=ROOT/'artifacts'/('technical' if technical else 'train')
    folder=folder/run_id/job['job_id'] if technical else folder/job['job_id'];folder.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for path,kind in [(ROOT/'state/execution_phase.lock',fcntl.LOCK_SH),(folder/'worker.lock',fcntl.LOCK_EX)]:
            path.parent.mkdir(parents=True,exist_ok=True);handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,kind|fcntl.LOCK_NB)
        if str(device).startswith('cuda'):
            binding=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if not binding or ',' in binding:raise ValueError('Worker must bind exactly one authorized physical GPU')
            path=ROOT/'state/worker_locks'/f'gpu_{binding}_slot_{slot}.lock';path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(job,folder,device,microbatch,technical,run_id,endpoint,limit)

def _run_locked(job,folder,device,microbatch,technical,run_id,endpoint,limit):
    from .data import load_cache
    task=job['task'];common.fp32_policy();torch.use_deterministic_algorithms(True)
    if str(device).startswith('cuda'):
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.reset_peak_memory_stats()
    seed=int(job['refit_seed'])+(10000000 if technical else 0);common.seed_all(seed)
    model=load_official(task,device);frozen=frozen_hashes(model);refit_mode(model,True)
    initial_path=folder/'FROZEN_INITIAL.json'
    if initial_path.exists():
        if common.read_json(initial_path)!=frozen:raise RuntimeError('Initial frozen official state changed')
    else:common.atomic_json(initial_path,frozen)
    manifest=whitelist_manifest(model);parameters=trainable_parameters(model)
    roles=common.read_json(f'manifests/{task}_data_roles.json');role='TECH' if technical else 'REFIT_TRAIN'
    adapter=WindowAdapter(load_cache(task,role),model.r3_contract,allowed_ids=role_ids(roles,role),role=role)
    mrole='TECH' if technical else 'MONITOR'
    ma=adapter if technical else WindowAdapter(load_cache(task,mrole),model.r3_contract,allowed_ids=role_ids(roles,mrole),role=mrole)
    windows=roles['technical_monitor_windows' if technical else 'monitor_windows']
    if not windows or len(windows)>256 or len({(e,int(s)) for e,s in windows})!=len(windows):raise ValueError('Fixed distinct monitor window count must be1..256')
    if any(e not in role_ids(roles,mrole) for e,s in windows):raise PermissionError('Monitor includes a forbidden data role')
    for cursor in windows:ma.batch([cursor],'cpu',5)
    if not technical and set(adapter.ids)&set(ma.ids):raise PermissionError('REFIT_TRAIN/MONITOR overlap')
    sampler=WindowSampler(adapter,seed)
    identity={'version':VERSION,'job':job,'inputs':common.identity_hashes(job),
        'source':{n:common.sha256(ROOT/'r3'/n) for n in ('common.py','model.py','train_worker.py','data.py')},
        'base_identity':model.r3_identity,'contract':model.r3_contract,'batch_size':BATCH,'microbatch':microbatch,
        'technical':technical,'run_id':run_id,'endpoint':endpoint,'initial_seed':seed,
        'role':role,'episode_ids':adapter.ids,'legal_starts_sha256':digest({e:s.tolist() for e,s in zip(adapter.ids,adapter.starts)}),
        'monitor_role':mrole,'monitor_windows_sha256':digest(windows),'attention_backend':'math_deterministic',
        'precision':'FP32_NO_AMP_NO_TF32','sampler':'UNIFORM_ALL_EXPLICIT_LEGAL_RAW_WINDOWS'}
    identity_hash=digest(identity);saved_identity=folder/'RUN_IDENTITY.json'
    if saved_identity.exists():
        if common.read_json(saved_identity)!=identity:raise RuntimeError('Saved run identity differs; never reuse/refit silently')
    else:common.atomic_json(saved_identity,identity)
    p=folder/'PARAMETER_WHITELIST.json'
    if p.exists():
        if common.read_json(p)!=manifest:raise RuntimeError('Official initialization/whitelist changed')
    else:common.atomic_json(p,manifest)
    journal=read_journal(folder/'updates.jsonl');valid=read_journal(folder/'monitor.jsonl')
    if [r['step'] for r in journal]!=list(range(1,len(journal)+1)):raise RuntimeError('Noncontiguous successful optimizer journal')
    if len({r['step'] for r in valid})!=len(valid):raise RuntimeError('Duplicate monitor step')
    last=folder/'last.json';resume_pointer=folder/('last.json' if technical else 'resume.json');step=0;checkpoint=None
    if resume_pointer.exists():
        pointer=common.read_json(resume_pointer);path=folder/pointer['path']
        if path.parent!=folder or common.sha256(path)!=pointer['sha256']:raise RuntimeError('Resume checkpoint identity differs')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['identity_sha256']!=identity_hash or checkpoint['frozen']!=frozen:raise RuntimeError('Resume source/frozen identity mismatch')
        step=int(checkpoint['step'])
        if checkpoint['scheduler']!=scheduler_record(step):raise RuntimeError('Resume learning-rate cursor differs')
        apply_delta(model,path);refit_mode(model,True);sampler.restore(checkpoint['sampler'])
    if len(journal)!=step:raise RuntimeError('Uncheckpointed or missing optimizer journal suffix; preserve, do not replay updates')
    in_flight_path=folder/'OPTIMIZER_INFLIGHT.json'
    if in_flight_path.exists():
        raise RuntimeError('Unresolved optimizer call intent; possible partial step must not be replayed')
    for failure_path in folder.glob('failure_*.json'):
        failed=common.read_json(failure_path)
        if failed.get('uncertain_optimizer_attempt') or failed.get('successful_updates',0)>step:
            raise RuntimeError('A failed attempt reports optimizer updates beyond the resume checkpoint; preserve without replay')
    if checkpoint and checkpoint['next_batch_sha256']!=digest(sampler.next(advance=False)[1]):raise RuntimeError('Resume next sampling cursor differs')
    if sampler.summary()['sampled_windows']!=step*BATCH:raise RuntimeError('Sampling counts disagree with successful updates')
    if step>limit:raise ValueError('Saved step exceeds requested endpoint')
    if step==endpoint:
        result=common.read_json(folder/'result.json')
        if result['identity_sha256']!=identity_hash or result['actual_updates']!=endpoint:raise RuntimeError('Completed result differs')
        if not technical:
            public=common.read_json(last);public_path=folder/public['path']
            if public['path']!='checkpoint_30000.pt' or public_path.parent!=folder or common.sha256(public_path)!=public['sha256'] or result['last_checkpoint']!=public:
                raise RuntimeError('Completed compact primary checkpoint differs')
        assert_frozen(model,frozen)
        if not technical and not (ROOT/'state/recovery_queue'/(job['job_id']+'.json')).exists():common.publish_job(folder,job)
        return result
    if step==limit:raise ValueError('Requested stop already reached; resume with a higher absolute stop')
    segment_start=step;segment=f'{run_id}/{job["job_id"]}/steps_{step}_{limit}' if technical else None
    if technical:common.reserve_technical(segment,job['job_id'],limit-step)
    optimizer=None;started=time.perf_counter();compute_seconds=0.;loader_seconds=0.;save_seconds=0.;monitor_seconds=0.
    actual=step;failure=None;step_in_flight=None;status='FAILED';timing=copy.deepcopy(checkpoint['timing']) if checkpoint else {'worker_seconds':0.,'compute_seconds':0.,'loader_seconds':0.,'checkpoint_seconds':0.,'monitor_seconds':0.}
    def synchronize():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    def observe(current):
        nonlocal monitor_seconds
        record=monitor(model,ma,windows,device,microbatch);record.update(step=current,role=mrole,classification='TECHNICAL_DIAGNOSTIC' if technical else 'REFIT_HELDOUT_MONITOR_NOT_SELECTION')
        append_json(folder/'monitor.jsonl',record);monitor_seconds+=record['seconds'];assert_frozen(model,frozen)
    def save(current):
        nonlocal save_seconds
        common.ensure_space();t=time.perf_counter();assert_frozen(model,frozen)
        if common.identity_hashes(job)!=identity['inputs'] or any(common.sha256(ROOT/'r3'/n)!=d for n,d in identity['source'].items()):
            raise RuntimeError('Execution source/config/cache identity changed during training')
        payload={'version':VERSION,'step':current,'identity_sha256':identity_hash,'delta':delta_payload(model),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler_record(current),'rng':rng_state(),'sampler':sampler.state(),
            'next_batch_sha256':digest(sampler.next(advance=False)[1]),'frozen':frozen,'sampling_summary':sampler.summary(),
            'timing':{k:timing[k]+v for k,v in [('worker_seconds',time.perf_counter()-started),('compute_seconds',compute_seconds),
                      ('loader_seconds',loader_seconds),('checkpoint_seconds',save_seconds),('monitor_seconds',monitor_seconds)]},
            'optimizer_state_is_new_post_refit':True,'new_encoder_updates':0,'state_labels_read':0,'pid':os.getpid()}
        save_checkpoint_files(folder,payload,technical)
        append_json(folder/'freeze_checks.jsonl',{'step':current,'frozen_sha256':frozen['sha256'],'passed':True,**sampler.summary()})
        save_seconds+=time.perf_counter()-t
    try:
        optimizer=torch.optim.AdamW(parameters,lr=lr_at(max(1,step)),betas=(.9,.999),eps=1e-8,weight_decay=1e-3,foreach=False)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer']);optimizer_to(optimizer,device);restore_rng(checkpoint['rng'])
        if not technical and step==0 and not valid:observe(0)
        for current in range(step+1,limit+1):
            common.ensure_space();start=time.perf_counter();indices,cursor=sampler.next()
            z,a=adapter.batch(cursor,device,1);synchronize();loader_seconds+=time.perf_counter()-start
            lr=lr_at(current)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);synchronize();start=time.perf_counter();loss_value=0.
            for begin in range(0,BATCH,microbatch):
                prediction=cached_predict(model,z[begin:begin+microbatch,:adapter.L],a[begin:begin+microbatch])
                loss=(prediction-z[begin:begin+microbatch,1:adapter.L+1]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; no successful optimizer step')
                (loss*(microbatch/BATCH)).backward();loss_value+=float(loss.detach())*(microbatch/BATCH)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(('predictor.','pred_proj.'))):raise AssertionError('Gradient escaped predictor whitelist')
            # A process loss or exception inside AdamW can leave a partial update.
            # Persist intent before entering it; never replay an ambiguous call.
            intent_start=time.perf_counter()
            common.atomic_json(in_flight_path,{'step':current,'identity_sha256':identity_hash,
                'batch_cursor_sha256':digest(cursor),'pid':os.getpid(),'optimizer_call_not_committed':True})
            intent_seconds=time.perf_counter()-intent_start
            step_in_flight=current
            optimizer.step();synchronize();actual=current;step_in_flight=None;sampler.count(indices)
            duration=time.perf_counter()-start-intent_seconds;compute_seconds+=duration
            append_json(folder/'updates.jsonl',{'step':current,'loss_raw_MSE':loss_value,'lr':lr,
                'preclip_gradient_norm':float(norm),'batch_cursor_sha256':digest(cursor),'sampled_windows':BATCH,
                'predicted_tokens':BATCH*adapter.L,'raw_action_exposures':BATCH*adapter.L*adapter.stride,
                'compute_seconds':duration,'optimizer_intent_commit_seconds':intent_seconds,'pid':os.getpid(),'technical':technical})
            in_flight_path.unlink()
            if (not technical and current%1000==0) or (technical and current==endpoint):observe(current)
            if current%100==0 or current in MILESTONES or current==limit:save(current)
        status='TECHNICAL_COMPLETE' if technical and actual==endpoint else 'REFIT_TRAINING_COMPLETE_UNSCORED' if actual==30000 else 'PAUSED_WITH_RESUMABLE_CHECKPOINT'
        sampler.save_counts(folder/'sampling_counts.npz')
        result={'version':VERSION,'status':status,'job_id':job['job_id'],'task':task,'actual_updates':actual,
            'identity_sha256':identity_hash,'base_identity':model.r3_identity,'frozen_before':frozen['sha256'],
            'frozen_after':assert_frozen(model,frozen)['sha256'],'sampling':sampler.summary(),
            'trainable_parameters':manifest['trainable_parameters'],'frozen_parameters':manifest['frozen_parameters'],
            'new_encoder_updates':0,'state_labels_read':0,'checkpoint_selection':False,'final_primary_step':30000,
            'sampling_counts_file':'sampling_counts.npz','last_checkpoint':common.read_json(last),
            'resume_checkpoint':common.read_json(resume_pointer),'formal_milestones_are_compact_parameter_deltas':not technical,
            'microbatch':microbatch,'effective_batch':BATCH,'precision':'FP32_NO_AMP_NO_TF32','technical':technical,
            'timing':{k:timing[k]+v for k,v in [('worker_seconds',time.perf_counter()-started),('compute_seconds',compute_seconds),
                      ('loader_seconds',loader_seconds),('checkpoint_seconds',save_seconds),('monitor_seconds',monitor_seconds)]},
            'peak_allocated_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,
            'device':str(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES',''),
            'formal_initialization_inherits_technical_updates':False}
        common.atomic_json(folder/'result.json',clean(result))
        if not technical and actual==30000:common.publish_job(folder,job)
        return result
    except BaseException as error:
        failure={'type':type(error).__name__,'message':str(error),'successful_updates':actual,
                 'checkpoint_updates':common.read_json(resume_pointer)['step'] if resume_pointer.exists() else 0,'no_uncheckpointed_suffix_replay':True,
                 'uncertain_optimizer_attempt':step_in_flight is not None,'attempted_step':step_in_flight,
                 'optimizer_updates_lower_bound':actual,'optimizer_updates_upper_bound':actual+int(step_in_flight is not None)}
        common.atomic_json(folder/f'failure_{os.getpid()}.json',failure)
        raise
    finally:
        if technical and step_in_flight is None:
            common.finish_technical(segment,int(actual-segment_start),{'run_id':run_id,'job_id':job['job_id'],
                'segment_start':segment_start,'segment_end':actual,'status':status,'failure':failure,'optimizer_updates_inherit_into_formal':False})
        elif technical:
            # None preserves the entire reservation rather than falsely freeing
            # budget after an optimizer call that may have partially executed.
            common.finish_technical(segment,None,{'run_id':run_id,'job_id':job['job_id'],
                'segment_start':segment_start,'status':'UNCERTAIN_OPTIMIZER_ATTEMPT',
                'known_successful_updates':int(actual-segment_start),
                'attempt_upper_bound':int(actual-segment_start+1),'failure':failure,
                'preserve_full_reserved_budget':True,'optimizer_updates_inherit_into_formal':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--device',default='cuda')
    parser.add_argument('--microbatch',type=int,default=128);parser.add_argument('--technical-steps',type=int)
    parser.add_argument('--run-id');parser.add_argument('--stop-after',type=int);parser.add_argument('--slot',type=int,default=0)
    args=parser.parse_args();matching=[j for j in common.jobs() if j['job_id']==args.job]
    if len(matching)!=1:raise ValueError('Exactly one frozen job ID required')
    print(json.dumps(clean(train_job(matching[0],args.device,args.microbatch,args.technical_steps,args.run_id,args.stop_after,args.slot))))
if __name__=='__main__':main()
