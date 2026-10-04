"""Three fixed 30k predictor refits using the exact R3 batch/loss/optimizer recipe."""
from __future__ import annotations
import contextlib, fcntl, os, random, time
import numpy as np
from . import core, data

def train(task,seed,device='cuda',microbatch=128,technical_steps=None,stop_after=None,slot=0):
    import torch
    if task not in core.TASKS or seed not in core.SEEDS or microbatch not in (32,64,128) or slot not in (0,1):raise ValueError('Outside fixed X1 matrix')
    technical=technical_steps is not None;endpoint=int(technical_steps) if technical else 30000
    if technical and not 1<=endpoint<=1024:raise ValueError('TECH cap exceeded')
    if not technical and not str(device).startswith('cuda'):raise ValueError('Formal X1 requires its assigned GPU')
    limit=endpoint if stop_after is None else int(stop_after)
    if not 1<=limit<=endpoint:raise ValueError('Invalid fixed stop cursor')
    folder=core.ROOT/('technical_train' if technical else 'train')/str(seed);folder.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        h=stack.enter_context((folder/'worker.lock').open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if str(device).startswith('cuda'):
            binding=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if not binding or ',' in binding:raise ValueError('One physical GPU per worker required')
            permitted={'4','5'} if task=='tworoom' else {'6','7'}
            if binding not in permitted:raise ValueError('X1 worker outside task GPU partition')
            p=core.ROOT/'state'/f'gpu_{binding}_slot_{slot}.lock';p.parent.mkdir(parents=True,exist_ok=True)
            h=stack.enter_context(p.open('a+'));fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(task,seed,device,microbatch,endpoint,limit,technical,folder)

def _train(task,seed,device,microbatch,endpoint,limit,technical,folder):
    import torch
    core.policy();m=core.r3('model');w=core.r3('train_worker');actual_seed=seed+(10000000 if technical else 0)
    random.seed(actual_seed);np.random.seed(actual_seed);torch.manual_seed(actual_seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(actual_seed)
    model=core.load_model(task,device);frozen=m.frozen_hashes(model);m.refit_mode(model,True);params=m.trainable_parameters(model)
    check=core.read(core.ROOT/'technical_model/MODEL_CHECK.json')
    if check['status']!='PASS' or check['actual_author_model_identity']!=model.r3_identity or not check['all_model_state_unchanged']:
        raise RuntimeError('Real-author zero-update interface TECH check required')
    if not technical:
        tech=core.read(core.ROOT/'technical_train/103201/result.json')
        if tech['status']!='TECHNICAL_COMPLETE' or not tech['technical'] or tech['actual_updates']!=128 or tech['frozen_before']!=tech['frozen_after'] or tech['frozen_before']!=frozen['sha256']:
            raise RuntimeError('Separate fixed128-step TECH refit must pass before formal training')
    roles=core.read(core.ROOT/'manifests'/f'{task}_data_roles.json');role='TECH' if technical else 'REFIT_TRAIN';mr='TECH' if technical else 'MONITOR'
    ids=lambda r:[e['episode_id'] for e in roles['episodes'] if e['role']==r]
    adapter=w.WindowAdapter(data.load_cache(task,role),model.r3_contract,allowed_ids=ids(role),role=role)
    monitor=adapter if technical else w.WindowAdapter(data.load_cache(task,mr),model.r3_contract,allowed_ids=ids(mr),role=mr)
    windows=roles['technical_monitor_windows' if technical else 'monitor_windows'];sampler=w.WindowSampler(adapter,actual_seed)
    if not windows or len(windows)>256:raise ValueError('Missing fixed monitor windows')
    for cursor in windows:monitor.batch([cursor],'cpu',5)
    identity={'task':task,'seed':seed,'initial_seed':actual_seed,'technical':technical,'endpoint':endpoint,'role':role,
        'batch':128,'microbatch':microbatch,'model':model.r3_identity,'contract':model.r3_contract,'frozen':frozen['sha256'],
        'roles_sha256':core.sha(core.ROOT/'manifests'/f'{task}_data_roles.json'),
        'cache_sha256':core.sha(core.ROOT/'manifests'/f'{task}_cache.json'),
        'x1_source':{n:core.sha(core.Path(__file__).parent/n) for n in ('core.py','data.py','train.py')},
        'r3_reused_source':{n:core.sha(core.R3_ROOT/'r3'/n) for n in ('model.py','train_worker.py')},
        'optimizer':'R3 AdamW/lr_at unchanged','precision':'FP32_NO_AMP_NO_TF32'}
    core.freeze(folder/'RUN_IDENTITY.json',identity);identity_sha=core.digest(identity);core.freeze(folder/'PARAMETER_WHITELIST.json',m.whitelist_manifest(model))
    cp=None;step=0;pointer=folder/'resume.json';journal=folder/'updates.jsonl'
    if pointer.exists():
        cp=torch.load(core.verify(core.read(pointer)),map_location='cpu',weights_only=True)
        if cp['identity_sha256']!=identity_sha or cp['frozen']!=frozen:raise RuntimeError('Resume identity differs')
        step=int(cp['step']);m.apply_delta(model,folder/'resume.pt');m.refit_mode(model,True);sampler.restore(cp['sampler'])
        if cp['scheduler']!=w.scheduler_record(step):raise RuntimeError('Resume LR differs')
    updates=w.read_journal(journal)
    if [r['step'] for r in updates]!=list(range(1,step+1)) or (folder/'OPTIMIZER_INFLIGHT.json').exists():
        raise RuntimeError('Uncheckpointed/ambiguous optimizer suffix retained; no silent repeated updates')
    if sampler.summary()['sampled_windows']!=128*step:raise RuntimeError('Sampler cursor mismatch')
    if step==endpoint:
        result=core.read(folder/'result.json')
        if result['identity_sha256']!=identity_sha:raise RuntimeError('Result identity differs')
        return result
    if step>=limit:raise ValueError('Stop cursor already reached')
    optimizer=torch.optim.AdamW(params,lr=w.lr_at(max(1,step)),betas=(.9,.999),eps=1e-8,weight_decay=1e-3,foreach=False)
    if cp:optimizer.load_state_dict(cp['optimizer']);w.optimizer_to(optimizer,device);w.restore_rng(cp['rng'])
    began=time.monotonic();compute=0.;previous_seconds=float(cp.get('seconds',0)) if cp else 0.
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    def observe(current):
        record=w.monitor(model,monitor,windows,device,microbatch);record.update(step=current,role=mr,checkpoint_selection=False)
        w.append_json(folder/'monitor.jsonl',record);m.assert_frozen(model,frozen)
    def save(current):
        m.assert_frozen(model,frozen)
        payload={'step':current,'identity_sha256':identity_sha,'delta':m.delta_payload(model),'frozen':frozen,
            'optimizer':optimizer.state_dict(),'scheduler':w.scheduler_record(current),'rng':w.rng_state(),'sampler':sampler.state(),
            'next_batch_sha256':core.digest(sampler.next(advance=False)[1]),'seconds':previous_seconds+time.monotonic()-began}
        w.atomic_torch(folder/'resume.pt',payload);core.atomic(pointer,core.file_record(folder/'resume.pt'))
        dest=folder/(f'checkpoint_{current}.pt' if current in (3000,10000,30000) or technical and current==endpoint else 'last_delta.pt')
        w.atomic_torch(dest,{k:payload[k] for k in ('step','identity_sha256','delta','frozen')})
        core.atomic(folder/'last.json',core.file_record(dest))
    if not updates:observe(0)
    for current in range(step+1,limit+1):
        index,cursor=sampler.next();z,a=adapter.batch(cursor,device,1);optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=w.lr_at(current)
        sync();start=time.monotonic();loss_value=0.
        for offset in range(0,128,microbatch):
            pred=m.cached_predict(model,z[offset:offset+microbatch,:adapter.L],a[offset:offset+microbatch])
            loss=(pred-z[offset:offset+microbatch,1:adapter.L+1]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refit loss')
            (loss*microbatch/128).backward();loss_value+=float(loss.detach())*microbatch/128
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(m.TRAINABLE_PREFIXES)):raise AssertionError('Gradient escaped whitelist')
        core.atomic(folder/'OPTIMIZER_INFLIGHT.json',{'step':current,'identity_sha256':identity_sha,'batch_sha256':core.digest(cursor)})
        optimizer.step();sync();sampler.count(index);elapsed=time.monotonic()-start;compute+=elapsed
        w.append_json(journal,{'step':current,'loss_raw_MSE':loss_value,'lr':w.lr_at(current),'preclip_gradient_norm':float(norm),
            'batch_cursor_sha256':core.digest(cursor),'sampled_windows':128,'compute_seconds':elapsed,'technical':technical})
        (folder/'OPTIMIZER_INFLIGHT.json').unlink()
        if current%1000==0 or technical and current==endpoint:observe(current)
        if current%100==0 or current==limit:save(current)
    sampler.save_counts(folder/'sampling_counts.npz')
    result={'status':'TECHNICAL_COMPLETE' if technical and limit==endpoint else 'REFIT_TRAINING_COMPLETE_UNSCORED' if limit==30000 else 'PAUSED',
        'task':task,'seed':seed,'technical':technical,'actual_updates':limit,'identity_sha256':identity_sha,
        'frozen_before':frozen['sha256'],'frozen_after':m.assert_frozen(model,frozen)['sha256'],'new_encoder_updates':0,
        'state_labels_read':0,'checkpoint_selection':False,'batch':128,'microbatch':microbatch,'sampling':sampler.summary(),
        'seconds':previous_seconds+time.monotonic()-began,'last_checkpoint':core.read(folder/'last.json') if (folder/'last.json').exists() else None}
    core.atomic(folder/'result.json',result);return result
