"""R5-only frozen encoder cache; original R3 trees are read-only."""
import argparse, os, sys, time
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np
from common import atomic,checked,file,fold,now,npz,read,sha,task_config

def run(a):
    import torch
    os.environ['R3_ROOT']=a.r3_root;sys.path.insert(0,a.r3_root)
    from r3.data import RawH5,image_transform,action_processor
    from r3.model import load_official,frozen_hashes,assert_frozen
    from r3.common import fp32_policy
    torch.set_num_threads(2);torch.set_num_interop_threads(1);fp32_policy()
    cfg=task_config(a.task);cfg['root']=Path(a.r3_root);cfg['roles']=cfg['root']/'manifests'/f'{a.task}_data_roles.json'
    roles=read(cfg['roles']);source=read(cfg['root']/'manifests'/f'{a.task}_source_map.json')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    asset=next(iter(source['assets'].values()));h5=cfg['root']/asset['path']
    # Source SHA is verified once by prepare.py, not repeatedly by every worker.
    acceptance=read(out.parent.parent/'INPUT_AUDIT.json');assert acceptance['tasks'][a.task]['source_h5']['sha256']==asset['sha256']
    assert h5.stat().st_size==asset['bytes']
    model=load_official(a.task,'cuda');before=frozen_hashes(model);versions=tuple(t._version for t in list(model.parameters())+list(model.buffers()))
    transform=image_transform();processor=action_processor(a.task)
    identity={'task':a.task,'roles_sha256':sha(cfg['roles']),'source_h5':asset,'model':model.r3_identity,'frozen':before,'source':file(__file__),'protocol':file(Path(__file__).with_name('protocol.json')),'batch':128,'shard':a.shard,'shards':2,'precision':fp32_policy(),'optimizer_updates':0,'state_labels_read':0}
    episodes=[e for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN','TECH','EVAL')][a.shard::2]
    records={};t=time.monotonic()
    with RawH5(h5,keys=['pixels','action']) as reader:
        for i,e in enumerate(episodes):
            name=__import__('hashlib').sha256(e['episode_id'].encode()).hexdigest();p=out/(name+'.npz');receipt=out/(name+'.json')
            if receipt.exists():
                d=read(receipt);assert d['identity']==identity and d['episode']==e;checked(d['file']);records[e['episode_id']]=d['file'];continue
            idx=e['source_episode_idx'];n=e['length'];assert reader.lengths[idx]==n
            actions=processor.transform(reader.array(idx,'action')).astype(np.float32);z=[]
            with torch.inference_mode():
                for start in range(0,n,128):
                    px=reader.array(idx,'pixels',start,min(n,start+128));x=torch.from_numpy(px)
                    if px.shape[-1]==3:x=x.permute(0,3,1,2)
                    z.append(model.encode({'pixels':transform(x).to('cuda')[:,None]})['emb'][:,0].cpu().numpy())
            z=np.concatenate(z).astype(np.float32);assert z.shape==(n,192) and np.isfinite(z).all()
            assert versions==tuple(t._version for t in list(model.parameters())+list(model.buffers()))
            assert not any(p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules())
            npz(p,z=z,actions=actions,raw_indices=np.arange(n),stride=np.array(5));rec={**file(p),'role':e['role'],'length':n}
            atomic(receipt,{'identity':identity,'episode':e,'file':rec});records[e['episode_id']]=rec
            if (i+1)%100==0:
                atomic(out/f'progress_{a.shard}.json',{'completed':i+1,'total':len(episodes),'seconds':time.monotonic()-t,'pid':os.getpid(),'utc':now()})
                print(a.task,a.shard,i+1,len(episodes),round(time.monotonic()-t,1),flush=True)
    assert_frozen(model,before)
    atomic(out/f'part_{a.shard}.json',{'status':'COMPLETE','identity':identity,'episodes':records,'optimizer_updates':0,'seconds':time.monotonic()-t})

def merge(a):
    out=Path(a.output);parts=[read(out/f'part_{s}.json') for s in range(2)];records={}
    for p in parts:
        for k,v in p['episodes'].items():
            assert k not in records;checked(v);records[k]=v
    roles=read(Path(a.r3_root)/'manifests'/f'{a.task}_data_roles.json');expected={e['episode_id'] for e in roles['episodes'] if e['role'] in ('REFIT_TRAIN','TECH','EVAL')};assert set(records)==expected
    atomic(out/'manifest.json',{'status':'FROZEN_OBSERVED_CACHE_COMPLETE','task':a.task,'episodes':records,'parts':[file(out/f'part_{s}.json') for s in range(2)],'roles_sha256':sha(Path(a.r3_root)/'manifests'/f'{a.task}_data_roles.json'),'optimizer_updates':0,'state_labels_read':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['pusht','reacher']);p.add_argument('--shard',type=int,choices=[0,1]);p.add_argument('--output',required=True);p.add_argument('--r3-root',default='/workspace/shared_data/r3');p.add_argument('--merge',action='store_true');a=p.parse_args()
    merge(a) if a.merge else run(a)
