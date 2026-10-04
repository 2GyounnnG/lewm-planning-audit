"""Fixed 64-entry menus locked before any privileged simulator scoring."""
from __future__ import annotations
import hashlib
from .common import ARMS,digest

def hash_indices(task,case_id,fork,source,count,total=300):
    return sorted(range(total),key=lambda i:(digest(['R4_MENU_SAMPLE_V1',task,case_id,fork,source,i]),i))[:count]

def initial_candidates(task,case_id,fork,device='cpu'):
    import torch
    seed=int.from_bytes(hashlib.sha256(f'R4_INITIAL_MENU_V1/{task}/{case_id}/{fork}'.encode()).digest()[:8],'big')
    generator=torch.Generator(device=device).manual_seed(seed)
    return torch.randn(16,5,10,generator=generator,device=device,dtype=torch.float32).cpu().numpy()

def build_menu(task,case_id,fork,proposals,action_processor,*,kind='S2',device='cpu'):
    import numpy as np
    if kind not in ('S2','S3'):raise ValueError('Explicit S2 four-model or S3 H0-only menu')
    sources=ARMS if kind=='S2' else ('H0',);extra=11 if kind=='S2' else 47
    plans=[];metadata=[]
    for source in sources:
        proposal=proposals[source]
        if proposal['returned'].shape!=(5,10) or proposal['last_generation'].shape!=(300,5,10):raise ValueError('Exact original 300-candidate5x10 plans required')
        plans.append(proposal['returned'].copy());metadata.append({'id':f'{source}/RETURNED','source':source,'kind':'RETURNED_CEM_ELITE_MEAN','sample_index':None})
        for i in hash_indices(task,case_id,fork,source,extra):
            plans.append(proposal['last_generation'][i].copy());metadata.append({'id':f'{source}/SAMPLE_{i:03d}','source':source,'kind':'HASH_SELECTED_LAST_GENERATION','sample_index':i})
    for i,a in enumerate(initial_candidates(task,case_id,fork,device)):
        plans.append(a);metadata.append({'id':f'INITIAL/{i:02d}','source':'INITIAL_DISTRIBUTION','kind':'INITIAL_DISTRIBUTION','sample_index':i})
    normalized=np.asarray(plans,dtype=np.float32);raw=action_processor.inverse_transform(normalized.reshape(-1,2)).reshape(64,25,2)
    first={}
    for i,(a,row) in enumerate(zip(raw,metadata)):
        sha=hashlib.sha256(a.tobytes(order='C')).hexdigest();row.update(logical_index=i,raw_action_sha256=sha,alias_of=first.get(sha));first.setdefault(sha,i)
    if len(metadata)!=64:raise AssertionError('Exactly 64 logical candidates retained')
    return {'normalized':normalized,'raw':raw,'metadata':metadata,'menu_sha256':digest({'metadata':metadata,'normalized_sha256':hashlib.sha256(normalized.tobytes()).hexdigest()}),'kind':kind}
