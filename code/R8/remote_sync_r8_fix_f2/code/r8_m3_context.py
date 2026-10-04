"""Fixed R8 M3 history constructions."""
import numpy as np
def history_indices(raw_count,mode='REAL3'):
    if raw_count==0:return [0]
    if raw_count not in (25,50,75):raise RuntimeError(f'Unexpected M3 replan anchor {raw_count}')
    if mode=='REPEAT3':return [raw_count,raw_count,raw_count]
    if mode in ('REAL2',):return [raw_count-5,raw_count]
    return [raw_count-10,raw_count-5,raw_count]
def real_history_observation(info,observations,raw_count,history_mode='REAL3'):
    # The official policy is queried at every raw step, while a new plan is
    # solved only at the registered 25-raw-step receding anchors.  Preserve
    # the single-frame observation between anchors.
    if raw_count==0 or raw_count not in (25,50,75):return info
    if len(observations)!=raw_count+1:raise RuntimeError('M3 history observation/raw alignment')
    out=dict(info);idx=history_indices(raw_count,history_mode)
    out['pixels']=np.stack([observations[i]['pixels'] for i in idx])[None]
    return out
def candidates_with_prefix(info,candidates,rows,action_processor,history_mode='REAL3'):
    import torch
    if info['pixels'].shape[2]==1:return candidates
    if history_mode=='REAL2':
        raw=np.stack([r['action'] for r in rows[-5:]])
        # One five-raw-step macro prefix: the official action encoder stores
        # it as one token with ten normalized scalar features.
        prefix=action_processor.transform(raw).reshape(1,1,1,10)
    elif history_mode=='REPEAT3':
        prefix=torch.zeros((1,1,2,10),device=candidates.device,dtype=candidates.dtype)
    else:
        raw=np.stack([r['action'] for r in rows[-10:]])
        prefix=action_processor.transform(raw).reshape(1,1,2,10)
    prefix=torch.as_tensor(prefix,device=candidates.device,dtype=candidates.dtype) if not torch.is_tensor(prefix) else prefix
    prefix=prefix.expand(candidates.shape[0],candidates.shape[1],prefix.shape[2],prefix.shape[3])
    return torch.cat([prefix,candidates],dim=2)
