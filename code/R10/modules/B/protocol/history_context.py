"""R5/R8 history intervention, with action dimension inferred from checkpoint."""
import numpy as np
import torch

MODES=('H_REAL3_REPLAN','REAL3_NULLACT','REPEAT3_REALACT')

def indices(raw_count, mode):
    if mode=='H_POLICY':return [raw_count]
    if raw_count==0:return [0]
    assert raw_count==25 and mode in MODES
    return [raw_count]*3 if mode=='REPEAT3_REALACT' else [raw_count-10,raw_count-5,raw_count]

def observation(info, frames, raw_count, mode):
    if mode=='H_POLICY':return info
    if raw_count!=25:return info
    assert len(frames)==raw_count+1
    out=dict(info);out['pixels']=np.stack([frames[i] for i in indices(raw_count,mode)])[None]
    return out

def with_prefix(info, candidates, actions, scaler, mode, macro_dim):
    assert tuple(candidates.shape)==(1,300,5,macro_dim)
    if mode=='H_POLICY':
        assert info['pixels'].shape[2]==1
        return candidates
    if len(actions)==0:
        assert info['pixels'].shape[2]==1
        return candidates
    assert len(actions)==25 and info['pixels'].shape[2]==3
    raw=np.stack(actions[-10:]);assert raw.shape==(10,macro_dim//5)
    prefix=np.zeros((2,macro_dim),np.float32) if mode=='REAL3_NULLACT' else scaler.transform(raw).reshape(2,macro_dim)
    prefix=torch.as_tensor(prefix,device=candidates.device,dtype=candidates.dtype).reshape(1,1,2,macro_dim)
    return torch.cat([prefix.expand(1,300,2,macro_dim),candidates],dim=2)
