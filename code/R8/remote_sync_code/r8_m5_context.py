"""R8 M5 FULL_HISTORY context intervention (protocol extension only)."""
import numpy as np

def history_indices(raw_count):
    if raw_count == 0: return [0]
    return [raw_count]

def real_history_observation(info, observations, raw_count, history_mode='FULL_HISTORY', prefix_pixels=None):
    if history_mode != 'FULL_HISTORY':
        return info
    if raw_count == 0:
        if prefix_pixels is None or np.asarray(prefix_pixels).shape != (2,224,224,3):
            raise RuntimeError('FULL_HISTORY prefix pixels missing or wrong shape')
        if len(observations) != 1: raise RuntimeError('FULL_HISTORY initial observation alignment')
        out = dict(info)
        out['pixels'] = np.stack([prefix_pixels[0], prefix_pixels[1], observations[0]['pixels']])[None]
        return out
    # The second and later plans use the executing current frame only.
    return info

def candidates_with_prefix(info, candidates, rows, action_processor, history_mode='FULL_HISTORY', prefix_actions=None):
    import torch
    if history_mode != 'FULL_HISTORY': return candidates
    history = int(info['pixels'].shape[2])
    if history == 1:
        if rows == []: return candidates
        return candidates
    if history != 3 or rows != []: raise RuntimeError('FULL_HISTORY first-plan context/action alignment')
    if prefix_actions is None or np.asarray(prefix_actions).shape != (10,2):
        raise RuntimeError('FULL_HISTORY prefix actions missing or wrong shape')
    raw = np.asarray(prefix_actions)
    prefix = action_processor.transform(raw).reshape(1,1,2,10)
    prefix = torch.as_tensor(prefix, device=candidates.device, dtype=candidates.dtype)
    prefix = prefix.expand(candidates.shape[0], candidates.shape[1], 2, 10)
    if candidates.shape[-2:] != (5,10): raise RuntimeError('CEM horizon/action block changed')
    return torch.cat([prefix, candidates], dim=2)
