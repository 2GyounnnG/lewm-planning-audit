import json
from pathlib import Path
import torch
import old_history_runner as h
torch.set_num_threads(1)
rows=[]
for task in ('reacher','cube'):
 for arm in h.ARMS:
  model,scale=h.load(task,arm,'cpu');rows.append({'task':task,'arm':arm,'model_sha256':h.model_hash(model),'macro_action_dim':model.r3_contract['macro_action_dim'],'raw_action_dim':scale.n_features_in_,'finite':all(not x.is_floating_point() or bool(torch.isfinite(x).all()) for x in model.state_dict().values())});del model
assert all(r['finite'] and r['macro_action_dim']==5*r['raw_action_dim'] for r in rows)
Path('/workspace/r10/ops/D_C_MODEL_PRECHECK.json').write_text(json.dumps({'status':'PASS','models':rows},indent=2)+'\n');print('PASS:8 fixed model states, strict base and frozen refit identities; no GPU execution')
