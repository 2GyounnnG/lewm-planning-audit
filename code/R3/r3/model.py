"""Fixed official LeWM construction and predictor-only parameter replacements.

No hub/network resolution, optimizer, decoder, whitening or source mutation occurs
here. The returned model is the author's JEPA, so its planning API is preserved.
"""
from __future__ import annotations
import ast, functools, hashlib, importlib.util, json, logging, os, sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(os.environ.get('R3_ROOT',Path(__file__).resolve().parents[1])).resolve()
TRAINABLE_PREFIXES=('predictor.','pred_proj.')
DELTA_FORMAT='R3_PREDICTOR_PARAMETER_REPLACEMENTS_V1'

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def tensor_sha256(value):
    t=value.detach().cpu().contiguous()
    h=hashlib.sha256(json.dumps({'shape':list(t.shape),'dtype':str(t.dtype)},sort_keys=True).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes());return h.hexdigest()

def _checked_source(component,relative):
    path=ROOT/'source'/component/relative
    manifest=ROOT/'state'/f'{component}_source_manifest.json'
    record=json.loads(manifest.read_text())['files'][relative]
    if path.stat().st_size!=record['bytes'] or file_sha256(path)!=record['sha256']:
        raise RuntimeError('Official source identity changed: '+str(path))
    return path

def _source_module(component,relative,alias):
    path=_checked_source(component,relative)
    key=f'_r3_official_{alias}_{file_sha256(path)[:16]}'
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec)
        sys.modules[key]=module;spec.loader.exec_module(module)
    return sys.modules[key]

def _vit_factory():
    """Execute the exact audited constructor, avoiding unrelated package imports."""
    path=_checked_source('spt','stable_pretraining/backbone/utils.py')
    from transformers import ViTConfig,ViTModel
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='vit_hf')
    namespace={'nn':nn,'ViTConfig':ViTConfig,'ViTModel':ViTModel,'_TRANSFORMERS_AVAILABLE':True,'logging':logging}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['vit_hf']

def official_config(task):
    if task not in ('pusht','reacher'):raise ValueError('Only the two authorized official tasks')
    config=json.loads((ROOT/'official'/task/'config.json').read_text())
    if config.get('_target_') not in ('jepa.JEPA','stable_worldmodel.wm.lewm.LeWM'):
        raise ValueError('Unexpected official world-model constructor')
    if set(config)-{'_target_','encoder','predictor','action_encoder','projector','pred_proj'}:
        raise ValueError('Unreviewed top-level official model config fields')
    return config

def construct_official(config):
    """Known resolved official config only, preserving constructor arguments."""
    jepa=_source_module('lewm','jepa.py','jepa');module=_source_module('lewm','module.py','module')
    types={'module.ARPredictor':module.ARPredictor,'stable_worldmodel.wm.lewm.module.Predictor':module.ARPredictor,
           'module.Embedder':module.Embedder,'stable_worldmodel.wm.lewm.module.Embedder':module.Embedder,
           'module.MLP':module.MLP,'stable_worldmodel.wm.lewm.module.MLP':module.MLP,
           'torch.nn.BatchNorm1d':nn.BatchNorm1d,'torch.nn.LayerNorm':nn.LayerNorm,'torch.nn.Identity':nn.Identity}
    def create(spec):
        if spec is None:return None
        if not isinstance(spec,dict) or '_target_' not in spec:raise ValueError('Resolved constructor dict required')
        target=spec['_target_'];args={k:create(v) if isinstance(v,dict) and '_target_' in v else v for k,v in spec.items() if not k.startswith('_')}
        if set(k for k in spec if k.startswith('_'))-{'_target_','_partial_'}:raise ValueError('Unreviewed Hydra directive')
        if target=='stable_pretraining.backbone.utils.vit_hf':
            if args.get('pretrained') is not False:raise ValueError('Backbone construction cannot download or substitute pretrained weights')
            ctor=_vit_factory()
        elif target in types:ctor=types[target]
        else:raise ValueError('Unreviewed constructor: '+target)
        return functools.partial(ctor,**args) if spec.get('_partial_',False) else ctor(**args)
    required=('encoder','predictor','action_encoder','projector','pred_proj')
    if any(k not in config for k in required):raise ValueError('Official model config is incomplete')
    model=jepa.JEPA(**{k:create(config[k]) for k in required})
    # Original LeWM class has no num_frames attribute; expose only config metadata.
    model.r3_config=config
    model.r3_contract={'history_size':int(config['predictor']['num_frames']),
        'latent_dim':int(config['projector'].get('output_dim',config['projector']['input_dim'])),
        'macro_action_dim':int(config['action_encoder']['input_dim']),
        'image_size':int(config['encoder']['image_size']),'target_shift_macro':1,
        'latent_coordinate':'official eval ViT CLS followed by observation projector; no additional transform',
        'checkpoint_constructor_mapping':{'declared_model':config['_target_'],'implemented_model':'fixed lewm.jepa.JEPA',
             'declared_predictor':config['predictor']['_target_'],'implemented_predictor':'fixed lewm.module.ARPredictor'},
        'mapping_requires_strict_weight_load_and_zero_update_interface_validation':True}
    if model.r3_contract['latent_dim']!=int(config['predictor']['input_dim']) or model.r3_contract['latent_dim']!=int(config['predictor']['output_dim']):
        raise ValueError('Official projected latent/predictor dimensions differ')
    return model.float().eval()

def official_identity(task):
    folder=ROOT/'official'/task
    names={'lewm_jepa':('lewm','jepa.py'),'lewm_module':('lewm','module.py'),
           'lewm_train':('lewm','train.py'),'lewm_utils':('lewm','utils.py'),
           'vit_constructor':('spt','stable_pretraining/backbone/utils.py')}
    return {'task':task,'weights_sha256':file_sha256(folder/'weights.pt'),
        'config_sha256':file_sha256(folder/'config.json'),
        'source_sha256':{k:file_sha256(_checked_source(*pair)) for k,pair in names.items()}}

def load_official(task,device='cpu',expected=None):
    identity=official_identity(task)
    if expected is None:
        from .common import official_asset
        expected=official_asset(task)
    if expected is not None:
        for key,value in expected.items():
            if key in identity and identity[key]!=value:raise RuntimeError('Official asset identity differs: '+key)
    model=construct_official(official_config(task))
    state=torch.load(ROOT/'official'/task/'weights.pt',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or any(not isinstance(k,str) or not torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('Official weights must be a plain tensor state_dict; no object checkpoints')
    # No prefix stripping, renamed keys, discarded keys or non-strict fallback.
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:raise RuntimeError('Strict load did not match')
    if any(torch.is_floating_point(t) and not torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite official weights')
    model.r3_identity=identity;model.r3_strict_load={'missing_keys':[],'unexpected_keys':[],'state_dict_keys':len(state)}
    del state
    model.requires_grad_(False);return model.to(device=device,dtype=torch.float32).eval()

def _parameter_groups(model):
    groups={}
    for name,param in model.named_parameters(remove_duplicate=False):groups.setdefault(id(param),[]).append((name,param))
    for pairs in groups.values():
        allowed={name.startswith(TRAINABLE_PREFIXES) for name,_ in pairs}
        if len(allowed)>1:raise ValueError('Parameter alias crosses frozen/trainable boundary: '+str([n for n,_ in pairs]))
    return list(groups.values())

def refit_mode(model,training=True):
    model.eval();model.requires_grad_(False)
    for pairs in _parameter_groups(model):
        if pairs[0][0].startswith(TRAINABLE_PREFIXES):pairs[0][1].requires_grad_(training)
    if training:model.predictor.train();model.pred_proj.train()
    for module in model.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
    return model

def trainable_parameters(model):
    result=[pairs[0][1] for pairs in _parameter_groups(model) if pairs[0][0].startswith(TRAINABLE_PREFIXES)]
    if not result or set(map(id,result))!={id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Optimizer parameter list does not exactly match the whitelist')
    return result

def whitelist_manifest(model):
    rows=[]
    for pairs in _parameter_groups(model):
        for name,param in pairs:
            rows.append({'name':name,'shape':list(param.shape),'numel':param.numel(),'dtype':str(param.dtype),
              'trainable':name.startswith(TRAINABLE_PREFIXES),'initial_sha256':tensor_sha256(param),
              'shared_parameter_identity':pairs[0][0],'alias_names':[n for n,_ in pairs]})
    return {'parameters':rows,'trainable_parameters':sum(pairs[0][1].numel() for pairs in _parameter_groups(model) if pairs[0][0].startswith(TRAINABLE_PREFIXES)),
        'frozen_parameters':sum(pairs[0][1].numel() for pairs in _parameter_groups(model) if not pairs[0][0].startswith(TRAINABLE_PREFIXES)),
        'all_buffers_frozen':True,'all_BN_running_statistics_frozen':True,
        'parameter_whitelist':list(TRAINABLE_PREFIXES),'SIGReg_has_trainable_parameter_path':False}

def frozen_hashes(model):
    tensors={f'parameter:{n}':tensor_sha256(p) for n,p in model.named_parameters() if not n.startswith(TRAINABLE_PREFIXES)}
    tensors.update({f'buffer:{n}':tensor_sha256(b) for n,b in model.named_buffers()})
    return {'sha256':hashlib.sha256(json.dumps(tensors,sort_keys=True).encode()).hexdigest(),'tensors':tensors}

def assert_frozen(model,expected):
    actual=frozen_hashes(model)
    if actual!=expected:raise AssertionError('Frozen official parameters or buffers changed')
    if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith(TRAINABLE_PREFIXES)):
        raise AssertionError('Frozen official parameter received a gradient')
    if any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)):
        raise AssertionError('BatchNorm statistics are not frozen')
    if any(m.training for name in ('encoder','projector','action_encoder') for m in getattr(model,name).modules()):
        raise AssertionError('A frozen source component left evaluation mode')
    return actual

def cached_predict(model,z,normalized_macro_actions):
    if z.ndim!=3 or normalized_macro_actions.ndim!=3 or z.shape[:2]!=normalized_macro_actions.shape[:2]:raise ValueError('Expected matching [batch,history,channel] inputs')
    contract=model.r3_contract
    if not 1<=z.shape[1]<=contract['history_size'] or z.shape[-1]!=contract['latent_dim'] or normalized_macro_actions.shape[-1]!=contract['macro_action_dim']:
        raise ValueError('Cached predictor dimensions differ from official config')
    if z.requires_grad or normalized_macro_actions.requires_grad:raise ValueError('Cached observations/actions must be frozen')
    # action_encoder remains frozen even while predictor and pred_proj train.
    with torch.no_grad():act_emb=model.action_encoder(normalized_macro_actions.float())
    return model.predict(z.float(),act_emb)

def cached_rollout(model,initial_z,normalized_macro_actions,horizon):
    if horizon<1 or normalized_macro_actions.shape[1]!=initial_z.shape[1]+horizon-1:raise ValueError('Free rollout action length')
    history=initial_z;outputs=[];limit=model.r3_contract['history_size']
    for step in range(horizon):
        lo=max(0,history.shape[1]-limit)
        # No future observation enters this function; action index leaves its frame.
        acts=normalized_macro_actions[:,lo:initial_z.shape[1]+step]
        with torch.no_grad():act_emb=model.action_encoder(acts.float())
        prediction=model.predict(history[:,lo:].float(),act_emb)[:,-1]
        outputs.append(prediction);history=torch.cat((history,prediction[:,None]),dim=1)
    return torch.stack(outputs,dim=1)

def delta_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(TRAINABLE_PREFIXES)}

def delta_payload(model,**metadata):
    return {'format':DELTA_FORMAT,'base_identity':model.r3_identity,'contract':model.r3_contract,
        'replacement_parameters':delta_state(model),'frozen_sha256':frozen_hashes(model)['sha256'],
        'storage_semantics':'exact predictor/pred_proj parameter replacements, not lossy arithmetic differences',**metadata}

def apply_delta(model,path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    payload=saved.get('delta',saved)
    if payload.get('format')!=DELTA_FORMAT or payload.get('base_identity')!=model.r3_identity or payload.get('contract')!=model.r3_contract:
        raise RuntimeError('Predictor replacement identity/coordinate mismatch')
    before=frozen_hashes(model)
    if payload['frozen_sha256']!=before['sha256']:raise RuntimeError('Predictor replacement frozen base differs')
    target={n:p for n,p in model.named_parameters() if n.startswith(TRAINABLE_PREFIXES)}
    values=payload['replacement_parameters']
    if set(values)!=set(target):raise ValueError('Predictor replacement key whitelist mismatch')
    with torch.no_grad():
        for name,p in target.items():
            value=values[name]
            if value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all():raise ValueError('Invalid predictor replacement: '+name)
            p.copy_(value.to(p.device))
    model.eval();assert_frozen(model,before)
    return {k:v for k,v in payload.items() if k!='replacement_parameters'}
