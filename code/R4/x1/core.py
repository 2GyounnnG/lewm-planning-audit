"""Isolated X1 configuration and verified read-only R3 implementation reuse."""
from __future__ import annotations
import hashlib, importlib, json, os, sys, tempfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

TASKS = ('tworoom', 'cube')
SEEDS = (103201, 103202, 103203)
LEWM_COMMIT = '8edfeb336732b5f3ce7b8b210d0ba370a09e2cac'
SWM_COMMIT = 'abdced49809d5eae38e24b27dc7b635c502c4812'
R3_ROOT = Path(os.environ.get('X1_R3_ROOT', '/workspace/shared_data/r3')).resolve()
ROOT = Path(os.environ.get('X1_ROOT', '/workspace/x1_tworoom')).resolve()
PLAN = {'horizon': 5, 'receding_horizon': 5, 'action_block': 5}
CEM = {'batch_size': 1, 'num_samples': 300, 'var_scale': 1., 'n_steps': 30, 'topk': 30}
CONFIG = {
    'tworoom': {'repo': 'quentinll/lewm-tworooms', 'dataset_name': 'tworoom',
        'model_revision': '77adaae0bc31deab21c93740d1f8bb947cd0bdec',
        'weights_sha256': '566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd',
        'data_revision': '6903a2de048b13819d812da0b4dd661290bc01e4',
        'archive_sha256': '494b1a02f0765cd9a0d9daf1786c419ced1009977fc45d01e3158932f8d080ca',
        'world': {'env_name': 'swm/TwoRoom-v1'},
        'reset_keys': ['proprio'], 'goal_key': 'proprio',
        'callables': [{'method': '_set_state', 'args': {'state': {'value': 'proprio'}}},
            {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_proprio'}}}],
        'success': 'step terminated iff Euclidean distance(agent,target) < 16.0; World OR over executed raw steps',
        'success_source': 'stable_worldmodel/envs/two_room/env.py'},
    'cube': {'repo': 'quentinll/lewm-cube', 'dataset_name': 'ogbench/cube_single_expert',
        'model_revision': 'b0747c5002e86d2ce8f3cd8178004b97524c587d',
        'weights_sha256': '2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89',
        'data_revision': '02a19a67a0dc8c9d6215f89c19e0a597691e152a',
        'archive_sha256': '3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600',
        'world': {'env_name': 'swm/OGBCube-v0', 'env_type': 'single', 'ob_type': 'states',
            'multiview': False, 'width': 224, 'height': 224, 'visualize_info': False, 'terminate_at_goal': True},
        'reset_keys': ['qpos', 'qvel', 'privileged_block_0_pos', 'privileged_block_0_quat'],
        'goal_key': 'privileged_block_0_pos',
        'callables': [{'method': 'set_state', 'args': {'qpos': {'value': 'qpos'}, 'qvel': {'value': 'qvel'}}},
            {'method': 'set_target_pos', 'args': {'cube_id': {'value': 0, 'in_dataset': False},
                'target_pos': {'value': 'goal_privileged_block_0_pos'}, 'target_quat': {'value': 'goal_privileged_block_0_quat'}}}],
        'success': 'single block xyz Euclidean distance to target <= 0.04m; target quaternion does not enter success; World OR of termination',
        'success_source': 'stable_worldmodel/envs/ogbench/cube_env.py'},
}

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f: f.write(canonical(value) + b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise RuntimeError('Frozen input differs: ' + str(path))
    else: atomic(path, value)

def file_record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def verify(record):
    path = Path(record['path'])
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise RuntimeError('Asset identity differs: ' + str(path))
    return path

def r3(module):
    """Import the historical source; no writes or replacement of its files."""
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; sys.dont_write_bytecode = True
    os.environ['R3_ROOT'] = str(R3_ROOT)
    if str(R3_ROOT) not in sys.path: sys.path.insert(0, str(R3_ROOT))
    result = importlib.import_module('r3.' + module)
    if not Path(result.__file__).resolve().is_relative_to(R3_ROOT): raise RuntimeError('R3 import escaped read-only source')
    return result

def policy():
    import torch
    torch.set_num_threads(int(os.environ.get('X1_THREADS', '1')))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

def task_contract(task):
    if task not in TASKS: raise ValueError('X1 task must be tworoom or cube')
    files = {}
    for component, names in {'lewm': ['config/eval/' + task + '.yaml', 'config/eval/solver/cem.yaml', 'eval.py', 'jepa.py', 'module.py'],
            'swm_compat': ['stable_worldmodel/policy.py', 'stable_worldmodel/solver/cem.py', 'stable_worldmodel/world/world.py', CONFIG[task]['success_source']]}.items():
        manifest = read(R3_ROOT / 'state' / (component + '_source_manifest.json'))
        expected_revision = LEWM_COMMIT if component == 'lewm' else SWM_COMMIT
        if manifest['revision'] != expected_revision: raise RuntimeError('Wrong pinned commit: ' + component)
        # Each real source manifest binds individual source bytes to its commit.
        for name in names:
            p = R3_ROOT / 'source' / component / name
            if sha(p) != manifest['files'][name]['sha256']: raise RuntimeError('Pinned source differs: ' + str(p))
            files[component + '/' + name] = sha(p)
    return {'version': 'R4_V2_3_X1_V1', 'task': task, 'lewm_commit': LEWM_COMMIT, 'swm_commit': SWM_COMMIT,
        'source_sha256': files, 'official': CONFIG[task], 'plan': PLAN, 'cem': CEM,
        'goal_offset_raw': 25, 'raw_budget': 50, 'max_episode_steps': 100,
        'executed_raw_per_replan': 25, 'eval_cases': 100, 'tech_cases': 4,
        'refit_seeds': list(SEEDS), 'refit_updates': 30000, 'batch_size': 128,
        'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .001,
            'lr_peak': 5e-5, 'lr_end': 5e-6, 'warmup': 500, 'schedule_horizon': 30000, 'clip_grad_norm': 1., 'foreach': False},
        'precision': 'FP32_NO_AMP_NO_TF32', 'trainable': ['predictor.', 'pred_proj.'], 'all_buffers_frozen': True,
        'raw_action_dimension': 'Read from strict official checkpoint config and HDF5; never assume 2 for Cube',
        'split': 'Exact R3 metadata hash namespaces, rounding and role builder; task allowlist extended only',
        'family': 'Explicit source family if established; otherwise FAMILY_UNKNOWN without independence claim',
        'case_seed': 'Explicit source seed; absent requires a separate successful TECH reset receipt',
        'main_arms': ['H0'] + [f'REFIT_{s}' for s in SEEDS],
        'second_batch': 'R4_ALT_CEM_1/2 only after main delivery and TECH P90<=30 seconds'}

def load_model(task, device='cpu'):
    import torch
    model_api = r3('model'); assets = read(ROOT / 'manifests' / f'{task}_model_assets.json')
    cfgpath = verify(assets['files']['config.json']); weightpath = verify(assets['files']['weights.pt'])
    config = read(cfgpath)
    model = model_api.construct_official(config)
    state = torch.load(weightpath, map_location='cpu', weights_only=True)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys: raise RuntimeError('Official strict load mismatch')
    if any(torch.is_floating_point(v) and not torch.isfinite(v).all() for v in state.values()): raise ValueError('Nonfinite official weights')
    if model.r3_contract['history_size'] != 3 or model.r3_contract['macro_action_dim'] % 5:
        raise ValueError('Official checkpoint does not satisfy inherited history/action-block contract')
    model.r3_identity = {'task': task, 'weights_sha256': sha(weightpath), 'config_sha256': sha(cfgpath),
        'official_repo': assets['repo'], 'official_revision': assets['revision'], 'source_contract_sha256': digest(task_contract(task))}
    return model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
