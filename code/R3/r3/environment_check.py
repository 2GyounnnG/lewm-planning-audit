"""Root-invoked headless environment checks, never a CEM trajectory or fit.

No environment/dependency import at module import. The CLI only resets/steps
when --run is explicit. Run on the authorized host in the isolated R3 venv.
"""
from __future__ import annotations
import argparse, ctypes.util, hashlib, importlib, importlib.metadata, inspect
import json, os, platform, sys, time, traceback, xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from .common import ROOT, atomic_json, now, sha256

PACKAGES = ('torch', 'torchvision', 'numpy', 'gymnasium', 'dm-control', 'dm-env',
            'mujoco', 'pygame', 'pymunk', 'shapely', 'opencv-python',
            'opencv-python-headless', 'Pillow', 'lancedb', 'pylance', 'pyarrow',
            'h5py', 'hdf5plugin', 'imageio', 'imageio-ffmpeg', 'loguru',
            'stable-worldmodel', 'stable-pretraining', 'transformers')


def _error(error):
    return {'type': type(error).__name__, 'message': str(error),
            'missing_module': getattr(error, 'name', None) if isinstance(error, ImportError) else None,
            'traceback': traceback.format_exc()}


def versions():
    result = {}
    for name in PACKAGES:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = None
    return result


def referenced_assets(xml, assets, suite_dir):
    """Metadata-only resolution of MJCF file references before env creation.

    Sources are dm_control's get_model_and_assets result and its installed suite
    directory. No download, inferred asset substitute or model compilation.
    """
    suite_dir = Path(suite_dir).resolve()
    if not isinstance(xml, bytes) or not isinstance(assets, dict):
        raise TypeError('Expected actual dm_control XML bytes and asset mapping')
    supplied = {}
    for name, body in assets.items():
        if not isinstance(body, bytes): raise TypeError('Unexpected asset payload type: '+str(name))
        supplied[name] = {'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    records, missing, visited = [], [], set()
    def scan(body, owner):
        identity = (owner, hashlib.sha256(body).hexdigest())
        if identity in visited: return
        visited.add(identity)
        tree = ET.fromstring(body)
        for element in tree.iter():
            name = element.attrib.get('file')
            if name is None: continue
            candidates = list(dict.fromkeys((name, str(PurePosixPath(name)),
                str(PurePosixPath(owner).parent / name))))
            value = None; origin = None
            for candidate in candidates:
                if candidate in assets:
                    value = assets[candidate]; origin = {'kind': 'provided_asset', 'name': candidate}; break
                path = (suite_dir / candidate).resolve()
                if path.is_relative_to(suite_dir) and path.is_file():
                    value = path.read_bytes(); origin = {'kind': 'installed_package_file', 'path': str(path)}; break
            if value is None:
                missing.append({'owner': owner, 'file': name, 'attempted_relative_names': candidates}); continue
            records.append({'owner': owner, 'file': name, 'resolved': origin,
                            'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
            if element.tag == 'include' or PurePosixPath(name).suffix.lower() == '.xml':
                scan(value, str(PurePosixPath(owner).parent / name))
    # The pinned Reacher constructor performs exactly this path replacement.
    rewritten = xml.replace(b'file="./common/', b'file="common/')
    scan(rewritten, 'reacher.xml')
    return {'status': 'PASS' if not missing else 'MISSING_ASSETS',
            'original_xml_sha256': hashlib.sha256(xml).hexdigest(), 'original_xml_bytes': len(xml),
            'constructor_rewritten_xml_sha256': hashlib.sha256(rewritten).hexdigest(),
            'provided_assets': supplied, 'required_file_references': records, 'missing': missing,
            'suite_directory': str(suite_dir), 'compiled_environment': False}


def action_scale_probe(action_scale_module):
    """Exercise installed action_scale with a recording stub, no simulator."""
    import numpy as np
    from dm_env import specs
    lower = np.array([-2., 10.]); upper = np.array([4., 20.])
    class RecordingActionSink:
        def action_spec(self): return specs.BoundedArray((2,), np.float64, lower, upper)
        def step(self, action): return np.asarray(action, dtype=np.float64).copy()
    wrapper = action_scale_module.Wrapper(RecordingActionSink(), minimum=-1., maximum=1.)
    inputs = np.array([[-1., -1.], [0., 0.], [1., 1.], [2., -2.]])
    outputs = np.stack([wrapper.step(action.copy()) for action in inputs])
    affine = (inputs + 1.) * (upper - lower) / 2. + lower
    clipped = np.clip(affine, lower, upper)
    if np.allclose(outputs, affine, rtol=0, atol=1e-12): semantics = 'AFFINE_RESCALE_WITHOUT_CLIP'
    elif np.allclose(outputs, clipped, rtol=0, atol=1e-12): semantics = 'AFFINE_RESCALE_WITH_CLIP'
    else: semantics = 'OTHER_MAPPING_REQUIRES_SOURCE_REVIEW'
    return {'status': 'PASS' if semantics != 'OTHER_MAPPING_REQUIRES_SOURCE_REVIEW' else 'REVIEW_REQUIRED',
            'semantics': semantics, 'synthetic_input_bounds': [-1., 1.],
            'native_lower': lower.tolist(), 'native_upper': upper.tolist(),
            'input_actions': inputs.tolist(), 'forwarded_actions': outputs.tolist(),
            'environment_resets': 0, 'environment_steps': 0,
            'note': 'Recording stub probes installed wrapper code including out-of-bound inputs; this is not a control trajectory.'}


def _array_info(value):
    import numpy as np
    array = np.asarray(value)
    numeric = array.dtype.kind in 'biufc'
    finite = bool(np.isfinite(array).all()) if numeric else None
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'finite': finite,
            'minimum': float(array.min()) if numeric and array.size and finite else None,
            'maximum': float(array.max()) if numeric and array.size and finite else None,
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def _observation_info(info):
    keys = ('pixels', 'state', 'proprio', 'qpos', 'qvel')
    result = {key: _array_info(info[key]) for key in keys if key in info}
    if 'pixels' not in result or result['pixels']['shape'] != [1, 1, 224, 224, 3]:
        raise ValueError('Expected genuine rendered World pixels[1,1,224,224,3]')
    if any(row['finite'] is not True for row in result.values()):
        raise ValueError('Nonfinite environment image/state observation')
    return result


def _task_check(swm, task, seed):
    import numpy as np
    record = {'task': task, 'reset_seed': seed, 'status': 'STARTED',
              'environment_resets': 0, 'raw_environment_steps': 0,
              'environment_reset_attempts': 0, 'raw_step_attempts': 0, 'close_called': False}
    world = None; started = time.perf_counter()
    try:
        kwargs = {'task': 'qpos_match'} if task == 'reacher' else {}
        world = swm.World(env_name='swm/PushT-v1' if task == 'pusht' else 'swm/ReacherDMControl-v0',
                          num_envs=1, max_episode_steps=100, image_shape=(224, 224), **kwargs)
        environment = world.envs.envs[0].unwrapped
        record['actual_environment_class'] = type(environment).__module__+'.'+type(environment).__name__
        record['environment_reset_attempts'] = 1
        world.reset(seed=[seed]); record['environment_resets'] = 1
        record['reset_observation'] = _observation_info(world.infos)
        action_space = world.envs.single_action_space
        record['action_space'] = {'shape': list(action_space.shape), 'dtype': str(action_space.dtype),
                                  'low': action_space.low.tolist(), 'high': action_space.high.tolist()}
        if tuple(action_space.shape) != (2,): raise ValueError('Unexpected action shape')
        if task == 'pusht':
            state = np.asarray(world.infos['state'][0, -1]).copy()
            environment._set_state(state)
            target = state.copy(); target[2:4] += 30.
            environment._set_goal_state(target)
            record['reset_methods_checked'] = ['_set_state', '_set_goal_state']
            record['pusht_relative_actions'] = bool(environment.relative)
            record['pusht_relative_action_scale'] = float(environment.action_scale)
        else:
            if environment._task_name != 'qpos_match': raise ValueError('Incorrect Reacher task')
            qpos = np.asarray(world.infos['qpos'][0, -1]).copy()
            qvel = np.asarray(world.infos['qvel'][0, -1]).copy()
            environment.set_state(qpos, qvel)
            target = qpos + np.array([0.25, -0.25]); environment.set_target_qpos(target)
            record['reset_methods_checked'] = ['set_state', 'set_target_qpos']
            record['actual_task'] = environment._task_name
            record['dm_control_action_repeat'] = int(environment.action_repeat)
            record['qpos_match_threshold_radians'] = float(environment.env.task.qpos_threshold)
            record['mujoco_model'] = {'nq': int(environment.env.physics.model.nq),
                                      'nv': int(environment.env.physics.model.nv),
                                      'nu': int(environment.env.physics.model.nu)}
        record['synthetic_diagnostic_target'] = target.tolist()
        action = np.array([[0.05, -0.05]], dtype=action_space.dtype)
        if not np.all(action >= action_space.low) or not np.all(action <= action_space.high):
            raise ValueError('Diagnostic action outside official bounds')
        before = time.perf_counter()
        record['raw_step_attempts'] = 1
        _, reward, terminated, truncated, info = world.envs.step(action)
        record['raw_environment_steps'] = 1
        record['single_step_wall_seconds'] = time.perf_counter() - before
        record['step_observation'] = _observation_info(info)
        if np.asarray(reward).shape != (1,) or not np.isfinite(reward).all(): raise ValueError('Nonfinite/wrong-shaped reward')
        if np.asarray(terminated).shape != (1,) or np.asarray(truncated).shape != (1,): raise ValueError('Wrong-shaped done flags')
        record.update(status='PASS', action=action[0].tolist(), reward=float(reward[0]),
                      terminated=bool(terminated[0]), truncated=bool(truncated[0]),
                      headless_render_evidence='actual finite 224x224 RGB arrays returned on reset and step; no video saved')
    except Exception as error:
        record.update(status='FAILED', error=_error(error))
    finally:
        if world is not None:
            try:
                world.close(); record['close_called'] = True
            except Exception as error:
                record['close_error'] = _error(error); record['status'] = 'FAILED'
        record['wall_seconds'] = time.perf_counter() - started
    return record


def preflight(output_dir, *, execute=False, source_root=None):
    """Called only by the root's authorized remote execution process."""
    from .planning import load_official_api, verify_official_sources
    folder = Path(output_dir)
    if (folder / 'ENVIRONMENT_PREFLIGHT.json').exists():
        raise RuntimeError('Keep the prior check evidence; use a new attempt directory')
    folder.mkdir(parents=True, exist_ok=True)
    result = {'status': 'STARTED', 'started_at': now(), 'execute_reset_and_single_step': execute,
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'versions': versions(), 'helper_sha256': sha256(__file__),
              'counts': {'optimizer_updates': 0, 'model_forward_calls': 0, 'planner_solve_calls': 0,
                         'complete_policy_trajectories': 0, 'environment_resets': 0, 'raw_environment_steps': 0,
                         'environment_reset_attempts': 0, 'raw_step_attempts': 0},
              'source_case_reset_equivalence_checked': False,
              'scope': 'Dependency/resource/headless reset+single-step checks only; not official model or CEM scientific reproduction.'}
    try:
        result['source'] = verify_official_sources(source_root)
        if os.environ.get('MUJOCO_GL') not in (None, 'egl'):
            raise RuntimeError('Use a fresh process with MUJOCO_GL=egl, matching official eval.py')
        os.environ['MUJOCO_GL'] = 'egl'; os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        result['headless_environment'] = {k: os.environ.get(k) for k in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', 'SDL_VIDEODRIVER', 'SDL_AUDIODRIVER', 'DISPLAY', 'CUDA_VISIBLE_DEVICES')}
        result['system_GL_libraries'] = {name: ctypes.util.find_library(name) for name in ('EGL', 'GL', 'OSMesa')}
        swm, cem, policy, _ = load_official_api(source_root)
        result['actual_imports'] = {}
        for label, obj in (('swm', swm), ('official_cem', cem), ('official_policy', policy)):
            path = Path(inspect.getfile(obj))
            result['actual_imports'][label] = {'path': str(path), 'sha256': sha256(path)}
        try:
            reacher = importlib.import_module('dm_control.suite.reacher')
            action_scale = importlib.import_module('dm_control.suite.wrappers.action_scale')
            xml, assets = reacher.get_model_and_assets()
            result['reacher_assets'] = referenced_assets(xml, assets or {}, Path(reacher.__file__).parent)
            source_path = Path(inspect.getsourcefile(action_scale))
            source = source_path.read_bytes(); (folder / 'dm_control_action_scale.py').write_bytes(source)
            result['dm_control_action_scale'] = {'installed_source_path': str(source_path),
                'source_sha256': hashlib.sha256(source).hexdigest(), 'bytes': len(source),
                'saved_source': 'dm_control_action_scale.py', 'wrapper_source': inspect.getsource(action_scale.Wrapper),
                'probe': action_scale_probe(action_scale)}
        except Exception as error:
            result['reacher_assets_or_action_scale_error'] = _error(error)
        result['pusht_resource_audit'] = {'repository_nontext_assets_required': [],
            'basis': 'Pinned PushT env.py and envs/utils.py construct geometry/rendering procedurally; no image/XML/mesh file load on this path.'}
        if execute:
            result['tasks'] = {'pusht': _task_check(swm, 'pusht', 1030001)}
            if result.get('reacher_assets', {}).get('status') == 'MISSING_ASSETS':
                result['tasks']['reacher'] = {'task': 'reacher', 'status': 'BLOCKED_RESOURCE',
                    'reason': 'Required MJCF file references are missing; no environment instantiated',
                    'environment_resets': 0, 'raw_environment_steps': 0, 'environment_reset_attempts': 0,
                    'raw_step_attempts': 0, 'close_called': False}
            else:
                result['tasks']['reacher'] = _task_check(swm, 'reacher', 1030002)
            for key in ('environment_resets', 'raw_environment_steps', 'environment_reset_attempts', 'raw_step_attempts'):
                result['counts'][key] = sum(r[key] for r in result['tasks'].values())
        assets_ok = result.get('reacher_assets', {}).get('status') == 'PASS'
        scale_ok = result.get('dm_control_action_scale', {}).get('probe', {}).get('status') == 'PASS'
        tasks_ok = execute and all(r['status'] == 'PASS' and r['close_called'] for r in result['tasks'].values())
        result['status'] = 'PASS' if assets_ok and scale_ok and tasks_ok else 'INSPECTED_NOT_EXECUTED' if not execute and assets_ok and scale_ok else 'FAILED'
    except Exception as error:
        result.update(status='FAILED', import_or_source_error=_error(error))
    result['completed_at'] = now()
    atomic_json(folder / 'ENVIRONMENT_PREFLIGHT.json', result)
    files = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'SHA256.json'}
    atomic_json(folder / 'SHA256.json', {'status': result['status'], 'files': files})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--run', action='store_true', help='Actually reset and single-step both environments; no model/policy trajectory.')
    args = parser.parse_args()
    result = preflight(args.output_dir, execute=args.run, source_root=args.source_root)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'output_dir': str(args.output_dir)}))
    raise SystemExit(0 if result['status'] in ('PASS', 'INSPECTED_NOT_EXECUTED') else 1)


if __name__ == '__main__': main()
