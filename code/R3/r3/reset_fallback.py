"""Bind explicitly validated missing-seed routing without changing raw case roles."""
from pathlib import Path
import copy
from . import common

MANIFEST='manifests/RESET_FALLBACK_MANIFEST.json'
PROVENANCE='SOURCE_SEED_UNKNOWN_VALIDATED_FIXED_SEED0_NOT_RECOVERED'

def checked_record(record):
    p=Path(record['path'])
    if p.is_absolute() or '..' in p.parts:raise RuntimeError('Fallback evidence path must stay relative to R3')
    path=common.ROOT/p
    if not path.resolve().is_relative_to(common.ROOT) or path.stat().st_size!=record['bytes'] or common.sha256(path)!=record['sha256']:raise RuntimeError('Frozen reset evidence changed')
    return path

def evidence(task):
    if task == 'pusht':
        from scripts.validate_reset_fallback import verify_receipt
        helper = 'scripts/validate_reset_fallback.py'
    elif task == 'reacher':
        from scripts.validate_reacher_reset import verify_receipt
        helper = 'scripts/validate_reacher_reset.py'
    else:
        raise RuntimeError('Unsupported task in reset fallback routing')
    manifest=common.read_json(MANIFEST)
    if manifest.get('status')!='VALIDATED_FIXED_RESET_FALLBACK_FROZEN' or task not in manifest['tasks']:raise RuntimeError('Missing real validated reset fallback')
    entry=manifest['tasks'][task]
    if entry['fallback_seed']!=0 or entry['provenance']!=PROVENANCE:raise RuntimeError('Only actually validated fixedseed0 fallback may be routed')
    roles_path=checked_record(entry['data_roles']);roles=common.read_json(roles_path)
    if entry['data_roles']['path']!=f'manifests/{task}_data_roles.json' or roles['task']!=task:raise RuntimeError('Fallback roles/task mismatch')
    cases=roles['cases']['TECH']+roles['cases']['EVAL'];affected=sorted(c['case_id'] for c in cases if c['reset_seed'] is None)
    if entry['affected_case_ids']!=affected:raise RuntimeError('Fallback cannot change which cases require validation')
    receipt_path=checked_record(entry['validation_receipt'])
    assets={c['source_asset_sha256'] for c in cases if c['reset_seed'] is None}
    if len(assets)!=1:raise RuntimeError('Fallback validation covers one explicit official source')
    receipt=verify_receipt(receipt_path,roles_sha256=entry['data_roles']['sha256'],source_asset_sha256=next(iter(assets)))
    if receipt['cases']!=roles['cases']['TECH'][:2]:raise RuntimeError('Reset evidence did not use the pre-frozen TECH cases')
    paths=[MANIFEST,entry['data_roles']['path'],entry['validation_receipt']['path'],'r3/reset_fallback.py',helper]
    for name,h in receipt['input_hashes'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or common.sha256(common.ROOT/p)!=h:raise RuntimeError('Validated reset implementation/input changed: '+name)
        paths.append(name)
    paths.extend(str(receipt_path.parent.relative_to(common.ROOT)/name) for name in receipt['files'])
    return manifest,entry,roles,sorted(set(paths))

def effective_case(task,original_case):
    result=copy.deepcopy(original_case)
    if result['reset_seed'] is not None:return result
    manifest,entry,roles,paths=evidence(task)
    actual=[c for c in roles['cases']['TECH']+roles['cases']['EVAL'] if c['case_id']==result['case_id']]
    if len(actual)!=1 or actual[0]!=result or result['source_seed'] is not None or result['requires_validated_reset_fallback'] is not True:raise RuntimeError('Fallback may only overlay its exact original frozen missing-seed case')
    result.update(reset_seed=entry['fallback_seed'],reset_seed_validation_sha256=entry['validation_receipt']['sha256'],reset_seed_provenance=entry['provenance'],reset_fallback_manifest_sha256=common.sha256(common.ROOT/MANIFEST))
    return result

def evidence_paths(task,roles):
    if not any(c['reset_seed'] is None for c in roles['cases']['TECH']+roles['cases']['EVAL']):return []
    return evidence(task)[3]
