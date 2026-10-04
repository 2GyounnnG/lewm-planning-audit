"""Read-only model/case/anchor identity audit for all reused H3X controls."""
import os,sys,json,csv
from pathlib import Path
from . import common
from .evaluate import file

def main():
    sys.path.insert(0,'/workspace/r4_v23_execution')
    from r4.common import digest
    from r3.model import official_identity
    r3=common.R3_ROOT;h2=Path('/workspace/r5/H2/reacher');m=common.read_json(h2/'AUDIT_MODEL_IDENTITY.json');a=common.read_json(h2/'CONTROL_REUSE_AUDIT.json')
    assert m['status']=='PASS_MODEL_IDENTITIES' and a['status']=='PASS_CONTROL_REUSE_IDENTITY' and a['control_count']==1200
    assert common.sha256(h2/'CONTROL_REUSE_AUDIT.json')==m['control_audit']['sha256'];official=official_identity('reacher');assert m['actual_load_official_identity']==official
    rolespath=r3/'manifests/reacher_data_roles.json';roles=common.read_json(rolespath);cases={c['case_id']:c for c in roles['cases']['EVAL']};assert len(cases)==100
    # CASE_WINDOWS incorporates the previously verified reset-seed provenance.
    assets=Path('/workspace/shared_data/r4_assets_reacher');windows=common.read_json(assets/'CASE_WINDOWS.json');casewin={e['case']['case_id']:e['case'] for e in windows['tasks']['reacher']['cases'] if e['case']['role']=='EVAL'}
    checks=[];counts={k:0 for k in ('H0','REFIT_103201','REFIT_103202','REFIT_103203')}
    for arm,cp in m['fixed30k_refit_routes'].items():assert common.sha256(r3/cp['path'])==cp['sha256'] and (r3/cp['path']).stat().st_size==cp['bytes']
    for c in a['controls']:
        folder=Path(c['control_folder']);complete=common.read_json(folder/'COMPLETE.json');assert common.sha256(folder/'COMPLETE.json')==c['complete']['sha256'];assert complete['identity_sha256']==c['identity_sha256'];r=common.read_json(folder/'result.json');assert common.sha256(folder/'result.json')==c['files']['result.json']['sha256'];assert r['status']=='COMPLETE'
        found=[]
        for p in folder.glob('attempt_*/STARTED.json'):
            d=common.read_json(p)
            if digest(d['identity'])==complete['identity_sha256']:found.append((p,d['identity']))
        assert len(found)==1;p,identity=found[0];assert identity['case']==casewin[c['case_id']] and identity['stream']==c['stream'] and identity['arm']==c['arm'];prov=identity['provenance'];assert prov==c['provenance'] and prov['official']==official
        assert prov['case_manifest']==common.sha256(assets/'CASE_WINDOWS.json')
        if c['arm']!='H0':assert prov['checkpoint']==m['fixed30k_refit_routes'][c['arm']]
        else:assert 'checkpoint' not in prov
        counts[c['arm']]+=1;checks.append({'case_id':c['case_id'],'arm':c['arm'],'stream':c['stream'],'identity_sha256':complete['identity_sha256'],'started':file(p),'result':file(folder/'result.json'),'complete':file(folder/'COMPLETE.json')})
    assert set(counts.values())=={300}
    reference=r3/'artifacts/open_loop/reacher';inputs=reference/'inputs.npz';targets=reference/'targets.npz';insha=common.sha256(inputs);tarsha=common.sha256(targets);rolesha=common.sha256(rolespath);h1b=[]
    for h in (1,2,5):
        p=Path(f'/workspace/r5/H1b/fits/reacher/RIDGE/latent_{h}/three/seed0/case_values.json');rows=common.read_json(p);assert len(rows)==100 and {r['case_id'] for r in rows}==set(cases)
        for r in rows:assert r['anchor_reference_sha256']==insha and r['target_reference_sha256']==tarsha and r['roles_sha256']==rolesha and r['anchor_raw']==cases[r['case_id']]['open_loop_anchor_raw']
        h1b.append(file(p))
    merged=Path('/workspace/r5/H1b/HISTORY_CONDITION_4TASK_RAW.csv');assert common.sha256(merged)=='1ff8f0396ccdf1ab0abb679087895e71998c969a769510b17097bfa483d47281'
    with merged.open() as f:rows=[r for r in csv.DictReader(f) if r['task']=='reacher']
    assert len(rows)==2400
    for r in rows:assert int(r['anchor_raw'])==cases[r['case_id']]['open_loop_anchor_raw']
    out=common.ROOT/'inputs/REFERENCE_CONTROL_AUDIT.json';common.atomic_json(out,{'status':'PASS_ALL_REUSED_MODELS_CASES_AND_ANCHORS','label':common.LABEL,'preset_H3_status':'NOT_TRIGGERED','closed_controls':1200,'closed_model_counts':counts,'open_reference_rows':2400,'independent_cases':100,'official_identity':official,'fixed30k_reference_routes':m['fixed30k_refit_routes'],'h2_model_audit':file(h2/'AUDIT_MODEL_IDENTITY.json'),'h2_control_audit':file(h2/'CONTROL_REUSE_AUDIT.json'),'case_windows':file(assets/'CASE_WINDOWS.json'),'roles':file(rolespath),'H1a_merged_reference':file(merged),'H1b_anchor_bindings':h1b,'open_loop_inputs':file(inputs),'open_loop_targets':file(targets),'control_checks':checks,'new_simulator_steps':0,'optimizer_updates':0,'code':file(__file__)})
    print('REFERENCE_CONTROL_AUDIT_PASS',1200,2400,common.sha256(out),flush=True)
if __name__=='__main__':main()
