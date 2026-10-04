"""Independent model-identity supplement; never overwrites the control audit."""
import argparse,json,os,sys
from pathlib import Path
from r4.common import file_record,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--r3-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    os.environ['R3_ROOT']=str(a.r3_root);sys.path.insert(0,str(a.r3_root))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from r3.model import load_official
    model=load_official(a.task,'cpu');identity=model.r3_identity
    audit_path=a.root/'CONTROL_REUSE_AUDIT.json';audit=json.loads(audit_path.read_text())
    routes=json.loads((a.r3_root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms']
    refits={}
    for seed in [103201,103202,103203]:
        route=next(r for r in routes if r.get('step')==30000 and r['seed']==seed)
        ck=route['checkpoint'];actual=file_record(a.r3_root/ck['path'])
        assert actual=={k:ck[k] for k in ['bytes','sha256']}
        refits['REFIT_'+str(seed)]=ck
    counts={arm:0 for arm in ['H0',*refits]}
    for c in audit['controls']:
        expected={'official':identity,'case_manifest':audit['case_manifest']['sha256']}
        if c['arm']!='H0':expected['checkpoint']=refits[c['arm']]
        assert c['provenance']==expected
        counts[c['arm']]+=1
    assert set(counts.values())=={300}
    d={'status':'PASS_MODEL_IDENTITIES','task':a.task,'actual_load_official_identity':identity,'fixed30k_refit_routes':refits,'compared_control_counts':counts,
        'control_audit':{'path':str(audit_path),**file_record(audit_path)},'OPEN_LOOP_ROUTING':{'path':str(a.r3_root/'manifests/OPEN_LOOP_ROUTING.json'),**file_record(a.r3_root/'manifests/OPEN_LOOP_ROUTING.json')},
        'model_parameters_frozen':all(not v.requires_grad for v in model.parameters()),'optimizer_updates':0,'simulator_steps':0,'producer':file_record(__file__)}
    out=a.root/'AUDIT_MODEL_IDENTITY.json'
    if out.exists():raise RuntimeError('Model audit receipt already exists; not overwritten')
    atomic_json(out,d);print(json.dumps(d),flush=True)

if __name__=='__main__':main()
