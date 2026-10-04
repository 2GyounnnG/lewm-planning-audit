"""Read-only verification of all fixed R4 S3 learned-control identities."""
import argparse,json,os,sys
from pathlib import Path
from r4.common import ARMS,STREAMS,sha256,file_record,atomic_json

def audit(r3_root,assets,task,control_root,out):
    os.environ['R3_ROOT']=str(r3_root);sys.path.insert(0,str(r3_root))
    from r3 import planning as p
    manifest=json.loads((assets/'CASE_WINDOWS.json').read_text())
    entries=[e for e in manifest['tasks'][task]['cases'] if e['case']['role']=='EVAL']
    assert len(entries)==100
    from r4 import runner
    source={n:sha256(Path(runner.__file__).parent/n) for n in ['runner.py','common.py','export_cases.py']}
    official_source=p.verify_official_sources();rows=[]
    for entry in entries:
      for stream in STREAMS:
       for arm in ARMS:
        folder=control_root/'FORMAL'/stream/entry['case']['case_id']/arm
        comp=json.loads((folder/'COMPLETE.json').read_text());result=json.loads((folder/'result.json').read_text())
        for n,expected in comp['files'].items():
            if file_record(folder/n)!=expected:raise RuntimeError('Control bytes changed: '+str(folder/n))
        starts=sorted(folder.glob('attempt_*/STARTED.json'))
        identities=[json.loads(s.read_text())['identity'] for s in starts]
        from r4.common import digest
        match=[v for v in identities if digest(v)==comp['identity_sha256']]
        assert len(match)>=1
        identity=match[-1]
        assert identity['case']==entry['case'] and identity['task']==task and identity['arm']==arm and identity['stream']==stream
        assert identity['code']==source and identity['source']==official_source and identity['phase']=='FORMAL'
        assert identity['plan']==p.PLAN_CONFIG and identity['cem']==p.CEM_CONFIG and identity['budget_raw']==50 and identity['optimizer_updates']==0
        assert identity['reranker'] is None and identity['provenance']['case_manifest']==sha256(assets/'CASE_WINDOWS.json')
        assert result['identity_sha256']==comp['identity_sha256'] and result['optimizer_updates']==0
        for idx,replan in enumerate(result['replans']):
            from r4.common import replan_seed
            assert replan['seed_uint64']==replan_seed(task,entry['case']['case_id'],idx,stream)
            assert replan['observed_history_frames']==1 and replan['anchor_raw']==idx*25
        rows.append({'task':task,'case_id':entry['case']['case_id'],'arm':arm,'stream':stream,
            'control_folder':str(folder),'identity_sha256':comp['identity_sha256'],'files':comp['files'],
            'complete':file_record(folder/'COMPLETE.json'),'provenance':identity['provenance']})
    receipt={'status':'PASS_CONTROL_REUSE_IDENTITY','task':task,'control_count':len(rows),'case_count':100,
        'r4_code_sha256':source,'official_source':official_source,'case_manifest':file_record(assets/'CASE_WINDOWS.json'),
        'controls':rows,'new_simulations':0,'optimizer_updates':0,'reused_control_mode':'H_POLICY','per_intervention_first_plan_gate':'BITWISE_EQUAL before first raw action'}
    atomic_json(out,receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='controls'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--r3-root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.r3_root,a.assets,a.task,a.control_root,a.out)
