"""Four original Cube TECH cases using retained exact source endpoints only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import old_history_runner as h
from x1 import core

class EndpointView(h.X1View):
    def load_chunk(self,episodes,starts,ends):
        c=self.case
        assert list(episodes)==[c['source_episode_idx']] and list(starts)==[c['start_raw_index']] and list(ends)==[c['goal_raw_index']+1]
        # The pinned World._extract_init_goal reads only rows0 and-1, and video=None.
        # Return the two exact retained source endpoints; do not invent 24 frames.
        assert self.raw.entry['format']=='EXACT_ENDPOINTS_ONLY'
        out={k:torch.from_numpy(self.raw.arrays[k].copy()) for k in self.column_names}
        assert all(len(v)==2 for v in out.values());self.reads+=1;assert self.reads==1
        return [out]

def main():
    p=argparse.ArgumentParser();p.add_argument('--prereg',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);h.fp32_policy();core.policy();core.R3_ROOT=h.ROOT/'shared_data/r3'
    pr=json.loads(Path(a.prereg).read_text());bind=h.file_record(Path(a.prereg))['sha256']
    for n,v in pr['execution_code_sha256'].items():assert h.file_record(Path(__file__).parent/n)['sha256']==v
    for k,n in [('manifest_sha256','CUBE_TECH_MANIFEST.json'),('baseline_sha256','CUBE_TECH_BASELINE.json'),('plans_sha256','CUBE_TECH_FIRST_PLANS.npz')]:assert h.file_record(h.ROOT/'inputs'/n)['sha256']==pr[k]
    entries=json.loads((h.ROOT/'inputs/CUBE_TECH_MANIFEST.json').read_text())['cases'];refs=json.loads((h.ROOT/'inputs/CUBE_TECH_BASELINE.json').read_text());plans=np.load(h.ROOT/'inputs/CUBE_TECH_FIRST_PLANS.npz')['plans']
    h.X1View=EndpointView;checks=[]
    for entry,ref in zip(entries,refs):
        assert entry['case']['case_id']==ref['case_id'];model,scale=h.load('cube',ref['arm'],'cuda:0');before=h.model_hash(model);base=Path(a.output)/'TECH'/ref['stream']/ref['case_id']/ref['arm']
        ok=h.run(model,scale,'cube',entry,h.ROOT/'assets/CUBE_TECH_ENDPOINTS',base/'H_POLICY',ref['arm'],ref['stream'],'H_POLICY',None,ref,'TECH',bind)
        if ok:
            first=np.load(base/'H_POLICY/trajectory.npz')['returned_plans_normalized'][0]
            ok=h.run(model,scale,'cube',entry,h.ROOT/'assets/CUBE_TECH_ENDPOINTS',base/'H_REAL3_REPLAN',ref['arm'],ref['stream'],'H_REAL3_REPLAN',first,ref,'TECH',bind)
            checks.append(dict(case_id=ref['case_id'],arm=ref['arm'],paired_gate=ok,cross_version_first_plan_equal=first.tobytes()==plans[ref['plan_index']].tobytes(),cross_version_max_abs_difference=float(np.max(abs(first-plans[ref['plan_index']])))))
        else:checks.append(dict(case_id=ref['case_id'],arm=ref['arm'],paired_gate=False))
        assert h.model_hash(model)==before;del model;torch.cuda.empty_cache()
    result=dict(status='PASS' if all(x['paired_gate'] for x in checks) and len(checks)==4 else 'FAIL',checks=checks,original_tech_cases=4,planned_trajectories=8)
    h.atomic_json(Path(a.output)/'TECH_GATE.json',result);print(json.dumps(result));assert result['status']=='PASS'
if __name__=='__main__':main()
