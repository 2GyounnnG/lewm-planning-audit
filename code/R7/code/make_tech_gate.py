from __future__ import annotations
import json, sys
from pathlib import Path

def main(root, task):
    root=Path(root); selection=json.loads((root/'ops'/f'R7_{task.upper()}_CASE_SELECTION.json').read_text())
    cases=selection['tech_cases']; rows=[]; problems=[]
    for c in cases:
        base=root/'raw'/task/'TECH'/'R3_ORIGINAL'/c['case_id']/'H0'
        for hist in ('H_POLICY','H_REAL3_REPLAN'):
            p=base/hist/'result.json'
            if not p.exists(): problems.append(f'missing:{p}'); continue
            d=json.loads(p.read_text()); rows.append(d)
            if d.get('status')!='COMPLETE' or d.get('method_failure') is not None: problems.append(f'incomplete:{p}')
            if d.get('executed_raw_steps',0)<=0: problems.append(f'no_steps:{p}')
        q=base/'H_REAL3_REPLAN'/'result.json'
        if q.exists() and not json.loads(q.read_text()).get('paired_first_plan_checked'): problems.append(f'pair_missing:{q}')
    result={'version':'R7_TECH_GATE_V1','evidence_label':'PREREGISTERED_STRESS_TEST_FRESH_CASES','task':task,
            'status':'PASS' if len(rows)==len(cases)*2 and not problems else 'BLOCKED',
            'tech_cases':len(cases),'trajectories':len(rows),'expected_trajectories':len(cases)*2,
            'problems':problems,'optimizer_updates':0,'rows':rows}
    out=root/'ops'/f'R7_{task.upper()}_TECH_GATE.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'task':task,'status':result['status'],'trajectories':len(rows),'problems':len(problems)}))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
