"""Flatten sealed S0 evidence without changing pass/fail or timing values."""
import argparse,csv,json
from pathlib import Path
from deliver import record,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];timings=[];inputs=[]
    path=source/'S0_A.json';receipt=record(path);inputs.append(receipt);data=json.loads(path.read_text())
    for r in data['rows']:rows.append({**r,'check':'S0_A','status':data['status'],'source_sha256':receipt['sha256']})
    for path in sorted(source.glob('*_S0_BCD.json')):
        receipt=record(path);inputs.append(receipt);data=json.loads(path.read_text())
        rows.append({**{k:data[k] for k in ('task','case_id','check','status')},**data['checks'],'source_sha256':receipt['sha256']})
        for i,seconds in enumerate(data['branch_seconds']):timings.append({**{k:data[k] for k in ('task','case_id')},'branch_index':i,'branch_seconds':seconds,'source_sha256':receipt['sha256']})
    if len(rows)!=8 or len(timings)!=16:raise ValueError('Expected four fixed TECH cases and all sixteen branch checks')
    for name,values in [('S0_CHECKS_RAW.csv',rows),('S0_BRANCH_TIMING_RAW.csv',timings)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(dict.fromkeys(k for r in values for k in r)));w.writeheader();w.writerows(values)
    atomic(out/'S0_MODULE_STATUS.json',{'status':'COMPLETE','all_checks_pass':all(r['status']=='PASS' for r in rows),'check_rows':len(rows),'branch_timing_rows':len(timings),'inputs':inputs,'outputs':[record(out/name) for name in ('S0_CHECKS_RAW.csv','S0_BRANCH_TIMING_RAW.csv')],'fixed_horizon_termination_is_separate_gate':True})

if __name__=='__main__':main()
