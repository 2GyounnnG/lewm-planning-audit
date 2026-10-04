"""Count sealed primary outputs; never label absent jobs as zero-valued outcomes."""
import argparse,csv,json
from pathlib import Path
from .common import ARMS,STREAMS,atomic_json,file_record

def summarize(line,task):
    line=Path(line);logs=[]
    for path in (line/'trajectories/FORMAL').glob('*/*/*/COMPLETE.json'):
        receipt=json.loads(path.read_text())
        for name,record in receipt['files'].items():
            if file_record(path.parent/name)!=record:raise RuntimeError('Sealed output differs: '+str(path))
        result=json.loads((path.parent/'result.json').read_text());logs.append(result)
    modules={};limitation=line/'SIM_TECHNICAL_LIMITATION.json';limitations=[]
    for name,arms,streams,expected in [('original400',ARMS,('R3_ORIGINAL',),400),('alternate800',ARMS,STREAMS[1:],800),('H0_MENU_RERANK',('H0_MENU_RERANK',),STREAMS,60),('SIM_LAT_RERANK',('SIM_LAT_RERANK',),STREAMS,60),('SIM_TASK_RERANK',('SIM_TASK_RERANK',),STREAMS,60)]:
        actual=sum(r['arm'] in arms and r['stream'] in streams for r in logs)
        if actual>expected:raise RuntimeError('More than registered primary records for '+name)
        row={'expected':expected,'completed':actual,'status':'COMPLETE' if actual==expected else 'IN_PROGRESS','all_completed_output_hashes_verified':True}
        if name.startswith('SIM_') and limitation.exists():
            reason=json.loads(limitation.read_text());row.update(completed=None,observed_completed_records=actual,status='TECHNICALLY_UNEVALUABLE',reason=reason['reason'],gate={'path':str(limitation),'file':file_record(limitation)})
            limitations.append({'module':name,'reason':reason['reason'],'gate':row['gate']})
        modules[name]=row
    for kind in ('offline','cross'):
        path=line/'s1'/('S1_'+kind+'_COMPLETE.json');raw=line/'s1'/('S1_'+kind+'_raw.csv')
        row={'expected':2400 if kind=='offline' else 'ALL_REGISTERED4MODEL_BY4SOURCE_LEGAL_AND_MISSING_ANCHORS','completed':None,'status':'IN_PROGRESS'}
        if path.exists():
            receipt=json.loads(path.read_text());actual=sum(1 for _ in csv.DictReader(raw.open()))
            if actual!=receipt['rows']:raise RuntimeError('S1 row count differs')
            row.update(completed=actual,valid=receipt['valid'],status='COMPLETE',raw_table={'path':str(raw),'file':file_record(raw)},receipt={'path':str(path),'file':file_record(path)})
        modules['S1_'+kind]=row
    forks=[]
    for path in (line/'s2').glob('*/COMPLETE.json'):
        for name,record in json.loads(path.read_text())['files'].items():
            if file_record(path.parent/name)!=record:raise RuntimeError('S2 sealed file differs')
        forks.append(json.loads((path.parent/'result.json').read_text()))
    missing=sum(r['status']=='TECHNICALLY_UNEVALUABLE_FIXED_HORIZON' for r in forks)
    row={'expected':100,'completed':len(forks),'status':'COMPLETE' if len(forks)==100 else 'IN_PROGRESS','forks_with_technical_missing_horizon':missing,'candidate_expected':6400,'candidate_completed':sum(r['logical_candidates'] for r in forks),'valid_fixed_horizon_candidates':sum(r['valid_fixed_horizon_candidates'] for r in forks)}
    if len(forks)==100 and missing:
        row.update(status='COMPLETE_WITH_TECHNICAL_LIMITATIONS',reason='TRUE_DMC_LAST_PREVENTS_COMMON_FIXED_HORIZON; all100forks and64candidate weights retained; complete-case ranking is not a full-task result')
        gate=line/'s0/TERMINATION_MENU_TECH_RAW.json'
        if not gate.exists():gate=line/'s0/TERMINATION_TECH_RAW.json'
        row['gate']={'path':str(gate),'file':file_record(gate)};limitations.append({'module':'S2_initial','reason':row['reason'],'gate':row['gate']})
    modules['S2_initial']=row
    complete=all(r['status'] in ('COMPLETE','TECHNICALLY_UNEVALUABLE','COMPLETE_WITH_TECHNICAL_LIMITATIONS') for r in modules.values())
    result={'version':'R4_V23_PRIMARY_MODULE_STATUS_V1','task':task,'status':('COMPLETE_WITH_TECHNICAL_LIMITATIONS' if limitations else 'COMPLETE') if complete else 'IN_PROGRESS','modules':modules,'technical_limitations':limitations,'absent_jobs_are_not_scientific_zeroes':True,'new_neural_training':0,'second_batch_excluded':True}
    atomic_json(line/'MAIN_MODULE_STATUS.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--line',required=True);p.add_argument('--task',required=True,choices=['pusht','reacher']);a=p.parse_args();print(json.dumps(summarize(a.line,a.task)),flush=True)

if __name__=='__main__':main()
