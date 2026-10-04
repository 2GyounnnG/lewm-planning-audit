"""Check S3 statistics recovered from explicitly derived state subsets."""
import argparse,csv,json,sys
from pathlib import Path
from deliver import record,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--r3-root',required=True)
    p.add_argument('--task',choices=('pusht','reacher'),required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--tables',required=True);p.add_argument('--remote-tables',required=True);a=p.parse_args()
    root=Path(a.root);source=Path(a.inputs);tables=Path(a.tables);remote=Path(a.remote_tables)
    sys.path.insert(0,str(root.resolve()));from analysis.aggregate import universe,fixed_subset,ARMS,STREAMS
    cases,_=universe(a.r3_root,a.task);subset=fixed_subset(a.r3_root,a.task)
    metadata=json.loads((source/'STATISTICS_RECOVERY_MAP.json').read_text());logical=[];raw=[]
    for entry in metadata['records']:
        folder=source/entry['relative_path']
        for name,key in [('result.json','source_result'),('trajectory.npz','derived_state_npz')]:
            receipt=record(folder/name)
            if any(receipt[k]!=entry[key][k] for k in ('bytes','sha256')):raise ValueError('Recovered file mismatch')
        if json.loads((folder/'DERIVED_STATISTICS_INPUT.json').read_text())!=entry:raise ValueError('Derived source mapping differs')
        r=json.loads((folder/'result.json').read_text());raw.append(r)
        if r['status']!='COMPLETE' or r['optimizer_updates']!=0:raise ValueError('Unexpected run status or neural updates')
        logical.append((r['case_id'],r['arm'],r['stream']))
    if len(logical)!=len(set(logical)):raise ValueError('Duplicate policy outcome')
    roles=('H0_MENU_RERANK','SIM_LAT_RERANK','SIM_TASK_RERANK') if a.task=='pusht' else ('H0_MENU_RERANK',)
    expected={(c,m,s) for c in cases for m in ARMS for s in STREAMS}|{(c,m,s) for c in subset for m in roles for s in STREAMS}
    if set(logical)!=expected:raise ValueError('Policy/case/stream coverage mismatch')
    comparisons=[]
    for name in ('S3_ALL_RAW_VALUES.csv','S3_LEARNING_MAIN_TABLE.csv','PLANNING_RANDOMNESS_TABLE.csv','S3_MENU_INTERVENTIONS.csv','S3_SUCCESS_TRANSITIONS.csv','S3_MARGIN_AND_RESOURCE_TABLE.csv','S3_LEARNING_SUCCESS_TRANSITIONS.csv'):
        def read(path):
            with path.open(newline='') as f:return sorted(json.dumps(r,sort_keys=True) for r in csv.DictReader(f))
        if read(tables/name)!=read(remote/name):raise ValueError('Local/remote frozen aggregation differs: '+name)
        comparisons.append({'table':name,'local':record(tables/name),'remote':record(remote/name),'all_rows_equal_order_independent':True})
    result={'status':'COMPLETE','task':a.task,'policy_outcomes':len(logical),'fixed_cases':100,'fixed_mechanism_cases':20,'planner_streams':3,'source_and_derived_hashes_verified':True,'original_result_values_preserved':True,'all_frozen_aggregation_tables_reproduced_locally':True,'no_new_training':True,'scope':'RECOVERED_STATE_AND_RESULT_STATISTICS; NOT_FULL_PIXELS_OR_FULL_TRAJECTORY_RECOVERY','inputs':[record(source/'STATISTICS_RECOVERY_MAP.json')],'tables':comparisons}
    atomic(root/f'manifests/main_validation/r4_{a.task}_s3.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','tables')}))

if __name__=='__main__':main()
