"""CPU recovery check of one fixed first middle fork, no simulator calls."""
import argparse,csv,json,os,sys
from pathlib import Path
from .common import ARMS,atomic_json,file_record,fixed_subset

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--secondary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.r3_root);os.environ['R3_ROOT']=str(root.resolve());sys.path.insert(0,str(root.resolve()))
    import numpy as np,torch
    from r3.common import fp32_policy
    from r3.model import load_official,apply_delta,tensor_sha256
    from .s1 import predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy();task='pusht'
    first=fixed_subset(task,json.loads((root/'manifests/pusht_data_roles.json').read_text())['cases']['EVAL'])[0]
    folder=Path(a.secondary)/'s2_middle'/first['case_id'];sealed=json.loads((folder/'COMPLETE.json').read_text())
    for name,record in sealed['files'].items():
        if file_record(folder/name)!=record:raise RuntimeError('Secondary source seal differs')
    result=json.loads((folder/'result.json').read_text())
    if result['status']!='COMPLETE':raise RuntimeError('First fixed fork unavailable: never replace it')
    raw=list(csv.DictReader((folder/'candidate_raw.csv').open()));lock=json.loads((folder/'MENU_LOCK.json').read_text())
    with np.load(folder/'fork_predictions.npz') as f:context={k:f[k].copy() for k in f.files}
    with np.load(folder/'menu.npz') as f:actions=f['normalized_actions'].copy()
    truths=[]
    for i,meta in enumerate(lock['metadata']):
        index=i if meta['alias_of'] is None else meta['alias_of']
        path=folder/f'branch_{index:02d}.npz';receipt=json.loads((folder/f'branch_{index:02d}.json').read_text())
        if file_record(path)!=receipt['array_file']:raise RuntimeError('Secondary branch receipt differs')
        with np.load(path) as f:truths.append(f['terminal_latent'].copy())
    truths=np.stack(truths);goal=context['goal_latent'];routes=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][task]['arms'];rows=[];checks=[]
    for arm in ARMS:
        model=load_official(task,'cpu')
        if arm!='H0':
            route=next(r for r in routes if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));cp=root/route['checkpoint']['path']
            if file_record(cp)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Fixed refit differs')
            apply_delta(model,cp)
        before={k:tensor_sha256(v) for k,v in model.state_dict().items()}
        prediction=predict(model,np.repeat(context['fork_latent'][None,None],64,axis=0),actions,5)[:,-1]
        after={k:tensor_sha256(v) for k,v in model.state_dict().items()}
        if before!=after:raise RuntimeError('Frozen state changed')
        checks.append({'arm':arm,'parameters_and_buffers_unchanged':True})
        for i,row in enumerate(raw):
            cost=float(np.sum((prediction[i]-goal)**2));mse=float(np.mean((prediction[i]-truths[i])**2))
            rows.append({'task':task,'case_id':first['case_id'],'arm':arm,'candidate_id':row['id'],'CPU_goal_cost':cost,'source_goal_cost':float(row['J_'+arm]),'goal_cost_abs_difference':abs(cost-float(row['J_'+arm])),
                'CPU_latent_mse':mse,'source_latent_mse':float(row['latent_mse_'+arm]),'latent_mse_abs_difference':abs(mse-float(row['latent_mse_'+arm])),
                'endpoint_max_abs_difference':float(np.max(np.abs(prediction[i]-context[arm+'_terminal_prediction'][i])))})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/'RECOVERED_MIDDLE_CPU_RAW.csv'
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt={'status':'COMPLETE_RECOVERY_COMPARISON','scope':'FIRST_METADATA_MIDDLE_FORK_64CANDIDATES_4MODELS','task':task,'case_id':first['case_id'],'candidate_model_cells':len(rows),'new_simulator_steps':0,'optimizer_updates':0,
        'model_checks':checks,'no_new_scientific_tolerance_or_gate':True,'metrics':{k:max(r[k] for r in rows) for k in ('goal_cost_abs_difference','latent_mse_abs_difference','endpoint_max_abs_difference')},'raw_table':file_record(path),
        'source_complete':file_record(folder/'COMPLETE.json'),'source_fork_predictions':file_record(folder/'fork_predictions.npz'),'source_menu':file_record(folder/'menu.npz')}
    atomic_json(out/'RECOVERY_CHECK.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
