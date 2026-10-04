"""Narrow frozen CPU gate: one metadata-first replan per task/model, 300 costs.

Uses saved GPU costs, exact actual planning images/actions/candidates, original
atol=rtol=1e-5, FP32, one CPU thread. No backend/precision/batch/seed retries.
"""
import argparse,os,sys,json,csv,hashlib,platform
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--r3-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--restored',type=Path,required=True);p.add_argument('--data-check',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'CPU_COST_RECOVERY_CHECK.json').exists():raise RuntimeError('Original CPU cost gate already exists; never overwrite or retry to seek PASS')
    selection=json.loads(a.selection.read_text());assert selection['atol']==selection['rtol']==1e-5 and selection['threads']==1 and selection['candidate_batch']==300
    data_check=json.loads(a.data_check.read_text());assert data_check['status']=='PASS_ALL_RECOVERED_MEMBERS_AND_ARRAY_IDENTITIES' and data_check['tasks']=={a.task:1200}
    os.environ['R3_ROOT']=str(a.r3_root.resolve());sys.path.insert(0,str(a.r3_root))
    import torch
    from torchvision import tv_tensors
    from r3.common import fp32_policy
    from r3.model import load_official,apply_delta,tensor_sha256
    from r3.data import action_processor,image_transform
    from .context import candidates_with_prefix
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy()
    def model_hash(m):return hashlib.sha256(json.dumps({k:tensor_sha256(v) for k,v in m.state_dict().items()},sort_keys=True).encode()).hexdigest()
    rows=[];summaries=[];predictions={}
    selected=[r for r in selection['rows'] if r['task']==a.task];assert len(selected)==4
    for item in selected:
        arm=item['arm'];directory=a.restored/(a.task+'_'+arm)/a.task/'FORMAL'/item['stream']/item['case_id']/arm
        source=directory/'compact_trajectory.npz';result_path=directory/'result.json';assert sha(result_path)==item['source_result_sha256']
        with np.load(source,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        assert arrays['replan_raw'].tolist()==[0,25]
        mapping={int(i):v for i,v in zip(arrays['planner_raw_indices'],arrays['planner_pixels'])}
        model=load_official(a.task,'cpu');assert model.r3_identity==item['provenance']['official']
        if arm!='H0':
            r=item['provenance']['checkpoint'];ck=a.r3_root/r['path'];assert ck.stat().st_size==r['bytes'] and sha(ck)==r['sha256'];apply_delta(model,ck)
        model.eval();model.requires_grad_(False);assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in model.parameters())
        before=model_hash(model);proc=action_processor(a.task);transform=image_transform()
        def prepare(pixels):
            return torch.stack([transform(tv_tensors.Image(np.transpose(v,(2,0,1)))) for v in pixels])[None]
        pix=prepare([mapping[i] for i in [15,20,25]]);goal=prepare([arrays['goal_pixels']])
        last=torch.from_numpy(proc.transform(arrays['raw_actions'][24:25]).reshape(1,1,2)).float()
        info={'pixels':pix[:,None].expand(1,300,*pix.shape[1:]),'goal':goal[:,None].expand(1,300,*goal.shape[1:]),'action':last[:,None].expand(1,300,1,2)}
        candidates=torch.from_numpy(arrays['last_generation_normalized'][1:2]);assert candidates.shape==(1,300,5,10) and candidates.dtype==torch.float32
        history_rows=[{'action':r} for r in arrays['raw_actions'][:25]]
        scored=candidates_with_prefix(info,candidates,history_rows,proc);assert scored.shape==(1,300,7,10)
        with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):cost=model.get_cost(info,scored).detach().numpy()[0].copy()
        reference=arrays['last_generation_cost'][1].copy();assert reference.shape==cost.shape==(300,)
        assert model_hash(model)==before and all(not p.requires_grad for p in model.parameters())
        diff=np.abs(cost.astype(np.float64)-reference.astype(np.float64));threshold=1e-5+1e-5*np.abs(reference.astype(np.float64));passed=np.isfinite(cost)&np.isfinite(reference)&(diff<=threshold)
        for i in range(300):rows.append({'task':a.task,'arm':arm,'case_id':item['case_id'],'stream':item['stream'],'replan_index':1,'candidate_index':i,
            'gpu_reference_cost':float(reference[i]),'cpu_cost':float(cost[i]),'absolute_difference':float(diff[i]),'unchanged_threshold':float(threshold[i]),'passed':bool(passed[i])})
        summaries.append({'task':a.task,'arm':arm,'case_id':item['case_id'],'stream':item['stream'],'candidates':300,'outliers':int((~passed).sum()),'max_abs_difference':float(diff.max()),
            'source':record(source),'result':record(result_path),'model_state_sha256_unchanged':before,'model_provenance':item['provenance']})
        predictions[arm+'_GPU_REFERENCE']=reference;predictions[arm+'_CPU_COST']=cost
        del model
    with (a.out/'CPU_COST_RAW.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.out/'CPU_COST_OUTLIERS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows([r for r in rows if not r['passed']])
    np.savez_compressed(a.out/'CPU_COST_VALUES.npz',**predictions)
    outliers=sum(not r['passed'] for r in rows)
    d={'status':'PASS_FROZEN_CPU_COST_TOLERANCE' if outliers==0 else 'FAIL_FROZEN_CPU_COST_TOLERANCE','task':a.task,'strict_cpu_cost_gate_passed':outliers==0,'outliers':outliers,
        'cases_model_stream_replans':4,'candidate_costs':1200,'scope':'One metadata-first actually visited replan1 for each of H0 and three fixed30k models; all300 saved last-generation CEM costs. Not all1200 trajectories or allCEM generations; no unsaved prediction-vector claim.',
        'atol':1e-5,'rtol':1e-5,'FP32':True,'TF32':False,'AMP':False,'threads':1,'candidate_batch':300,'new_training_updates':0,'new_CEM_optimizations':0,'new_simulator_steps':0,
        'release_gate_for_this_narrow_check':'PASS' if outliers==0 else 'HOLD','global_prior_H1a_HOLD_unaffected':True,'selection':record(a.selection),'data_recovery':record(a.data_check),
        'checks':summaries,'CPU_fingerprint':{'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
        'outputs':[record(a.out/n) for n in ['CPU_COST_RAW.csv','CPU_COST_OUTLIERS.csv','CPU_COST_VALUES.npz']],'producer':record(__file__),'context_source':record(Path(__file__).parent/'context.py')}
    with (a.out/'CPU_COST_RECOVERY_CHECK.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
    print(json.dumps({'task':a.task,'status':d['status'],'outliers':outliers,'candidate_costs':1200,'max_abs_difference':max(r['max_abs_difference'] for r in summaries)}),flush=True)

if __name__=='__main__':main()
