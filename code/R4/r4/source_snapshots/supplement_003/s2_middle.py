"""Second-batch20 fixed mid-forks. Never launch before primary delivery."""
import argparse,csv,json,os,sys
from pathlib import Path
from .common import ARMS,atomic_json,atomic_npz,digest,file_record,fixed_subset,sha256,verify_complete,replan_seed

def propose(task,case,factory,prefix,model,processor,transform,device):
    import torch
    from r3 import planning as p
    from .replay import ReplayBranch
    swm,solver_type,policy_type,_=p.load_official_api();timer=p._Timer(device)
    class Cost(p.AuditedCost):
        def get_cost(self,info,candidates):
            value=super().get_cost(info,candidates);self.last=candidates.detach().cpu().numpy().copy();return value
    cost=Cost(model,timer);solver=solver_type(model=cost,device=device,seed=replan_seed(task,case['case_id'],0),**p.CEM_CONFIG)
    policy=policy_type(solver=solver,config=swm.PlanConfig(**p.PLAN_CONFIG),process={'action':processor},transform={'pixels':transform,'goal':transform})
    result={};original=solver.solve
    def solve(info,init_action=None):
        value=original(info,init_action);result.update(returned=value['actions'][0].numpy().copy(),last_generation=cost.last[0].copy());return value
    solver.solve=solve
    with ReplayBranch(task,case,factory()) as branch:
        branch.replay(prefix)
        if branch.true_terminated:raise RuntimeError('MID_FORK_TRUE_TERMINATED')
        policy.set_env(branch.world.envs)
        with torch.inference_mode():policy.get_action(p.planner_observation(branch.world.infos))
    return result

def run_case(task,entry,assets,trajectories,models,processor,transform,output,device):
    import numpy as np
    from .export_cases import CaseWindows
    from .replay import ReplayBranch,run_branch,task_margin
    from .runner import encode_pixels
    from .candidates import build_menu
    from .s1 import predict
    case=entry['case'];folder=Path(output)/case['case_id'];source=Path(trajectories)/'FORMAL/R3_ORIGINAL'/case['case_id']/'H0'
    if not (source/'COMPLETE.json').exists():raise RuntimeError('Missing original H0 log, cannot invent prefix')
    completed=json.loads((source/'COMPLETE.json').read_text());actual=file_record(source/'trajectory.npz')
    if actual!=completed['files']['trajectory.npz']:raise RuntimeError('Original H0 trajectory changed')
    identity={'version':'R4_S2_MIDDLE_V1','case':case,'fork_raw':10,'source_trajectory':actual,'source_windows_sha256':entry['sha256'],'code':sha256(__file__),'cem_seed':'R3 original first-replan seed reused for this counterfactual query, common across4models','warm_start':'none; raw10 is not an original replanning event','optimizer_updates':0};identity_sha=digest(identity)
    saved=verify_complete(folder,identity_sha)
    if saved:return saved
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
    reason='INSUFFICIENT_EXECUTED_PREFIX' if len(old['raw_actions'])<10 else 'SOURCE_TRUE_DMC_LAST_BY_RAW10' if task=='reacher' and old['step_success'][:10].any() else None
    if reason:
        result={'status':'UNAVAILABLE_MID_FORK','task':task,'case_id':case['case_id'],'family_id':case['family_id'],'fork':'middle_raw10','missing_reason':reason,'replacement_case':False,'identity_sha256':identity_sha}
        atomic_json(folder/'result.json',result);atomic_json(folder/'COMPLETE.json',{'identity_sha256':identity_sha,'files':{'result.json':file_record(folder/'result.json')}});return result
    prefix=old['raw_actions'][:10];dataset=CaseWindows(assets,entry);factory=lambda:dataset
    with ReplayBranch(task,case,factory()) as b:
        b.replay(prefix);fork=b.observe();goal_pixels=b.world.infos['goal'][0,-1].copy();goal_state=b.goal_state.copy()
    if not np.allclose(fork['state'],old['raw_physical_state'][10],atol=1e-5,rtol=1e-5):raise RuntimeError('MID_FORK_NEW_HOST_REPLAY_MISMATCH')
    proposals={arm:propose(task,case,factory,prefix,m,processor,transform,device) for arm,m in models.items()}
    menu=build_menu(task,case['case_id'],'middle_raw10',proposals,processor,device=device)
    lock={'identity':identity,'menu_sha256':menu['menu_sha256'],'metadata':menu['metadata'],'locked_before_truth':True}
    if (folder/'MENU_LOCK.json').exists() and json.loads((folder/'MENU_LOCK.json').read_text())!=lock:raise RuntimeError('Mid menu identity changed')
    atomic_npz(folder/'menu.npz',normalized_actions=menu['normalized'],raw_actions=menu['raw'],prefix=prefix);atomic_json(folder/'MENU_LOCK.json',lock)
    z=encode_pixels(models['H0'],fork['pixels'][None],transform)[0];goal=encode_pixels(models['H0'],goal_pixels[None],transform)[0]
    learned={arm:predict(m,np.repeat(z[None,None],64,axis=0),menu['normalized'],5)[:,-1] for arm,m in models.items()};unique={};rows=[]
    for i,meta in enumerate(menu['metadata']):
        if meta['alias_of'] is not None:b=unique[meta['alias_of']]
        else:
            path=folder/f'branch_{i:02d}.npz';receipt=folder/f'branch_{i:02d}.json'
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if r['menu_sha256']!=menu['menu_sha256'] or file_record(path)!=r['array_file']:raise RuntimeError('Mid branch cache differs')
                with np.load(path,allow_pickle=False) as f:b={k:f[k].copy() for k in f.files}
            else:
                b=run_branch(task,case,factory,prefix,menu['raw'][i],models['H0'],transform);atomic_npz(path,**{k:v for k,v in b.items() if k not in ('terminal_pixels','scope')});atomic_json(receipt,{'menu_sha256':menu['menu_sha256'],'array_file':file_record(path)})
            unique[i]=b
        valid=bool(b['valid_fixed_horizon']);jlat=float(np.sum((b['terminal_latent']-goal)**2))
        row={'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'fork':'middle_raw10',**meta,'valid_fixed_horizon':valid,'missing_reason':str(b['missing_reason']),'J_sim_lat':jlat if valid else None,'J_sim_task':float(b['task_anytime']) if valid else None,'J_sim_task_terminal':float(b['task_terminal']) if valid else None,'success':int(b['any_success']),'fork_success':int(task_margin(task,fork['state'],goal_state)<1),'candidate_raw_steps':int(b['candidate_raw_steps']),'scope':'PRIVILEGED_FULL_STATE_BRANCH_REFERENCE'}
        for arm in ARMS:
            jm=float(np.sum((learned[arm][i]-goal)**2));row['J_'+arm]=jm;row['latent_mse_'+arm]=float(np.mean((learned[arm][i]-b['terminal_latent'])**2)) if valid else None;row['cost_error_'+arm]=jm-jlat if valid else None
        rows.append(row)
    with (folder/'candidate_raw.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'COMPLETE' if all(r['valid_fixed_horizon'] for r in rows) else 'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON','task':task,'case_id':case['case_id'],'fork':'middle_raw10','logical_candidates':64,'unique_branches':len(unique),'branch_raw_steps':sum(int(v['candidate_raw_steps']) for v in unique.values()),'prefix_replay_raw_steps':len(unique)*10,'valid_fixed_horizon_candidates':sum(r['valid_fixed_horizon'] for r in rows),'has_successful_candidate':any(r['success'] for r in rows),'replacement_case':False,'identity_sha256':identity_sha}
    atomic_json(folder/'result.json',result);atomic_json(folder/'COMPLETE.json',{'identity_sha256':identity_sha,'files':{n:file_record(folder/n) for n in ('result.json','candidate_raw.csv','menu.npz','MENU_LOCK.json')}});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--trajectories',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);p.add_argument('--primary-complete',required=True);p.add_argument('--s0-bcd',required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if json.loads(Path(a.primary_complete).read_text()).get('status')!='MAIN_RESULTS_DELIVERED':raise RuntimeError('Second batch waits for actual primary result delivery')
    receipts=[json.loads(p.read_text()) for p in Path(a.s0_bcd).glob('*_S0_BCD.json')]
    if not receipts or any(r['status']!='PASS' for r in receipts):raise RuntimeError('Midfork requires S0 b-d')
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch
    from r3.model import load_official,apply_delta
    from r3.data import action_processor,image_transform
    from r3.common import fp32_policy
    torch.set_num_threads(1);torch.set_num_interop_threads(1);fp32_policy();root=Path(a.r3_root)
    cases=json.loads((root/f'manifests/{a.task}_data_roles.json').read_text())['cases']['EVAL'];ids={c['case_id'] for c in fixed_subset(a.task,cases)}
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];entries=[e for e in entries if e['case']['case_id'] in ids][a.worker_index::a.workers]
    routes=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms'];models={}
    for arm in ARMS:
        models[arm]=load_official(a.task,a.device)
        if arm!='H0':
            route=next(r for r in routes if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));cp=root/route['checkpoint']['path']
            if file_record(cp)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Fixed model changed')
            apply_delta(models[arm],cp)
    for entry in entries:print(json.dumps(run_case(a.task,entry,a.assets,a.trajectories,models,action_processor(a.task),image_transform(),a.output,a.device)),flush=True)

if __name__=='__main__':main()
