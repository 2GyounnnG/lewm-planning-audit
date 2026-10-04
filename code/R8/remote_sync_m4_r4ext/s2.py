"""S2 initial forks: locked 64-candidate menus, four learning costs, sim truth."""
import argparse,csv,json,os,sys
from pathlib import Path
from .common import ARMS,atomic_json,atomic_npz,digest,file_record,verify_complete,sha256

def regenerate_initial(task,case,dataset_factory,model,processor,transform,device):
    import numpy as np
    import torch
    from r3 import planning as p
    from .replay import ReplayBranch
    from .common import replan_seed
    swm,solver_type,policy_type,_=p.load_official_api()
    timer=p._Timer(device)
    class Cost(p.AuditedCost):
        def get_cost(self,info,candidates):
            value=super().get_cost(info,candidates);self.last=candidates.detach().cpu().numpy().copy();return value
    cost=Cost(model,timer);solver=solver_type(model=cost,device=device,seed=replan_seed(task,case['case_id'],0),**p.CEM_CONFIG)
    policy=policy_type(solver=solver,config=swm.PlanConfig(**p.PLAN_CONFIG),process={'action':processor},transform={'pixels':transform,'goal':transform})
    result={};original=solver.solve
    def solve(info,init_action=None):
        value=original(info,init_action);result['returned']=value['actions'][0].numpy().copy();result['last_generation']=cost.last[0].copy();return value
    solver.solve=solve
    with ReplayBranch(task,case,dataset_factory()) as branch:
        policy.set_env(branch.world.envs)
        with torch.inference_mode():policy.get_action(p.planner_observation(branch.world.infos))
    return result

def run_initial(task,entry,assets,trajectories,models,processor,transform,output,device):
    import numpy as np
    from .candidates import build_menu
    from .export_cases import CaseWindows
    from .replay import run_branch,task_margin
    from .s1 import predict
    from .runner import encode_pixels
    from .replay import ReplayBranch
    case=entry['case'];folder=Path(output)/case['case_id'];proposals={}
    identity={'version':'R4_V23_S2_INITIAL_V2_MINIMAL_DEPENDENCY','task':task,'case':case,'exported_source_sha256':entry['sha256'],'optimizer_updates':0,'code':sha256(__file__)};identity_sha=digest(identity)
    done=verify_complete(folder,identity_sha)
    if done:return done
    factory=lambda:CaseWindows(assets,entry)
    verification=[]
    for arm in ARMS:
        fresh=regenerate_initial(task,case,factory,models[arm],processor,transform,device)
        proposals[arm]=fresh
        src=Path(trajectories)/'FORMAL/R3_ORIGINAL'/case['case_id']/arm if trajectories else None
        if src is None or not (src/'COMPLETE.json').exists():
            verification.append({'arm':arm,'status':'PENDING_LOCAL_RERUN_COMPARISON_NONBLOCKING'})
            continue
        receipt=json.loads((src/'COMPLETE.json').read_text())
        if file_record(src/'trajectory.npz')!=receipt['files']['trajectory.npz']:raise RuntimeError('Rerun source changed')
        with np.load(src/'trajectory.npz',allow_pickle=False) as f:
            old_return=f['returned_plans_normalized'][0].copy();old_last=f['last_generation_normalized'][0].copy()
        same_plan=bool(np.array_equal(fresh['returned'],old_return));same_generation=bool(np.array_equal(fresh['last_generation'],old_last))
        verification.append({'arm':arm,'status':'PASS' if same_plan and same_generation else 'FAIL','returned_plan_exact':same_plan,'last_generation_exact':same_generation,'max_plan_abs_difference':float(np.max(np.abs(fresh['returned']-old_return)))})
        if not same_plan or not same_generation:
            atomic_json(folder/'INITIAL_CEM_MISMATCH.json',{'status':'FAIL_INITIAL_CEM_REPRODUCTION','rows':verification})
            raise RuntimeError('Fresh initial-fork CEM does not reproduce local original-stream plans')
    menu=build_menu(task,case['case_id'],'initial',proposals,processor,device=device)
    folder.mkdir(parents=True,exist_ok=True)
    menu_record={'identity':identity,'menu_sha256':menu['menu_sha256'],'metadata':menu['metadata'],'kind':'S2_4x12_PLUS16','locked_before_any_simulator_query':True}
    if (folder/'MENU_LOCK.json').exists() and json.loads((folder/'MENU_LOCK.json').read_text())!=menu_record:raise RuntimeError('Locked candidate menu differs')
    atomic_npz(folder/'menu.npz',normalized_actions=menu['normalized'],raw_actions=menu['raw']);atomic_json(folder/'MENU_LOCK.json',menu_record)
    atomic_json(folder/'INITIAL_CEM_REPRODUCTION.json',{'rows':verification,'rerun_is_not_a_start_dependency':True})
    atomic_npz(folder/'initial_proposals.npz',**{arm+'_'+kind:value for arm,proposal in proposals.items() for kind,value in proposal.items()})
    with ReplayBranch(task,case,factory()) as branch:
        initial=branch.initial;initial_state=initial['state'];goal_state=branch.goal_state.copy()
        pixels=initial['pixels'];goal_pixels=branch.world.infos['goal'][0,-1].copy()
    z=encode_pixels(models['H0'],pixels[None],transform)[0];goal=encode_pixels(models['H0'],goal_pixels[None],transform)[0];learned={}
    for arm,model in models.items():
        prediction=predict(model,np.repeat(z[None,None],64,axis=0),menu['normalized'],5)[:,-1]
        learned[arm]={'prediction':prediction,'cost':np.sum((prediction-goal)**2,axis=1)}
    unique={};rows=[];sim_z=[];states=[]
    factory=lambda:CaseWindows(assets,entry)
    for i,meta in enumerate(menu['metadata']):
        if meta['alias_of'] is not None:result=unique[meta['alias_of']]
        else:
            path=folder/f'branch_{i:02d}.npz';rec=folder/f'branch_{i:02d}.json'
            if path.exists() and rec.exists():
                saved=json.loads(rec.read_text())
                if saved['menu_sha256']!=menu['menu_sha256'] or file_record(path)!=saved['array_file']:raise RuntimeError('Branch cache changed')
                with np.load(path,allow_pickle=False) as f:result={k:f[k].copy() for k in f.files}
            else:
                result=run_branch(task,case,factory,np.empty((0,2)),menu['raw'][i],models['H0'],transform)
                payload={k:v for k,v in result.items() if k not in ('terminal_pixels','scope')}
                atomic_npz(path,**payload);atomic_json(rec,{'menu_sha256':menu['menu_sha256'],'array_file':file_record(path),'scope':result['scope']})
            unique[i]=result
        terminal=result['terminal_latent'];sim_z.append(terminal);states.append(result['terminal_state']);jlat=float(np.sum((terminal-goal)**2))
        row={'task':task,'case_id':case['case_id'],'family_id':case['family_id'],'fork':'initial',**meta,
            'valid_fixed_horizon':bool(result['valid_fixed_horizon']),'missing_reason':str(result['missing_reason']),'candidate_raw_steps':int(result['candidate_raw_steps']),
            'J_sim_lat':jlat if result['valid_fixed_horizon'] else None,'J_sim_task':float(result['task_anytime']) if result['valid_fixed_horizon'] else None,'J_sim_task_terminal':float(result['task_terminal']) if result['valid_fixed_horizon'] else None,'J_sim_task_observed_prefix':float(result['task_anytime']),
            'success':int(result['any_success']),'initial_success':int(task_margin(task,initial_state,goal_state)<1),'scope':'PRIVILEGED_FULL_STATE_BRANCH_REFERENCE'}
        for arm in ARMS:
            row['J_'+arm]=float(learned[arm]['cost'][i]);row['latent_mse_'+arm]=float(np.mean((learned[arm]['prediction'][i]-terminal)**2)) if result['valid_fixed_horizon'] else None;row['cost_error_'+arm]=float(learned[arm]['cost'][i]-jlat) if result['valid_fixed_horizon'] else None
        rows.append(row)
    with (folder/'candidate_raw.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_npz(folder/'sim_truth.npz',terminal_latent=np.stack(sim_z),terminal_state=np.stack(states))
    result={'status':'COMPLETE' if all(r['valid_fixed_horizon'] for r in rows) else 'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON','valid_fixed_horizon_candidates':sum(r['valid_fixed_horizon'] for r in rows),'missing_reason':None if all(r['valid_fixed_horizon'] for r in rows) else 'TRUE_DMC_LAST_BEFORE_FIXED_HORIZON_PRESERVE_ALL64_NO_COMPLETE_CASE_SUBSET','task':task,'case_id':case['case_id'],'logical_candidates':64,'unique_branches':len(unique),'branch_raw_steps':sum(int(r['candidate_raw_steps']) for r in unique.values()),'has_successful_candidate':any(r['success'] for r in rows),'initial_distribution_success_rate':float(np.mean([r['success'] for r in rows if r['source']=='INITIAL_DISTRIBUTION'])),'identity_sha256':identity_sha,'optimizer_updates':0}
    atomic_json(folder/'result.json',result);atomic_json(folder/'COMPLETE.json',{'identity_sha256':identity_sha,'files':{n:file_record(folder/n) for n in ('candidate_raw.csv','sim_truth.npz','result.json','MENU_LOCK.json','menu.npz','initial_proposals.npz')}});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--trajectories');p.add_argument('--s0-a',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if json.loads(Path(a.s0_a).read_text())['status']!='PASS':raise RuntimeError('S2 initial requires S0(a)')
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from r3.model import load_official,apply_delta
    from r3.data import action_processor,image_transform
    from r3.common import fp32_policy
    fp32_policy();root=Path(a.r3_root);models={};routes=json.loads((root/'manifests/OPEN_LOOP_ROUTING.json').read_text())['tasks'][a.task]['arms']
    for arm in ARMS:
        models[arm]=load_official(a.task,a.device)
        if arm!='H0':
            route=next(r for r in routes if r.get('step')==30000 and r['seed']==int(arm.split('_')[1]));cp=root/route['checkpoint']['path']
            if file_record(cp)!={k:route['checkpoint'][k] for k in ('bytes','sha256')}:raise RuntimeError('Checkpoint bytes differ')
            apply_delta(models[arm],cp)
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];entries=[e for e in entries if e['case']['role']=='EVAL'][a.worker_index::a.workers]
    for entry in entries:print(json.dumps(run_initial(a.task,entry,a.assets,a.trajectories,models,action_processor(a.task),image_transform(),a.output,a.device)),flush=True)

if __name__=='__main__':main()
