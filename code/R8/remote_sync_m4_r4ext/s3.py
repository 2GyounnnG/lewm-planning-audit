"""Three fixed-menu closed-loop policies, each branches its own current state."""
import argparse,copy,json,os,sys
from pathlib import Path
from .common import STREAMS,atomic_json,atomic_npz,fixed_subset,digest,file_record
ROLES=('H0_MENU_RERANK','SIM_LAT_RERANK','SIM_TASK_RERANK')

class Reranker:
    def __init__(self,task,case,stream,role,model,processor,transform,dataset_factory,output,device):
        self.task,self.case,self.stream,self.role=task,case,stream,role
        self.model,self.processor,self.transform=model,processor,transform
        self.factory,self.output,self.device=dataset_factory,Path(output),device
    def __call__(self,result,info,last_generation,prefix,index):
        import numpy as np
        import torch
        from .candidates import build_menu
        from .replay import run_branch
        from .runner import encode_pixels
        fork=f'{self.stream}/replan_{index:02d}'
        menu=build_menu(self.task,self.case['case_id'],fork,{'H0':{'returned':result['actions'][0].numpy(),'last_generation':last_generation}},self.processor,kind='S3',device=self.device)
        out=self.output/f'replan_{index:02d}';out.mkdir(parents=True,exist_ok=True)
        lock={'menu_sha256':menu['menu_sha256'],'metadata':menu['metadata'],'prefix_sha256':digest(prefix.tolist()),'role':self.role,'anchor_raw':len(prefix),'locked_before_simulator_queries':True}
        if (out/'MENU_LOCK.json').exists() and json.loads((out/'MENU_LOCK.json').read_text())!=lock:raise RuntimeError('Menu/prefix changed on resume')
        atomic_npz(out/'menu.npz',normalized_actions=menu['normalized'],raw_actions=menu['raw']);atomic_json(out/'MENU_LOCK.json',lock)
        observations={k:info[k].detach().clone().to(self.device).unsqueeze(1).expand(1,64,*info[k].shape[1:]) for k in ('pixels','goal','action')}
        with torch.inference_mode():learned=self.model.get_cost(observations,torch.from_numpy(menu['normalized'][None]).to(self.device)).cpu().numpy()[0]
        valid=np.ones(64,dtype=bool);sim_lat=np.full(64,np.nan);sim_task=np.full(64,np.nan);terminal_task=np.full(64,np.nan);success=np.zeros(64,dtype=bool);branch_count=0;branch_raw=0;physics_steps=0;seconds=0.
        if self.role!='H0_MENU_RERANK':
            with torch.inference_mode():goal=self.model.encode({'pixels':info['goal'].detach().clone().to(self.device)})['emb'][0,-1].cpu().numpy()
            unique={}
            for i,meta in enumerate(menu['metadata']):
                if meta['alias_of'] is not None:b=unique[meta['alias_of']]
                else:
                    b=run_branch(self.task,self.case,self.factory,prefix,menu['raw'][i],self.model,self.transform);unique[i]=b;branch_count+=1;branch_raw+=b['candidate_raw_steps'];physics_steps+=b['physics_steps'];seconds+=b['seconds']
                valid[i]=b['valid_fixed_horizon'];sim_lat[i]=np.sum((b['terminal_latent']-goal)**2);sim_task[i]=b['task_anytime'];terminal_task[i]=b['task_terminal'];success[i]=b['any_success']
        if not valid.all():
            atomic_npz(out/'unavailable_raw.npz',valid_fixed_horizon=valid,J_sim_lat_observed_prefix=sim_lat,J_sim_task_observed_prefix=sim_task,success=success)
            atomic_json(out/'TECHNICALLY_UNEVALUABLE.json',{'status':'TECHNICALLY_UNEVALUABLE_FIXED_HORIZON','reason':'TRUE_DMC_LAST_BEFORE_FIXED_HORIZON','valid_candidates':int(valid.sum()),'logical_candidates':64,'no_remaining_candidate_subset_used':True,'branch_count':branch_count,'branch_raw_steps':branch_raw,'physics_steps_including_prefix':physics_steps})
            raise RuntimeError('TRUE_DMC_LAST_BEFORE_FIXED_HORIZON_SIM_RERANK_UNEVALUABLE')
        score={'H0_MENU_RERANK':learned,'SIM_LAT_RERANK':sim_lat,'SIM_TASK_RERANK':sim_task}[self.role]
        if not np.isfinite(score).all():raise RuntimeError('Nonfinite fixed-menu ranking')
        chosen=min(range(64),key=lambda i:(float(score[i]),menu['metadata'][i]['id']))
        record={'role':self.role,'selected_index':chosen,'selected_id':menu['metadata'][chosen]['id'],'selected_score':float(score[chosen]),'menu_sha256':menu['menu_sha256'],'branch_count':branch_count,'branch_raw_steps':branch_raw,'physics_steps_including_prefix':physics_steps,'prefix_replay_raw_steps':branch_count*len(prefix),'branch_seconds':seconds,'scope':'LEARNED_SAME_MENU_CONTROL' if self.role=='H0_MENU_RERANK' else 'PRIVILEGED_FULL_STATE_BRANCH_REFERENCE'}
        atomic_npz(out/'scores.npz',J_H0=learned,J_sim_lat=sim_lat,J_sim_task=sim_task,J_sim_task_terminal=terminal_task,success=success)
        atomic_json(out/'selection.json',record)
        returned=dict(result);returned['actions']=torch.from_numpy(menu['normalized'][chosen:chosen+1].copy())
        return returned,record

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3-root',required=True);p.add_argument('--assets',required=True);p.add_argument('--task',choices=['pusht','reacher'],required=True);p.add_argument('--output',required=True);p.add_argument('--role',choices=ROLES,required=True);p.add_argument('--stream',choices=STREAMS,required=True);p.add_argument('--s0-bcd');p.add_argument('--device',default='cuda');p.add_argument('--worker-index',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--reuse-root',help='read-only R4 S3 output root; completed case ids are excluded');p.add_argument('--case-limit',type=int,help='TECH-only prefix limit after reuse exclusion');a=p.parse_args()
    if a.role!='H0_MENU_RERANK':
        if not a.s0_bcd:raise ValueError('SIM requires same-host S0 b-d receipts')
        receipts=[json.loads(p.read_text()) for p in Path(a.s0_bcd).glob('*_S0_BCD.json')]
        if not receipts or any(r['status']!='PASS' for r in receipts):raise RuntimeError('S0 replay not established')
    os.environ['R3_ROOT']=str(Path(a.r3_root).resolve());sys.path.insert(0,a.r3_root)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from r3.model import load_official
    from r3.data import action_processor,image_transform
    from r3.common import fp32_policy
    from .runner import run_case
    from .export_cases import CaseWindows
    fp32_policy();model=load_official(a.task,a.device);processor=action_processor(a.task);transform=image_transform()
    original_cases=json.loads((Path(a.r3_root)/f'manifests/{a.task}_data_roles.json').read_text())['cases']['EVAL'];ids={c['case_id'] for c in original_cases}
    entries=json.loads((Path(a.assets)/'CASE_WINDOWS.json').read_text())['tasks'][a.task]['cases'];entries=[e for e in entries if e['case']['case_id'] in ids]
    reuse_ids=set()
    if a.reuse_root:
        reuse_root=Path(a.reuse_root)/'FORMAL'
        reuse_ids={q.parent.parent.name for q in reuse_root.glob('*/*/*/result.json') if q.parent.name in ROLES}
    entries=[e for e in entries if e['case']['case_id'] not in reuse_ids]
    if a.case_limit is not None: entries=entries[:a.case_limit]
    entries=entries[a.worker_index::a.workers]
    for entry in entries:
        case=entry['case'];folder=Path(a.output)/'FORMAL'/a.stream/case['case_id']/a.role
        factory=lambda:CaseWindows(a.assets,entry)
        reranker=Reranker(a.task,case,a.stream,a.role,model,processor,transform,factory,folder/'menus',a.device)
        result=run_case(model,a.task,case,factory(),processor,transform,arm=a.role,stream=a.stream,output=folder,provenance={'official':model.r3_identity,'subset_salt':'R4_CASE_SUBSET_V1','m4_extension':{'case_universe':'R4_S2_PUSHT_EVAL_100','reused_case_root':str(a.reuse_root) if a.reuse_root else None,'new_only':bool(a.reuse_root)}},device=a.device,reranker=reranker)
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
