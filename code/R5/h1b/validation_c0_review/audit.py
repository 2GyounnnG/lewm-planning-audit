"""Independent read-only engineering audit; no neural fit or simulation."""
import csv,datetime,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent;CODE=HERE.parent;BASE=CODE.parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(CODE))
from common import read,sha,atomic,file,fold
from fit import ridge

def main():
    freeze=read(CODE/'CODE_FREEZE.json');protocol=read(CODE/'FIT_PROTOCOL.json')
    for name,h in freeze['files'].items():assert sha(CODE/name)==h,('Frozen code mismatch',name)
    anchors={};rolechecks={}
    for task in ('pusht','reacher','tworoom','cube'):
        if task in ('pusht','reacher'):
            r=Path('/Volumes/MyProj/r3_official_lewm_predictor_refit/recovery')
            rp=r/'manifests'/f'{task}_data_roles.json';ip=r/'artifacts/open_loop'/task/'inputs.npz';tp=ip.with_name('targets.npz')
        else:
            r=BASE/'r4_v23_execution/x1/recovery'/task/task
            rp=r/'manifests'/f'{task}_data_roles.json';ip=r/'replay/fixed_open_loop_inputs.npz';tp=ip
        roles=read(rp);cases=roles['cases']['EVAL'];train=[e for e in roles['episodes'] if e['role']=='REFIT_TRAIN']
        assert len(cases)==100 and {e['episode_id'] for e in train}.isdisjoint({c['episode_id'] for c in cases})
        hold=sorted(train,key=lambda e:hashlib.sha256(f"R5_H1B_HOLDOUT_V1/{task}/{e['episode_id']}".encode()).digest())[:int(np.ceil(.1*len(train)))]
        for ep in train:
            expected=int.from_bytes(hashlib.sha256(f"R5_H1B_GROUP_CV_V1/{task}/episode/{ep['episode_id']}".encode()).digest()[:8],'big')%5
            assert expected==fold(ep,task)
        rolechecks[task]={'train_episodes':len(train),'train_frames':sum(e['length'] for e in train),'holdout_episodes':len(hold),
           'holdout_episode_ids_sha256':hashlib.sha256(json.dumps(sorted(e['episode_id'] for e in hold)).encode()).hexdigest(),
           'fold_episode_counts':[sum(fold(e,task)==j for e in train) for j in range(5)],'train_eval_episode_disjoint':True,'roles':file(rp)}
        with np.load(ip) as inputs,np.load(tp) as targets:
            ids=[str(x) for x in inputs['case_ids']];assert ids==[c['case_id'] for c in cases]
            for j,c in enumerate(cases):
                t=c['open_loop_anchor_raw']
                np.testing.assert_array_equal(inputs['history_raw_indices'][j],t+np.array([-10,-5,0]))
                np.testing.assert_array_equal(inputs['action_raw_indices'][j],np.arange(t-10,t+25))
                np.testing.assert_array_equal(targets['target_raw_indices'][j],t+5*np.arange(1,6))
            assert inputs['initial_z'].shape==(100,3,192) and targets['target_z'].shape==(100,5,192)
            anchors[task]={'cases':100,'original_metadata_order_identical':True,'history_indices_exact':True,'future_indices_exact':True,
               'input_dtype':str(inputs['initial_z'].dtype),'target_dtype':str(targets['target_z'].dtype),'inputs':file(ip),'targets':file(tp)}
    # Check grouped-CV weighting against an explicit per-fold standardized design.
    rng=np.random.default_rng(260104);sizes=[5,7,11,13,17];labels=np.repeat(np.arange(5),sizes);n=len(labels)
    x=rng.normal(size=(n,6));x[:,-1]=3.;y=rng.normal(size=(n,4))+x[:,:4]*np.array([.2,1.,3.,8.])
    folder=HERE/'synthetic_ridge';folder.mkdir(parents=True,exist_ok=True);atomic(folder/'manifest.json',{'synthetic':True,'rows':n,'fold_sizes':sizes})
    ds=SimpleNamespace(anchors=np.arange(n),folds=labels,h=1,path=folder,xy=lambda rows,residual=False:(x[rows],y[rows]))
    args=SimpleNamespace(technical=True);ridge(args,ds,folder);actual=read(folder/'fit.json');expected=[]
    for alpha in protocol['cv']['alphas']:
        loss=0.
        for f in range(5):
            tr=labels!=f;va=~tr;xx=x[tr];yy=y[tr];mx=xx.mean(0);sc=xx.std(0);sc[sc==0]=1
            design=(xx-mx)/sc;coef=np.linalg.solve(design.T@design+sum(tr)*alpha*np.eye(x.shape[1]),design.T@(yy-yy.mean(0)))
            prediction=((x[va]-mx)/sc)@coef+yy.mean(0)
            loss+=((prediction-y[va])**2).sum(0).mean()
        expected.append(loss/n)
    np.testing.assert_allclose(actual['mean_cv_losses'],expected,atol=2e-12,rtol=2e-12)
    selection=protocol['cv']['alphas'][max(i for i,v in enumerate(expected) if v==min(expected))]
    assert actual['alpha']==selection
    out={'status':'PASS_ENGINEERING_REVIEW_WITH_REPORT_PENDING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'scope':'Read-only frozen implementation audit plus synthetic closed-form arithmetic; zero neural optimizer updates, zero simulator calls; no owner file edited',
       'code_freeze':file(CODE/'CODE_FREEZE.json'),'fit_protocol':file(CODE/'FIT_PROTOCOL.json'),'roles':rolechecks,'eval_anchor_check':anchors,
       'independent_cv':{'unequal_fold_sizes':sizes,'alphas_checked':len(expected),'selected_alpha':selection,'max_abs_loss_difference':float(np.max(np.abs(np.asarray(expected)-actual['mean_cv_losses']))),'optimizer_updates':0},
       'verified_static':[
         'prepare selects REFIT_TRAIN only; all episode rows share grouped folds and held-out membership',
         'MLP means/scales use fitting90% rows only; held-out labels are used only for validation',
         'Single/three latent inputs share eligible anchors and chronological 5h recorded actions; no future frame in context',
         '2x256 GELU FP32, Adam1e-3 cosine horizon2000, batch512, no dropout/weight decay, seeds0/1/2, CPU-only',
         'Latent MLP standardizes future-minus-current residual, inversely transforms then adds current latent',
         'Every100 full-holdout validation; strict improvement; patience5; best checkpoint; stale>=5 resume executes no more updates',
         'Resume binds model/optimizer/sampler RNG/step/best/stale/logs to source/matrix/protocol identity',
         'EVAL is loaded only after fit parameters/checkpoint is saved; latent reporting casts prediction/target FP32 before MSE',
         'Original frozen encoder with no grad/optimizer, parameter/buffer version and final SHA guards',
         'Supervisor schedules only H1b fits, no H2/H3/C1/C2, max16 workers (<=64 MLP threads) within112-191'],
       'pending':['Three-seed error/R2 mean and H3-quantity reporting require audit of the not-yet-written report implementation; current fit keeps all seed predictions separately.'],
       'nonblocking_hardening':['Existing COMPLETE/manifest short-circuit trusts its original identity at worker entry; supervisor freeze and immutable run directories currently provide the guard. No evidence of wrong reuse found.'],
       'substantive_findings':[],'review_code':file(__file__)}
    atomic(HERE/'ENGINEERING_REVIEW.json',out);print(json.dumps({'status':out['status'],'tasks':4,'eval_cases':400,'cv_selected_alpha':selection,'cv_max_abs':out['independent_cv']['max_abs_loss_difference'],'findings':0}),flush=True)

if __name__=='__main__':main()
