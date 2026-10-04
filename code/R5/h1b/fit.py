"""TRAIN-only closed-form ridge and explicitly authorized CPU measurement MLP."""
import argparse, csv, math, os, time
from pathlib import Path
import numpy as np
from common import *
from dataset import Dataset,evaluation

def moments(ds,rows=None,residual=False,batch=8192):
    rows=np.arange(len(ds.anchors)) if rows is None else np.asarray(rows);out=None
    for start in range(0,len(rows),batch):
        x,y=ds.xy(rows[start:start+batch],residual);x=x.astype(np.float64);y=y.astype(np.float64)
        if out is None:out={'n':0,'sx':np.zeros(x.shape[1]),'sy':np.zeros(y.shape[1]),'xx':np.zeros((x.shape[1],x.shape[1])),'xy':np.zeros((x.shape[1],y.shape[1])),'yy':np.zeros(y.shape[1])}
        out['n']+=len(x);out['sx']+=x.sum(0);out['sy']+=y.sum(0);out['xx']+=x.T@x;out['xy']+=x.T@y;out['yy']+=(y*y).sum(0)
    if out is None:raise ValueError('Empty eligible training partition')
    return out

def ridge_path(m,alphas):
    n=m['n'];mx=m['sx']/n;my=m['sy']/n;cov=m['xx']-n*np.outer(mx,mx);v=np.maximum(np.diag(cov)/n,0);scale=np.sqrt(v);scale[scale==0]=1
    cc=cov/scale[:,None]/scale[None,:];cc=(cc+cc.T)/2;ev,u=np.linalg.eigh(cc);rhs=(m['xy']-n*np.outer(mx,my))/scale[:,None];proj=u.T@rhs
    result=[]
    for alpha in alphas:
        coef=(u@(proj/(np.maximum(ev,0)+n*alpha)[:,None]))/scale[:,None];intercept=my-mx@coef;result.append((coef,intercept))
    return result,{'min_covariance_eigenvalue':float(ev.min()),'constant_features':int((v==0).sum())}

def sse(m,w,b):
    return np.maximum(m['yy']-2*(w*m['xy']).sum(0)-2*b*m['sy']+(w*(m['xx']@w)).sum(0)+2*b*(m['sx']@w)+m['n']*b*b,0)

def save_eval(a,ds,predict,out,fitfile):
    x,y,z,cases,labels=evaluation(a.task,a.target,a.history);pred_raw=predict(x,z);pred=pred_raw.astype(np.float32) if ds.h else pred_raw.astype(np.float64);y=y.astype(np.float32) if ds.h else y.astype(np.float64);assert pred.shape==y.shape and np.isfinite(pred).all()
    npz(out/'evaluation.npz',x=x,y=y,pred=pred,pred_model_raw=pred_raw,current_z=z,case_ids=np.asarray([c['case_id'] for c in cases]),labels=np.asarray(labels))
    rows=[];sources={'input_matrix_manifest_sha256':sha(ds.path/'manifest.json'),'protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'fit_sha256':sha(fitfile)}
    for i,c in enumerate(cases):
        if ds.h:
            rows.append({**c,'task':a.task,'model':a.model,'history':a.history,'target':a.target,'seed':a.seed if a.model=='MLP' else None,'mse':float(np.mean((pred[i]-y[i])**2)),**sources})
        else:
            for j,label in enumerate(labels):rows.append({**c,'task':a.task,'model':a.model,'history':a.history,'target':a.target,'seed':a.seed if a.model=='MLP' else None,'component':label,'truth':float(y[i,j]),'prediction':float(pred[i,j]),'squared_error':float((pred[i,j]-y[i,j])**2),**sources})
    atomic(out/'case_values.json',rows)
    with (out/'case_values.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return {'rows':len(rows),'cases':len(cases),'raw':file(out/'case_values.json'),'predictions':file(out/'evaluation.npz')}

def ridge(a,ds,out):
    t=time.monotonic();protocol=read(Path(__file__).with_name('FIT_PROTOCOL.json'));alphas=protocol['cv']['alphas'];folds=[moments(ds,np.flatnonzero(ds.folds==f)) for f in range(5)];total={k:sum(f[k] for f in folds) for k in folds[0]};loss=np.zeros(len(alphas));records=[]
    for f,valid in enumerate(folds):
        train={k:total[k]-valid[k] for k in total};models,info=ridge_path(train,alphas);var=np.maximum(train['yy']/train['n']-(train['sy']/train['n'])**2,0);valid_components=var>0
        for j,(w,b) in enumerate(models):
            err=sse(valid,w,b);criterion=float(err.mean()) if ds.h else float(np.mean(err[valid_components]/var[valid_components])) if valid_components.any() else 0.
            loss[j]+=criterion;records.append({'fold':f,'alpha':alphas[j],'validation_n':valid['n'],'validation_loss_sum':criterion,'train_n':train['n'],**info})
    loss/=total['n'];minimum=float(loss.min());chosen=max(i for i,v in enumerate(loss) if v==minimum);final,info=ridge_path(total,[alphas[chosen]]);w,b=final[0]
    npz(out/'fit.npz',coefficient=w,intercept=b,alpha=np.array(alphas[chosen]));atomic(out/'fit.json',{'status':'FITTED_BEFORE_EVAL_ACCESS','alpha':alphas[chosen],'cv':records,'mean_cv_losses':loss.tolist(),'train_n':total['n'],'protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'input_matrix_manifest_sha256':sha(ds.path/'manifest.json'),'seconds':time.monotonic()-t,'optimizer_updates':0,**info})
    if a.technical:return {'status':'TECHNICAL_COMPLETE','train_n':total['n'],'seconds':time.monotonic()-t,'alpha':alphas[chosen],'optimizer_updates':0}
    ev=save_eval(a,ds,lambda x,z:x@w+b,out,out/'fit.npz');return {'status':'COMPLETE','train_n':total['n'],'seconds':time.monotonic()-t,'alpha':alphas[chosen],'optimizer_updates':0,**ev}

def mlp(a,ds,out):
    import torch
    from torch import nn
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    tr=np.flatnonzero(~ds.holdout);va=np.flatnonzero(ds.holdout);residual=bool(ds.h);m=moments(ds,tr,residual);mx=m['sx']/m['n'];my=m['sy']/m['n'];sx=np.sqrt(np.maximum(np.diag(m['xx'])/m['n']-mx*mx,0));sy=np.sqrt(np.maximum(m['yy']/m['n']-my*my,0));sx[sx==0]=1;sy[sy==0]=1
    mx=torch.tensor(mx,dtype=torch.float32);my=torch.tensor(my,dtype=torch.float32);sx=torch.tensor(sx,dtype=torch.float32);sy=torch.tensor(sy,dtype=torch.float32)
    net=nn.Sequential(nn.Linear(len(mx),256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,len(my))).float();optimizer=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=0,foreach=False);generator=torch.Generator().manual_seed(a.seed);start_step=0;best=float('inf');stale=0;logs=[];t=time.monotonic()
    identity={'task':a.task,'target':a.target,'history':a.history,'seed':a.seed,'input_manifest_sha256':sha(ds.path/'manifest.json'),'protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json')),'code_sha256':sha(__file__)}
    def save_torch(p,d):
        q=p.with_name(p.name+'.tmp');torch.save(d,q);os.replace(q,p)
    if (out/'resume.pt').exists():
        cp=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);assert cp['identity']==identity;net.load_state_dict(cp['net']);optimizer.load_state_dict(cp['optimizer']);generator.set_state(cp['rng']);start_step=cp['step'];best=cp['best'];stale=cp['stale'];logs=cp['logs']
    max_steps=a.technical_steps if a.technical else 2000;check_every=min(100,max_steps)
    for step in range(start_step,max_steps if stale<5 else start_step):
        lr=0.0005*(1+math.cos(math.pi*step/2000))
        for g in optimizer.param_groups:g['lr']=lr
        idx=tr[torch.randint(len(tr),(512,),generator=generator).numpy()];x,y=ds.xy(idx,residual);xt=(torch.from_numpy(x.astype(np.float32))-mx)/sx;yt=(torch.from_numpy(y.astype(np.float32))-my)/sy
        net.train();optimizer.zero_grad(set_to_none=True);loss=(net(xt)-yt).square().mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
        if (step+1)%check_every==0 or step+1==max_steps:
            net.eval();ss=0.;nnn=0
            with torch.no_grad():
                for start in range(0,len(va),4096):
                    xx,yy=ds.xy(va[start:start+4096],residual);xx=(torch.from_numpy(xx.astype(np.float32))-mx)/sx;yy=(torch.from_numpy(yy.astype(np.float32))-my)/sy;v=(net(xx)-yy).square();ss+=float(v.double().sum());nnn+=v.numel()
            score=ss/nnn;improved=score<best
            if improved:
                best=score;stale=0;save_torch(out/'best.pt',{'identity':identity,'net':net.state_dict(),'mean_x':mx,'scale_x':sx,'mean_y':my,'scale_y':sy,'step':step+1,'validation_loss':score})
            else:stale+=1
            logs.append({'step':step+1,'lr':lr,'last_batch_loss':float(loss.detach()),'holdout_mse_standardized':score,'improved':improved,'seconds':time.monotonic()-t});atomic(out/'training_log.json',logs)
            save_torch(out/'resume.pt',{'identity':identity,'net':net.state_dict(),'optimizer':optimizer.state_dict(),'rng':generator.get_state(),'step':step+1,'best':best,'stale':stale,'logs':logs})
            print(a.task,a.target,a.history,a.seed,step+1,round(score,6),round(time.monotonic()-t,1),flush=True)
            if stale>=5:break
    cp=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['net']);net.eval()
    receipt={'train_n':len(tr),'holdout_n':len(va),'optimizer_updates':logs[-1]['step'],'best_step':cp['step'],'best_holdout_loss':best,'seconds':time.monotonic()-t,'world_model_updates':0,'probe_use':'MEASUREMENT_ONLY','seed':a.seed}
    atomic(out/'fit.json',{'status':'FITTED_BEFORE_EVAL_ACCESS',**receipt,**identity})
    if a.technical:return {'status':'TECHNICAL_COMPLETE',**receipt}
    def predict(x,z):
        with torch.no_grad():p=(net((torch.from_numpy(x.astype(np.float32))-mx)/sx)*sy+my).numpy()
        return p+z if residual else p
    ev=save_eval(a,ds,predict,out,out/'best.pt');return {'status':'COMPLETE',**receipt,**ev}

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    ds=Dataset(a.matrix,a.target,a.history);atomic(out/'STARTED.json',{'utc':now(),'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)),'threads':a.threads,'task':a.task,'model':a.model,'target':a.target,'history':a.history,'seed':a.seed,'train_anchor_pool':len(ds.anchors),'kind':'TECHNICAL' if a.technical else 'FORMAL','source_code_sha256':sha(__file__),'protocol_sha256':sha(Path(__file__).with_name('FIT_PROTOCOL.json'))})
    result=ridge(a,ds,out) if a.model=='RIDGE' else mlp(a,ds,out);atomic(out/'COMPLETE.json',{**result,'utc':now(),'task':a.task,'model':a.model,'target':a.target,'history':a.history,'seed':a.seed,'kind':'TECHNICAL' if a.technical else 'FORMAL'});return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--model',choices=['RIDGE','MLP'],required=True);p.add_argument('--target',required=True);p.add_argument('--history',choices=['single','three'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--threads',type=int,default=1);p.add_argument('--technical',action='store_true');p.add_argument('--technical-steps',type=int,default=100);a=p.parse_args();main(a)
