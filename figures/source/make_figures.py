"""Generate manuscript figures from sealed local tables; no model calls or experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]  # paper_w1/
OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
R4=ROOT/'evidence/r4'; R5=ROOT/'evidence/r5'; R6=ROOT/'evidence/r6'
ft=pd.read_csv(R5/'FOUR_TASK_MAIN_TABLE_V2.csv')
hist=pd.read_csv(R5/'HISTORY_CONDITION_4TASK_TABLE.csv')
rh=pd.read_csv(R5/'REAL_HISTORY_REPLANNING_TABLE.csv')
h3=pd.read_csv(R5/'H3X_CONTEXT_MATCHED_REFIT_EXPLORATORY_TABLE.csv')
probe=pd.read_csv(R5/'SINGLE_FRAME_INFORMATION_TABLE.csv')
attr=pd.read_csv(R4/'R4_ATTRIBUTION_TABLE.csv')
r6=pd.read_csv(R6/'R6_MAIN_TABLE.csv')
r6raw=pd.read_csv(R6/'R6_RAW_VALUES.csv')
r6stats=json.loads((R6/'R6_STATS.json').read_text())

plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'figure.dpi':180,'savefig.bbox':'tight','font.family':'DejaVu Sans'})
BLUE='#0072B2'; ORANGE='#E69F00'; GREEN='#009E73'; RED='#D55E00'; PURPLE='#CC79A7'; GREY='#666666'; BLACK='#222222'

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.png',dpi=220)
    plt.close(fig)

# Fig 1: conceptual audit schematic.
fig,ax=plt.subplots(figsize=(7.0,2.35)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
boxes=[(0.02,0.36,0.16,0.30,'Observation\n(single frame)','\nquery'),(0.23,0.36,0.16,0.30,'LeWM\nlatent prediction','\npredictor'),(0.44,0.36,0.16,0.30,'CEM\ncandidate ranking','\nplanner'),(0.65,0.36,0.16,0.30,'Environment\n25 raw steps','\ncontrol'),(0.86,0.36,0.12,0.30,'Success\nor error','\noutcome')]
for x,y,w,h,t,sub in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#F4F6F8',edgecolor=BLUE,lw=1.2))
    ax.text(x+w/2,y+h*0.62,t,ha='center',va='center',color=BLACK,weight='bold')
    ax.text(x+w/2,y+h*0.25,sub,ha='center',va='center',color=GREY)
for (x1,y1,w1,h1,*_), (x2,y2,w2,h2,*__) in zip(boxes[:-1],boxes[1:]):
    ax.add_patch(FancyArrowPatch((x1+w1,y1+h1/2),(x2,y2+h2/2),arrowstyle='-|>',mutation_scale=11,color=GREY,lw=1))
ax.text(0.08,0.88,'query regime',ha='center',color=ORANGE,weight='bold')
ax.text(0.51,0.88,'ranking',ha='center',color=GREEN,weight='bold')
ax.text(0.87,0.88,'control',ha='center',color=RED,weight='bold')
ax.text(0.50,0.10,'Audit interventions: three-frame query, fixed-menu rescoring, real-history replanning, and planner-stream replication',ha='center',color=BLACK)
save(fig,'fig1_audit_framework')

# Fig 2: offline relative change and closed-loop differences.
tasks=['PushT','Reacher','TwoRoom','Cube']; keys=['pusht','reacher','tworoom','cube']; hs=[1,2,5]
fig,axs=plt.subplots(1,2,figsize=(7.0,2.8),gridspec_kw={'width_ratios':[1.2,1]})
ax=axs[0]
for i,(task,key) in enumerate(zip(tasks,keys)):
    h0=ft[(ft.task==key)&(ft.evaluation_kind=='OPEN_LOOP')&(ft.arm=='H0')].set_index('horizon_macro').latent_raw_MSE
    rr=ft[(ft.task==key)&(ft.evaluation_kind=='OPEN_LOOP')&(ft.arm=='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE')].set_index('horizon_macro').latent_raw_MSE
    vals=[100*(rr[h]/h0[h]-1) for h in hs]
    ax.plot(hs,vals,marker='o',lw=1.5,label=task)
ax.axhline(0,color='k',lw=.6); ax.set_xticks(hs); ax.set_xlabel('Prediction horizon (macro steps)'); ax.set_ylabel('Relative latent MSE change (%)'); ax.set_title('(a) Offline error')
ax.legend(frameon=False,ncol=2,loc='lower left'); ax.grid(axis='y',alpha=.2)
ax=axs[1]
vals=[]; los=[]; his=[]
for key in keys:
    r=ft[(ft.task==key)&(ft.evaluation_kind=='CLOSED_LOOP_CEM')&(ft.arm=='FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE')].iloc[0]
    vals.append(r.delta_success_pp); los.append(r.delta_success_pp-r.delta_case_ci95_low); his.append(r.delta_case_ci95_high-r.delta_success_pp)
y=np.arange(len(tasks)); ax.errorbar(vals,y,xerr=[los,his],fmt='o',color=BLUE,capsize=3); ax.axvline(0,color='k',lw=.6); ax.set_yticks(y,tasks); ax.set_xlabel('Refit − official success (pp)'); ax.set_title('(b) Closed-loop difference'); ax.grid(axis='x',alpha=.2)
save(fig,'fig2_offline_and_success')

# Fig 3: single versus three-frame Reacher error and velocity R2.
fig,axs=plt.subplots(1,2,figsize=(7.0,2.8))
ax=axs[0]
for arm,color,label in [('H0',BLUE,'official'),('FIXED3_REFIT_MEAN_NOT_ENSEMBLE',ORANGE,'three-seed refit')]:
    vals=[]
    for h in hs:
        q=hist[(hist.task=='reacher')&(hist.arm==arm)&(hist.history_kind=='H_POLICY')&(hist.horizon_macro==h)].estimate.iloc[0]
        vals.append(q)
    vals3=[]
    for h in hs:
        q=hist[(hist.task=='reacher')&(hist.arm==arm)&(hist.history_kind=='H_REAL3')&(hist.horizon_macro==h)].estimate.iloc[0]
        vals3.append(q)
    ax.plot(hs,vals,marker='o',lw=1.5,color=color,label=f'{label}, 1 frame')
    ax.plot(hs,vals3,marker='s',lw=1.5,ls='--',color=color,label=f'{label}, 3 frames')
ax.set_yscale('log'); ax.set_xticks(hs); ax.set_xlabel('Prediction horizon (macro steps)'); ax.set_ylabel('Latent MSE (log scale)'); ax.set_title('(a) Query-conditioned error'); ax.legend(frameon=False,fontsize=6); ax.grid(alpha=.2)
ax=axs[1]
comps=['joint_velocity_0','joint_velocity_1']; x=np.arange(2); w=.34
for j,histo in enumerate(['single','three']):
    vals=[probe[(probe.task=='reacher')&(probe.model=='MLP_3SEED_MEAN_NOT_ENSEMBLE')&(probe.history==histo)&(probe.component==c)&(probe.metric=='R2')].estimate.iloc[0] for c in comps]
    ax.bar(x+(j-.5)*w,vals,w,label='1 frame' if histo=='single' else '3 frames',color=[PURPLE,GREEN][j])
ax.axhline(0,color='k',lw=.6); ax.set_xticks(x,['velocity 0','velocity 1']); ax.set_ylabel('R²'); ax.set_title('(b) Velocity probe'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
save(fig,'fig3_query_condition')

# Fig 4: R5 and R6 replanning effects + stream bars.
fig,axs=plt.subplots(1,2,figsize=(7.0,2.8),gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]
r5rows=[]
for label,task,arm in [('Reacher official','reacher','H0'),('Reacher refit','reacher','FIXED3_REFIT_MEAN_NOT_ENSEMBLE'),('PushT refit','pusht','FIXED3_REFIT_MEAN_NOT_ENSEMBLE')]:
    q=rh[(rh.task==task)&(rh.arm==arm)&(rh.stream=='MEAN_OF_THREE_STREAMS_WITHIN_CASE')&(rh.metric.str.contains('success_difference_REAL3_minus_POLICY'))].iloc[0]
    r5rows.append((label,100*q.estimate,100*q.conditional95_low,100*q.conditional95_high))
# Recompute the preregistered R6 endpoint from the sealed raw CSV.  The
# bootstrap seed is the one recorded in R6_STATS.json; this is a read-only
# re-expression of the delivered calculation.
p = r6raw
pol = p[p.history=='H_POLICY'].groupby('case_id').success.mean()
real = p[p.history=='H_REAL3_REPLAN'].groupby('case_id').success.mean()
diff = (real - pol).dropna().to_numpy()
pe = r6stats['primary_endpoint']; rng = np.random.default_rng(int(pe['bootstrap_seed_uint64']))
idx = rng.integers(0, len(diff), size=(int(pe['bootstrap']), len(diff)))
boot = diff[idx].mean(axis=1)
r6_est, r6_lo, r6_hi = 100*diff.mean(), 100*np.quantile(boot, 0.025), 100*np.quantile(boot, 0.975)
r5rows.append(('R6 fresh Reacher',r6_est,r6_lo,r6_hi))
y=np.arange(len(r5rows)); vv=np.array([r[1] for r in r5rows]); lo=vv-np.array([r[2] for r in r5rows]); hi=np.array([r[3] for r in r5rows])-vv
ax.errorbar(vv,y,xerr=[lo,hi],fmt='o',capsize=3,color=RED); ax.axvline(0,color='k',lw=.6); ax.set_yticks(y,[r[0] for r in r5rows]); ax.set_xlabel('Real-history − single-frame (pp)'); ax.set_title('(a) Replanning effect'); ax.grid(axis='x',alpha=.2)
ax=axs[1]
# H0 R6 stream values
r6h=r6[(r6.arm=='H0')]
streams=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2']; x=np.arange(3); w=.35
for j,histname in enumerate(['H_POLICY','H_REAL3_REPLAN']):
    vals=[100*r6h[(r6h.stream==s)&(r6h.history==histname)].success_rate.iloc[0] for s in streams]
    ax.bar(x+(j-.5)*w,vals,w,label='single-frame' if histname=='H_POLICY' else 'real-history',color=[BLUE,ORANGE][j])
ax.set_xticks(x,['A','B','C']); ax.set_ylim(65,102); ax.set_ylabel('Success (%)'); ax.set_title('(b) Fresh-case stream sensitivity'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2)
save(fig,'fig4_replanning')

# Fig 5: fixed menu ranking.
fig,ax1=plt.subplots(figsize=(6.6,2.8)); labels=['Official','Three-seed refit']; x=np.arange(2)
h0=attr[(attr.scope=='INITIAL_FIXED64')&(attr.task=='pusht')&(attr.metric=='D_total')&(attr.arm=='H0')].iloc[0]
rr=attr[(attr.scope=='INITIAL_FIXED64')&(attr.task=='pusht')&(attr.metric=='D_total')&(attr.arm=='FIXED3_REFIT_MEAN_NOT_ENSEMBLE')].iloc[0]
s0=attr[(attr.scope=='INITIAL_FIXED64')&(attr.task=='pusht')&(attr.metric=='spearman_model_sim_lat')&(attr.arm=='H0')].iloc[0]
sr=attr[(attr.scope=='INITIAL_FIXED64')&(attr.task=='pusht')&(attr.metric=='spearman_model_sim_lat')&(attr.arm=='FIXED3_REFIT_MEAN_NOT_ENSEMBLE')].iloc[0]
ax1.bar(x-.18,[h0.estimate,rr.estimate],.36,color=[GREY,BLUE],label='Selection loss $D_{total}$'); ax1.set_ylabel('Selection loss (lower is better)'); ax1.set_xticks(x,labels); ax1.grid(axis='y',alpha=.2)
ax2=ax1.twinx(); ax2.bar(x+.18,[s0.estimate,sr.estimate],.36,color=[PURPLE,GREEN],label='Spearman'); ax2.set_ylabel('Spearman (higher is better)',color=GREEN); ax2.set_ylim(0.5,.85)
ax1.set_title('PushT fixed candidate menu (100 cases × 64 candidates)')
lines,lab=[],[]
for a in [ax1,ax2]:
    l,la=a.get_legend_handles_labels(); lines+=l; lab+=la
ax1.legend(lines,lab,frameon=False,loc='upper left',fontsize=7)
save(fig,'fig5_pusht_ranking')

# Source registration for figure audit.
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
source_lines=['# Figure source registration','', 'All figures are deterministic plots from sealed tables; no additional model calls were made.','']
for p in [R5/'FOUR_TASK_MAIN_TABLE_V2.csv',R5/'HISTORY_CONDITION_4TASK_TABLE.csv',R5/'REAL_HISTORY_REPLANNING_TABLE.csv',R5/'H3X_CONTEXT_MATCHED_REFIT_EXPLORATORY_TABLE.csv',R5/'SINGLE_FRAME_INFORMATION_TABLE.csv',R4/'R4_ATTRIBUTION_TABLE.csv',R6/'R6_MAIN_TABLE.csv',R6/'R6_RAW_VALUES.csv',R6/'R6_STATS.json']:
    source_lines.append(f'- `{p}` — SHA256 `{sha(p)}`')
(OUT/'FIGURE_SOURCES.md').write_text('\n'.join(source_lines)+'\n')
