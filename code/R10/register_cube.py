"""Seal Cube formal protocol only after the four original TECH pairs pass."""
import datetime, hashlib, json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
gate_path=root/'tech/C_attempt1/TECH_GATE.json'
gate=json.loads(gate_path.read_text())
assert gate['status']=='PASS' and len(gate['checks'])==4 and all(r['paired_gate'] for r in gate['checks'])
manifest=root/'inputs/M2_CASE_WINDOWS_CUBE.json'
selection=root/'inputs/M2_CUBE_CASE_SELECTION.json'
entries=json.loads(manifest.read_text())['cases']
assert len(entries)==100
assert {r['case']['case_id'] for r in entries}=={r['case_id'] for r in json.loads(selection.read_text())['cases']}
pr=dict(
 version='R10_MODULE_C_FORMAL_PREREG_V1',phase='FORMAL',module='C',task='cube',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 execution_code_sha256={n:sha(root/'code'/n) for n in ['old_history_runner_c.py','history_context.py','cube_reset.py','launch_cube.py']},
 analysis_code_sha256=sha(root/'code/report_old.py'),
 case_selection_sha256=sha(selection),manifest_sha256=sha(manifest),
 baseline_sha256=sha(root/'inputs/CUBE_BASELINE.json'),plans_sha256=sha(root/'inputs/CUBE_FIRST_PLANS.npz'),
 model_registry_sha256=sha(root/'inputs/X1_MODEL_REGISTRY.json'),
 manifest_file=manifest.name,baseline_file='CUBE_BASELINE.json',plans_file='CUBE_FIRST_PLANS.npz',asset_subdir='M2/cube',
 case_order=[r['case']['case_id'] for r in entries],
 arms=['H0','REFIT_103201','REFIT_103202','REFIT_103203'],
 streams=['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'],
 optimizer_updates=0,checkpoint_rule='Only fixed final 30000-step refits and official H0',
 primary_gate='BITWISE_EQUAL first plan against same-case/model/stream fresh single-frame control; exact5 future blocks; macro action dimension25 from checkpoint',
 fresh_paired_controls=True,tech_gate='PASS',tech_gate_sha256=sha(gate_path),
 tech_receipt='8 complete trajectories;4/4 paired first plans BITWISE_EQUAL on four original Cube TECH case/model pairs',
 planned_intervention_trajectories=1200,planned_new_control_trajectories=1200,planned_new_total=2400,
 primary_endpoints={'E_C':'success(H_REAL3_REPLAN) minus success(fresh same-host H_POLICY)'},
 population='100 fixed R8 M2 Cube cases;4 models x3 streams averaged within case',
 prediction='未检出',random_action_reference=0.48,
 history='Real online frames[t-10,t-5,t] and two preceding executed action blocks; no synthetic frames; horizon5 future blocks',
 planner=dict(horizon=5,receding=5,action_block=5,num_samples=300,num_steps=30,topk=30,goal_offset_raw=25,budget_raw=50,terminate_at_goal=True),
 reset_label='CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT',
 early_goal_rule='Reached within first call is retained with entered_replanning=False',
 secondary=['per-model effects','entered-replanning proportions','descriptive fresh-minus-sealed single-frame success drift'],
 bootstrap=dict(unit='case',replicates=5000,seed_rule='SHA256("R10_CASE_BOOTSTRAP/C/<endpoint>") first8 bytes unsigned big-endian',ci='linear percentile2.5,97.5',labels=['提高','降低','未检出']),
 missing_policy='Primary uses cases complete across all4models and3streams; no imputation or automatic retries; unequal first plan invalidates affected history row',
 stop_rule=dict(threshold=0.9,scope='all workers combined',check_interval_seconds=5,
  numerator='completed valid history trajectories',denominator='completed valid history plus resolved failed/invalid history or failed paired controls',
  condition='Stop spawned workers immediately at first poll with nonzero denominator and ratio below0.90; never retry formal attempts',
  final_delivery='Report completed history/1200, all unfinished attempts remain technical missing'),
 resources=dict(gpu_count=8,workers=16,workers_per_gpu=2),
 protocol_amendment_before_formal='Uniform fresh same-host controls for all1200 pairs, motivated by cross-host bitwise drift in B/D; strict equality unchanged. Sealed C1 TECH plan comparison and sealed R8 control success drift are descriptive. Original case/model/stream selection retained.',
 runner_derivation_sha256=sha(root/'ops/C_RUNNER_DERIVATION.json'),
 source_versions=dict(lewm='8edfeb336732b5f3ce7b8b210d0ba370a09e2cac',stable_worldmodel='abdced49809d5eae38e24b27dc7b635c502c4812',ogbench='1.2.1')
)
out=root/'prereg/MODULE_C_FORMAL_PREREG.json'
with out.open('x') as f:f.write(json.dumps(pr,ensure_ascii=False,indent=2)+'\n')
(out.with_suffix('.sha256')).write_text(sha(out)+'  '+out.name+'\n')
print(out,sha(out))
