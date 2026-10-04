"""Read-only timing/progress snapshots. No model, GPU, environment or network calls.

Projections are descriptive extrapolations from completed technical measurements,
not deadline promises or scientific outcomes. Mutable journals are snapshotted once.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('pusht', 'reacher')
ARMS = ('H0', 'REFIT_103201', 'REFIT_103202', 'REFIT_103203')
REPORT = 'reports/COMPUTE_STORAGE_AND_ETA.md'


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def quantile(values, q):
    values = sorted(float(x) for x in values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values): raise ValueError('Finite nonnegative timings required')
    k = (len(values)-1)*q; i = int(k)
    return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(k-i)


def slots_for(workers):
    if type(workers) is not int or workers not in (2, 4): raise ValueError('Only completed 2/4 worker profile allowed')
    return [(gpu, slot) for slot in range(workers//2) for gpu in (0, 1)]


def fifo_wall(jobs, workers, seconds_per_update, active=()):
    """Preserve active assignments, then original FIFO on earliest available slot."""
    slots = slots_for(workers); loads = {x: 0. for x in slots}; done = set(); rows = []
    by_id = {j['job_id']: j for j in jobs}
    for item in active:
        key = (item['gpu'], item['slot']); jid = item['job_id']
        if key not in loads or jid not in by_id or key in {r['slot_key'] for r in rows} or jid in done: raise ValueError('Invalid active FIFO assignment')
        end = by_id[jid]['remaining_updates']*seconds_per_update
        loads[key] = end; done.add(jid); rows.append({'job_id': jid, 'slot_key': key, 'start_seconds': 0., 'finish_seconds': end, 'active': True})
    queue = [(v, slots.index(k), k) for k, v in loads.items()]; heapq.heapify(queue)
    for job in jobs:
        if job['job_id'] in done or not job['remaining_updates']: continue
        start, order, key = heapq.heappop(queue); end = start+job['remaining_updates']*seconds_per_update
        rows.append({'job_id': job['job_id'], 'slot_key': key, 'start_seconds': start, 'finish_seconds': end, 'active': False})
        heapq.heappush(queue, (end, order, key))
    return {'wall_seconds': max((x[0] for x in queue), default=0.), 'assignment': rows}


def cem_wall(cases, seconds_by_task, completed=()):
    completed = set(completed); loads = [0., 0.]; counts = {task: [0, 0] for task in TASKS}
    for task in TASKS:
        for case in cases[task]:
            gpu = int(hashlib.sha256(case['case_id'].encode()).hexdigest(), 16) % 2
            remaining = sum((task, case['case_id'], arm) not in completed for arm in ARMS)
            loads[gpu] += remaining*seconds_by_task[task]; counts[task][gpu] += remaining
    return {'wall_seconds': max(loads), 'per_gpu_seconds': loads, 'remaining_trajectories_by_task_gpu': counts}


class Snapshot:
    def __init__(self, root): self.root = Path(root).resolve(); self.inputs = {}; self.cache = {}; self.warnings = []
    def path(self, rel):
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('ROOT-relative receipt path required')
        q = self.root/p
        if not q.resolve().is_relative_to(self.root): raise ValueError('Receipt escapes root')
        return q
    def raw(self, rel, optional=False):
        rel = str(rel)
        if rel in self.cache: return self.cache[rel]
        p = self.path(rel)
        if not p.exists() and optional: return None
        before = p.stat(); content = p.read_bytes(); after = p.stat()
        self.inputs[rel] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
            'mtime_ns': before.st_mtime_ns, 'changed_during_read': (before.st_size, before.st_mtime_ns)!=(after.st_size, after.st_mtime_ns)}
        if self.inputs[rel]['changed_during_read']: self.warnings.append('Live file changed during single read: '+rel)
        self.cache[rel] = content; return content
    def read(self, rel, optional=False):
        content = self.raw(rel, optional)
        return None if content is None else json.loads(content)
    def verified(self, rel, expected):
        value = self.read(rel); actual = self.inputs[str(rel)]
        expected = {'sha256': expected} if isinstance(expected, str) else expected
        if actual['sha256'] != expected['sha256'] or ('bytes' in expected and actual['bytes'] != expected['bytes']): raise RuntimeError('Completed receipt differs: '+str(rel))
        return value
    def journal(self, rel, updates=False):
        raw = self.raw(rel, True)
        if raw is None: return []
        lines = raw.splitlines(); rows = []
        for i, line in enumerate(lines):
            try: rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines)-1 and not raw.endswith(b'\n'):
                    self.warnings.append('Ignored incomplete final journal line: '+rel); break
                raise
        if updates and [x['step'] for x in rows] != list(range(1,len(rows)+1)): raise RuntimeError('Noncontiguous formal update journal: '+rel)
        return rows


def training(s):
    manifest = s.read('manifests/JOBS.json'); jobs = manifest['jobs']
    if len(jobs)!=6 or any(x['updates']!=30000 for x in jobs): raise RuntimeError('Fixed six-job budget differs')
    progress = []
    for job in jobs:
        rows = s.journal('artifacts/train/'+job['job_id']+'/updates.jsonl', updates=True)
        if len(rows)>30000 or any(x.get('technical') is not False for x in rows): raise RuntimeError('Formal journal budget/type differs')
        progress.append({'job_id':job['job_id'], 'task':job['task'], 'successful_updates':len(rows), 'remaining_updates':30000-len(rows),
            'journal_compute_seconds':sum(x.get('compute_seconds',0.) for x in rows),
            'journal_optimizer_intent_seconds':sum(x.get('optimizer_intent_commit_seconds',0.) for x in rows),
            'mean_observed_compute_seconds_per_update':sum(x.get('compute_seconds',0.) for x in rows)/len(rows) if rows else None})
    result = {'status':'PENDING_COMPLETED_PROFILE', 'jobs':progress, 'remaining_updates':sum(x['remaining_updates'] for x in progress),
              'formal_full_cache_startup_seconds':None, 'formal_startup_note':'TECH uses four-episode caches; its startup cannot estimate full formal TRAIN/MONITOR cache loading.',
              'unmeasured_additional_monitor_checkpoint_and_full_cache_overhead_seconds':None}
    profile = s.read('state/GPU_PROFILE.json', True); gate = s.read('state/TRAINING_TECHNICAL_GATE.json', True)
    if not profile or not gate: return result
    if profile.get('status')!='PROFILE_COMPLETE' or gate.get('status')!='TRAINING_TECHNICAL_GATES_PASS': return result
    workers = profile['selected_workers']; slots_for(workers)
    if gate['selected_workers']!=workers or gate['microbatch']!=128: raise RuntimeError('Profile and training gate disagree')
    if gate.get('profile_receipt'): s.verified('state/GPU_PROFILE.json', gate['profile_receipt'])
    phase = profile['A' if workers==2 else 'B']
    if phase.get('status')!='PASS' or len(phase['specs'])!=workers: raise RuntimeError('Selected profile incomplete')
    steady = phase['steady_update_span']; low = steady.get('aggregate_updates_per_second_lower'); high = steady.get('aggregate_updates_per_second_upper')
    state = s.read('state/formal_training/STATUS.json', True) or {}; active = state.get('active', [])
    active = [x for x in active if x.get('returncode') is None]
    result.update(selected_workers=workers, workers_per_gpu=workers//2, profile_stage=phase['stage'],
                  profile_end_to_end_aggregate_updates_per_second=phase['end_to_end_updates_per_second'],
                  profile_launch_wall_seconds=phase['launch_wall_seconds'], profile_completed_updates=phase['total_successful_updates'],
                  profile_steady_aggregate_updates_per_second_range=[low, high], controller_status=state.get('status', 'NOT_STARTED'),
                  scheduler_assumption='Preserve current active slots; dispatch remaining original-order jobs FIFO. Per-worker rate=selected aggregate/workers; short-profile task mix may differ from formal FIFO waves.',
                  setup_excluded_from_remaining_projection=True)
    if steady.get('status')!='BOUNDED_OBSERVED_INTERVAL' or not positive(low) or not positive(high) or low>high:
        result['status']='PENDING_RESOLVED_STEADY_THROUGHPUT'; return result
    fast=fifo_wall(progress,workers,workers/high,active); slow=fifo_wall(progress,workers,workers/low,active)
    midpoint=2/(low+high)*workers
    for row in progress:
        row['profile_assumed_worker_updates_per_second_range']=[low/workers, high/workers]
        row['remaining_worker_loop_seconds_range']=[row['remaining_updates']*workers/high,row['remaining_updates']*workers/low]
    result.update(status='DESCRIPTIVE_PROFILE_PROJECTION', remaining_critical_path_seconds_range=[fast['wall_seconds'],slow['wall_seconds']],
                  projected_schedule=fifo_wall(progress,workers,midpoint,active),
                  range_meaning='Polling interval uncertainty of one short completed profile, not confidence/prediction bounds; excludes unmeasured formal cache startup and future transfer/analysis.')
    if state.get('status')=='DRAINING_AFTER_FAILURE' or s.path('state/formal_training/STOP_REQUESTED.json').exists():
        result['status']='BLOCKED_CONTROLLER_REVIEW'; result['remaining_critical_path_seconds_range']=None
    return result


def planning(s):
    result={'status':'PENDING_EIGHT_COMPLETE_TECH_TRAJECTORIES', 'planner_workers':2,'workers_per_gpu':1,'tasks':{}}
    gate=s.read('state/PLANNING_TECH_GATE.json',True)
    gate_pass=bool(gate and gate.get('status')=='PASS' and gate.get('complete_trajectories')==8 and gate.get('clone_exact') is True)
    if gate_pass: sources=gate['files']
    else:
        ledger=s.read('state/TECHNICAL_LEDGER.json',True) or {}
        sources={v['result_path']:v['result_sha256'] for v in ledger.get('trajectories',{}).values() if v.get('status')=='COMPLETE'}
    samples={t:[] for t in TASKS}; seen=set()
    for rel, value in sources.items():
        if rel.startswith('artifacts/planning/TECH/') and rel.endswith('/result.json'):
            row=s.verified(rel,value)
            if row.get('status')!='COMPLETE' or row.get('phase')!='TECH' or row.get('arm') not in ('H0','H0_CLONE'): raise RuntimeError('Unexpected completed technical trajectory')
            key=(row['task'],row['case_id'],row['arm'])
            if key in seen or not positive(row['trajectory_wall_seconds']): raise RuntimeError('Duplicate/invalid timing')
            seen.add(key); samples[row['task']].append({'case_id':row['case_id'],'arm':row['arm'],'seconds':row['trajectory_wall_seconds'],'source':rel})
    for task in TASKS:
        values=[x['seconds'] for x in samples[task]]
        if values:
            result['tasks'][task]={'raw_samples':samples[task],'timing_trajectories':len(values),'unique_cases':len({x['case_id'] for x in samples[task]}),
                'median_seconds':quantile(values,.5),'p75_seconds':quantile(values,.75),'p90_seconds':quantile(values,.9),'min_seconds':min(values),'max_seconds':max(values),
                'measurement_status':'COMPLETE_ORIGINAL_TRAJECTORIES_ONLY_FULL_GATE_PENDING' if not gate_pass else 'TECH_GATE_COMPLETE'}
    if not gate_pass: return result
    if len(seen)!=8 or any(len(samples[t])!=4 or len({x['case_id'] for x in samples[t]})!=2 for t in TASKS): raise RuntimeError('Expected two cases times H0/clone per task')
    cases={t:s.read(f'manifests/{t}_data_roles.json')['cases']['EVAL'] for t in TASKS}
    ledger=s.read('state/FORMAL_TRAJECTORY_LEDGER.json',True) or {}; complete=set(); unresolved=[]
    for run_id, row in ledger.get('trajectories',{}).items():
        if row.get('status')=='COMPLETE':
            result_row=s.verified(row['result_path'], row['result_sha256'])
            if result_row.get('status')!='COMPLETE' or result_row.get('phase')!='FORMAL': raise RuntimeError('Formal timing ledger differs')
            complete.add((result_row['task'],result_row['case_id'],result_row['arm']))
        else: unresolved.append({'run_id':run_id,'status':row.get('status')})
    allowed={(t,c['case_id'],a) for t in TASKS for c in cases[t] for a in ARMS}
    if not complete.issubset(allowed): raise RuntimeError('Unexpected formal completion identity')
    for task in TASKS:
        values=[x['seconds'] for x in samples[task]]
        result['tasks'][task]={'raw_samples':samples[task],'timing_trajectories':4,'unique_cases':2,
            'median_seconds':quantile(values,.5),'p75_seconds':quantile(values,.75),'p90_seconds':quantile(values,.9),
            'min_seconds':min(values),'max_seconds':max(values),'formal_cases':len(cases[task]),
            'formal_completed_trajectories':sum(k[0]==task for k in complete)}
    median=cem_wall(cases,{t:result['tasks'][t]['median_seconds'] for t in TASKS},complete)
    slow=cem_wall(cases,{t:result['tasks'][t]['p90_seconds'] for t in TASKS},complete)
    result.update(status='SPARSE_TECH_TIMING_EXTRAPOLATION', median_projection=median,slow_p90_projection=slow,
                  remaining_critical_path_seconds_median_to_p90=[median['wall_seconds'],slow['wall_seconds']],
                  completed_formal_trajectories=len(complete),unresolved_ledger_entries=unresolved,
                  future_planner_worker_setup_seconds=None,
                  limitations='Only two unique TECH cases/task and their exact H0 clones (four timings), no refit-policy timing sample. Empirical median/p90 are sparse descriptive scenarios, not independent-sample confidence bounds or promised completion range. RESERVED trajectories conservatively count as wholly remaining; task/card hash allocation retained.')
    if any(x['status'] not in ('RESERVED',) for x in unresolved): result['status']='BLOCKED_INFRASTRUCTURE_REVIEW';result['remaining_critical_path_seconds_median_to_p90']=None
    return result


def actual_costs(s):
    rows=[]; candidates=set()
    for pattern in ('state/**/RESET_VALIDATION_RECEIPT.json','state/**/ENVIRONMENT_PREFLIGHT.json','state/**/REAL_DATA_MODEL_CHECK.json','state/**/COST_WRAPPER_CHECK.json','artifacts/**/RESET_VALIDATION_RECEIPT.json','artifacts/technical/*/*/result.json','artifacts/train/*/result.json','state/planning/*/worker_gpu*_pid*.json','state/technical_training/*/smoke_continuous8.json','state/technical_training/*/smoke_split4.json','state/technical_training/*/smoke_resume4.json'):
        candidates.update(str(p.relative_to(s.root)) for p in s.root.glob(pattern))
    candidates.update(p for p in ('state/ENVIRONMENT_READY.json','state/GPU_SMOKE.json','state/GPU_PROFILE.json','state/PLANNING_TECH_GATE.json','state/PLANNING_FORMAL_COMPLETE.json','state/PUSHT_ENCODER_TIMING.json') if s.path(p).exists())
    for task in TASKS:
        candidates.update(f'manifests/{task}_{k}.json' for k in ('model_assets','data_assets','data_unpacked','cache') if s.path(f'manifests/{task}_{k}.json').exists())
    timing_names={'seconds','wall_seconds','controller_wall_seconds','total_counted_trajectory_wall_seconds','all_worker_attempt_wall_seconds','all_worker_attempt_setup_seconds','timing','counts','optimizer_updates','environment_steps','complete_CEM_trajectories','setup_seconds','worker_wall_seconds','launch_wall_seconds','actual_updates','total_successful_updates'}
    for path in sorted(candidates):
        value=s.read(path); fields={k:v for k,v in value.items() if k in timing_names}
        if path=='state/GPU_PROFILE.json': fields['profile_phases']={k:{n:value[k].get(n) for n in ('status','launch_wall_seconds','total_successful_updates','sum_worker_compute_seconds','sum_worker_loader_seconds')} for k in ('A','B') if value.get(k)}
        if path.endswith('ENVIRONMENT_PREFLIGHT.json'): fields['tasks']=value.get('tasks')
        if path.endswith('PUSHT_ENCODER_TIMING.json'): fields['timing_evidence']=value
        rows.append({'source':path,'status':value.get('status'),'task':value.get('task'),'actual_recorded_fields':fields,
                     'missing_total_wall_is_unknown':not any(k in fields for k in ('seconds','wall_seconds','controller_wall_seconds')),
                     'scope':'As recorded by this receipt; nested/worker/controller timers overlap; resumed invocation may not include prior attempts.'})
    ledger=s.read('state/TECHNICAL_LEDGER.json',True) or {}; runs=ledger.get('runs',{})
    for run_id, row in runs.items(): rows.append({'source':'state/TECHNICAL_LEDGER.json','run_id':run_id,'actual_recorded_fields':row,'scope':'Technical segment, including failure/unknown actual counts; no timing invented.'})
    return {'records':rows,'technical_successful_updates_known':sum(v.get('actual_updates') or 0 for v in runs.values()),
            'technical_charged_updates':sum(v['reserved_updates'] if v.get('actual_updates') is None else v['actual_updates'] for v in runs.values()),
            'technical_unknown_actual_segments':[k for k,v in runs.items() if v.get('actual_updates') is None],
            'technical_trajectory_ledger':ledger.get('trajectories',{}),
            'overlapping_durations_not_summed':True,'unrecorded_attempt_wall_seconds':None}


def render(out):
    train=out['training']; cem=out['planning']; lines=['# 实测计算、存储与剩余时间快照','',f"生成时间：{out['generated_utc']}。此文件只更新运行时间估计，不读取科学效果作选择。",'',
        '训练与 CEM 使用 GPU 的阶段互斥；各阶段分别按两张卡的关键路径计算。工作线程耗时之和不等于墙钟时间。没有已完成实测的阶段保持未知。','',
        f"训练状态：{train['status']}；剩余更新 {train['remaining_updates']:,}/180,000。",'',
        '| Job | 已成功更新 | 剩余更新 | 记录的计算秒/更新 |','|---|---:|---:|---:|']
    for row in train['jobs']: lines.append(f"| {row['job_id']} | {row['successful_updates']} | {row['remaining_updates']} | {row['mean_observed_compute_seconds_per_update']} |")
    if 'selected_workers' in train:
        lines += ['',f"已完成 profile 选择 {train['selected_workers']} workers（每 GPU {train['workers_per_gpu']}）。聚合 steady 吞吐范围：{train['profile_steady_aggregate_updates_per_second_range']} 更新/秒。",
                  f"FIFO 剩余训练关键路径投影：{train.get('remaining_critical_path_seconds_range')} 秒。",'这是一次短 profile 的轮询时间范围，假定每 worker 分得相同聚合吞吐；正式 FIFO 各波次任务比例可能不同，不能视为统计置信区间或工期保证。']
    lines += ['', '正式完整 TRAIN/MONITOR 缓存启动时间未知；TECH 四 episode 小缓存启动不能代替。未来监控、保存和传输额外开销也未保证覆盖。', '', f"CEM 状态：{cem['status']}。"]
    if cem.get('tasks'):
        lines += ['', '| Task | 技术轨迹 / 独立 case | 中位数秒 | p75秒 | p90秒 | 实测最小–最大秒 |', '|---|---:|---:|---:|---:|---:|']
        for task,row in cem['tasks'].items(): lines.append(f"| {task} | {row['timing_trajectories']} / {row['unique_cases']} | {row['median_seconds']:.3f} | {row['p75_seconds']:.3f} | {row['p90_seconds']:.3f} | {row['min_seconds']:.3f}–{row['max_seconds']:.3f} |")
        lines += ['',f"按实际 case SHA 分卡、每卡 1 planner、每 case 四臂串行的剩余关键路径：{cem.get('remaining_critical_path_seconds_median_to_p90')} 秒（中位数及慢 p90 情景）。",
                  '每 task 仅两个 TECH case 的 H0/clone 四次计时；没有 refit 臂的正式运行分布。样本稀疏且 clone 不是独立案例，范围不保证未来耗时。未完成预约按整条轨迹计入。']
    lines += ['',f"训练+CEM 已测部分的串行投影：{out['measured_phase_projection_seconds_range']} 秒；这不是项目交付 ETA。",'Open-loop、最终统计/图件、恢复传输和交付验收的剩余耗时均未知；不作 1–2 天承诺。','',
              '## 已发生开销与存储','',f"技术更新：已知成功 {out['actual_costs']['technical_successful_updates_known']}，保守记账 {out['actual_costs']['technical_charged_updates']}。未知 actual 的技术段单列在 JSON。",
              '所有 reset 失败/成功 attempt、环境建立、下载解包、编码缓存、TECH 和正式日志已保存为各自范围的原始字段；嵌套或并发 timer 不相加。缺失 timer 不当作零。','',
              '| Receipt | 状态 | 已记录开销字段 |','|---|---|---|']
    for row in out['actual_costs']['records']:
        fields=json.dumps(row.get('actual_recorded_fields'),ensure_ascii=False,sort_keys=True)
        lines.append('| '+row['source']+' | '+str(row.get('status','SEGMENT'))+' | '+fields.replace('|',r'\|')+' |')
    lines += ['',f"文件系统快照：{out['storage_snapshot']}。不是本项目独占使用量。",'', '机器可读原值、所有输入 SHA/字节数、逐 job 投影、规划分卡和范围说明见 `state/ETA_PROFILE.json`。']
    if out['warnings']: lines += ['', '读取说明：']+['- '+x for x in out['warnings']]
    return '\n'.join(lines)+'\n'


def atomic(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def run(root=ROOT):
    s=Snapshot(root); seal=s.read('manifests/ANALYSIS_OUTPUTS.json',True)
    if seal and seal.get('status')=='COMPLETE' and REPORT in seal.get('files',{}): raise RuntimeError('Final analysis already seals this report; ETA refresh cannot rewrite it')
    train=training(s); cem=planning(s); costs=actual_costs(s); usage=shutil.disk_usage(s.root)
    a=train.get('remaining_critical_path_seconds_range');b=cem.get('remaining_critical_path_seconds_median_to_p90')
    total=[a[i]+b[i] for i in range(2)] if a is not None and b is not None else None
    out={'version':'R3_MEASURED_ETA_SNAPSHOT_V1','generated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
         'status':'TIMING_SNAPSHOT_NOT_COMPLETION_PROMISE','training':train,'planning':cem,'actual_costs':costs,
         'measured_phase_projection_seconds_range':total,'total_delivery_eta_seconds':None,
         'unknown_remaining_stages':['formal full-cache startup','future planner setup','open-loop','final analysis/plots','recovery transfer','delivery acceptance'],
         'storage_snapshot':{'filesystem_total_bytes':usage.total,'filesystem_used_bytes':usage.used,'filesystem_free_bytes':usage.free},
         'inputs':s.inputs,'warnings':s.warnings,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'new_optimizer_updates':0,'new_trajectories':0,'new_environment_steps':0}
    atomic(s.path('state/ETA_PROFILE.json'),(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    atomic(s.path(REPORT),render(out).encode());return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();out=run(args.root);print(json.dumps({'status':out['status'],'training':out['training']['status'],'planning':out['planning']['status'],'total_delivery_eta_seconds':None}))
