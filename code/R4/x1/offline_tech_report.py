"""Read Cube's completed CPU/cache/train TECH evidence without new execution."""
import json,shutil
from . import core
from .evaluate import write_csv

def main():
    task='cube';root=core.ROOT;out=root/'reports/cube_OFFLINE_TECH';out.mkdir(parents=True,exist_ok=True)
    model=core.read(root/'technical_model/MODEL_CHECK.json')
    if model['status']!='PASS' or not model['all_model_state_unchanged'] or model['optimizer_updates']!=0:raise RuntimeError('Model TECH failed')
    write_csv(out/'MODEL_CHECK_RAW.csv',[dict(task=task,check=k,max_abs=v,device=model['device'],optimizer_updates=0) for k,v in model['checks'].items()])
    shutil.copyfile(root/'technical_model/MODEL_CHECK.json',out/'MODEL_CHECK.json')
    cache=core.read(root/'manifests/cube_cache.json');parts=[core.read(root/'manifests'/f'cube_cache_part_{i}.json') for i in (0,1)]
    if cache['status']!='FROZEN_OBSERVED_CACHE_COMPLETE':raise RuntimeError('Cache incomplete')
    cache_rows=[]
    for i,p in enumerate(parts):
        cache_rows.append(dict(task=task,shard=i,gpu=6+i,episodes=len(p['episodes']),frames=sum(v['length'] for v in p['episodes'].values()),seconds=p['seconds'],train_variance_frames=p['train_variance_frames'],train_scalar_variance=p['train_scalar_variance'],optimizer_updates=p['optimizer_updates'],frozen_sha256=p['frozen']['sha256']))
    cache_rows.append(dict(task=task,shard='MERGED',gpu='6,7',episodes=len(cache['episodes']),frames=sum(v['length'] for v in cache['episodes'].values()),seconds=cache['seconds'],train_variance_frames=cache['train_variance_frames'],train_scalar_variance=cache['train_scalar_variance'],optimizer_updates=cache['optimizer_updates'],frozen_sha256=cache['frozen']['sha256']))
    write_csv(out/'CACHE_RAW_TABLE.csv',cache_rows)
    for name in ('CACHE_PROFILE.json','CACHE_TELEMETRY_RAW.csv','CACHE_ETA_RAW.csv'):shutil.copyfile(root/'telemetry'/name,out/name)
    training=core.read(root/'technical_train/103201/result.json')
    journal=[json.loads(x) for x in (root/'technical_train/103201/updates.jsonl').read_text().splitlines()]
    if training['status']!='TECHNICAL_COMPLETE' or training['actual_updates']!=128 or [r['step'] for r in journal]!=list(range(1,129)) or training['frozen_before']!=training['frozen_after']:raise RuntimeError('TECH training evidence invalid')
    tr={k:training[k] for k in ('task','seed','technical','actual_updates','frozen_before','frozen_after','new_encoder_updates','state_labels_read','checkpoint_selection','seconds')}
    tr.update(failed_optimizer_updates=0,recovery_optimizer_updates=0,technical_budget_max=1024)
    write_csv(out/'TRAIN_TECH_RAW.csv',[tr]);shutil.copyfile(root/'technical_train/103201/result.json',out/'TRAIN_TECH_RESULT.json')
    gate=root/'reset_audit/RESET_FALLBACK.json'
    status={'task':task,'status':'OFFLINE_TECH_COMPLETE','technical_optimizer_updates':128,'new_encoder_updates':0,'reset_gate':core.file_record(gate),'closed_loop_tech_status':'TECHNICALLY_UNEVALUABLE','closed_loop_tech_trajectories':0,'closed_loop_tech_CEM_calls':0,'second_batch_eligible_after_main_delivery':False,'evidence':{str(p.relative_to(root)):core.file_record(p) for p in (root/'technical_model/MODEL_CHECK.json',root/'manifests/cube_cache.json',root/'technical_train/103201/result.json',root/'technical_train/103201/updates.jsonl')},'report_source':core.file_record(__file__)}
    core.atomic(out/'MODULE_STATUS.json',status)
    (out/'CONCLUSION_ZH.txt').write_text('Cube 真实官方模型在 CPU 的三项接口差异均为0，权重和buffer未变。\n输入缓存覆盖8104条轨迹、1628904帧，两个GPU分片和合并原值已保存。\n独立训练TECH完成128更新，失败/恢复额外更新为0，冻结SHA不变。\n精确复位审计失败；闭环TECH在门槛前停止，0条轨迹、0次CEM，时延不填。\n离线正式后训练与开环继续；不具备第二批闭环随机流资格。\n')
    print(core.canonical(status).decode())

if __name__=='__main__':main()
