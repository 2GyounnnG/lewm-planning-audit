"""Verify and deliver the completed fixed refit module without loading a GPU."""
import argparse,json
from . import core
from .evaluate import write_csv

def main(task):
    rows=[];evidence={}
    for seed in core.SEEDS:
        folder=core.ROOT/'train'/str(seed);r=core.read(folder/'result.json');identity=core.read(folder/'RUN_IDENTITY.json')
        journal=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
        if r['status']!='REFIT_TRAINING_COMPLETE_UNSCORED' or r['actual_updates']!=30000 or r['technical']:
            raise RuntimeError('Incomplete fixed30k training')
        if [x['step'] for x in journal]!=list(range(1,30001)) or r['frozen_before']!=r['frozen_after'] or r['identity_sha256']!=core.digest(identity):
            raise RuntimeError('Training count or identity mismatch')
        pointer=core.read(folder/'last.json');core.verify(pointer)
        rows.append({k:r[k] for k in ('task','seed','actual_updates','seconds','batch','microbatch','new_encoder_updates','state_labels_read',
            'checkpoint_selection','frozen_before','frozen_after')})
        rows[-1].update({k:r['sampling'][k] for k in ('sampled_windows','unique_windows','unique_episodes','predicted_target_tokens')})
        rows[-1].update(final_checkpoint_sha256=pointer['sha256'],successful_optimizer_journal_rows=len(journal),final_checkpoint_step=30000)
        evidence[str(seed)]={name:core.file_record(folder/name) for name in ('result.json','RUN_IDENTITY.json','updates.jsonl','last.json')}
    out=core.ROOT/'reports'/f'{task}_TRAIN';write_csv(out/'TRAIN_RAW_TABLE.csv',rows)
    status={'status':'COMPLETE','task':task,'formal_seeds':list(core.SEEDS),'formal_successful_optimizer_updates':90000,'technical_updates_separate':128,
        'new_encoder_updates':0,'state_labels_read':0,'evidence':evidence,'raw_table':core.file_record(out/'TRAIN_RAW_TABLE.csv')}
    core.atomic(out/'MODULE_STATUS.json',status)
    (out/'CONCLUSION_ZH.txt').write_text(f'{task} 三个固定 seed 均完成 30,000 次更新，共 90,000 次正式更新。\n每个 seed 的更新日志严格连续，主评价固定使用 checkpoint_30000。\n只更新 predictor 与 pred_proj；冻结参数和全部 buffer 的前后 SHA 一致。\n独立 TECH 的 128 次更新单列，未继承进正式训练。\n开环与闭环结果由后续完整固定 case 评价给出。\n')
    print(core.canonical(status).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();main(a.task)
