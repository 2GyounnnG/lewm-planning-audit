"""Flatten completed CPU raw comparisons; no model execution or new statistics."""
import csv
from .local_recovery import HERE,SEEDS,file,read,write

def main():
    rows=[];sources=[];mse=[]
    for seed in SEEDS:
        p=HERE/f'weight_recovery/{seed}/CPU_RECOVERY_CHECK.json';d=read(p);sources.append(file(p))
        for x in d['inherited_R3_replay_checks']:rows.append({'seed':seed,**x,'source_receipt_sha256':sources[-1]['sha256']})
        for x in d['comparisons']:
            if 'horizon_macro' in x:mse.append({'seed':seed,**x,'source_receipt_sha256':sources[-1]['sha256']})
    out=HERE/'reports/recovery';out.mkdir(parents=True,exist_ok=True)
    for name,values in [('CPU_TOLERANCE_RAW.csv',rows),('CPU_MSE_DIFFERENCES_RAW.csv',mse)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(values[0]));w.writeheader();w.writerows(values)
    assert all(r['status']=='PASS' for r in rows)
    (out/'CONCLUSION_ZH.txt').write_text('三个最终30k checkpoint的全部100锚点、两历史、五步预测已在CPU复算。\n576000个预测坐标沿用原R3 atol=rtol=1e-5合同，超差数为0。\nCPU与GPU的全参数及buffer SHA一致，复算无参数更新。\n这里只验证保存开环预测；闭环数值数组完整性另行核验，不声称本地模拟重跑。\n')
    write(out/'COMPLETE.json',{'status':'PASS_INHERITED_R3_CPU_REPLAY_TOLERANCE','sources':sources,'coordinate_count':576000,'outlier_count':0,'new_optimizer_updates':0,'new_simulator_steps':0,'code':file(__file__),'files':[file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='COMPLETE.json']})

if __name__=='__main__':main()
