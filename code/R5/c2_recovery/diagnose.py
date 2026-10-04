"""Read-only comparison of the already failed C2 reset; no simulation."""
from pathlib import Path
import numpy as np
from c2.common import *
def main():
    c=read(ROOT/'C2_CONTRACT.json');cid='R3_cube_ed7d40ae89e31ab29e099b34'
    a=Path('/workspace/r5/C1/closed_loop/EVAL/R3_ORIGINAL/H0')/cid
    b=ROOT/'closed_loop/EVAL/R4_ALT_CEM_1/H0'/cid;out=ROOT/'diagnostic/INITIAL_RENDER_MISMATCH';out.mkdir(parents=True,exist_ok=True)
    reference=verify(c['initial_equivalence_sources'][cid]);sources={};states={}
    for label,folder in [('C1_GPU6',a),('C2_GPU0',b)]:
        sources[label]={n:record(folder/n) for n in ('STARTED.json','result.json','states.npz','trajectory.npz')}
        with np.load(folder/'states.npz') as f:initial={k[4:]:f[k].copy() for k in f.files if k.startswith('000:')}
        states[label]=initial;save_npz(out/(label+'_INITIAL.npz'),**initial)
    comparison=comparisons(states['C1_GPU6'],states['C2_GPU0']);csv_write(out/'ALL_STATE_AND_RENDER_COMPARISON_RAW.csv',comparison)
    x=states['C1_GPU6']['render'];y=states['C2_GPU0']['render'];indices=np.argwhere(x!=y)
    rows=[{'case_id':cid,'row':int(i),'column':int(j),'channel':int(k),'C1_GPU6_uint8':int(x[i,j,k]),'C2_GPU0_uint8':int(y[i,j,k]),'C2_minus_C1':int(y[i,j,k])-int(x[i,j,k])} for i,j,k in indices]
    csv_write(out/'RENDER_CHANNEL_DIFFERENCE_RAW.csv',rows)
    pixelcheck={}
    with np.load(a/'trajectory.npz') as f,np.load(b/'trajectory.npz') as g:
        for label,key in [('initial_policy_pixel','raw_pixels'),('goal_pixel','goal_pixels')]:
            xx=f[key][0] if key=='raw_pixels' else f[key];yy=g[key][0] if key=='raw_pixels' else g[key]
            import hashlib
            pixelcheck[label]={'bitwise_equal':xx.dtype==yy.dtype and xx.shape==yy.shape and xx.tobytes()==yy.tobytes(),'C1_sha256':hashlib.sha256(xx.tobytes()).hexdigest(),'C2_sha256':hashlib.sha256(yy.tobytes()).hexdigest()}
    gate_records={k:c[k] for k in ('c0_reset','c0_contract','c0_gate','c1_contract','c1_tech_gate','main_delivery_gate')}
    for r in gate_records.values():verify(r)
    result=read(b/'result.json');index=next(i for i,x in enumerate(c['cases']['EVAL']) if x['case_id']==cid)
    summary={'status':'TECHNICAL_HOLD_RENDER_BITWISE_MISMATCH','case_id':cid,'metadata_zero_based_index':index,'C1_gpu_physical':6,'C2_gpu_physical':0,'C1_shard':0,'C2_shard':0,'executed_raw_steps':result['executed_raw_steps'],'CEM_calls':result['replan_calls'],'nonrender_fields':len(comparison)-1,'nonrender_fields_bitwise_equal':all(r['bitwise_equal'] for r in comparison if r['field']!='render'),'render_different_channels':len(indices),'render_different_pixels':len(set(tuple(v[:2]) for v in indices)),'render_max_abs':max(abs(r['C2_minus_C1']) for r in rows),'official_injected_policy_input_pixel_comparison':pixelcheck,'tolerance_changed':False,'new_simulation':False,'source_or_model_mutation_detected':False,'source_locks_verified':gate_records,'sources':sources,'diagnostic_code':record(__file__),'cause':'Only render differs in preserved evidence. GPU changed6to0; device causality not yet established. No new reset trial, same-device replay or tolerance change performed.'}
    atomic(out/'DIAGNOSTIC_RECEIPT.json',summary)
    (out/'CONCLUSION_ZH.txt').write_text('C2首次规划前逐位门槛触发；该attempt为0执行raw步、0次CEM。\n75项非渲染完整状态全部逐位相同；仅13个RGB通道分量相差1灰阶。\n官方注入的初始policy图像与goal图像完全相同，C0复位源码及来源锁校验通过。\nC1使用GPU6、C2使用GPU0；现有证据不能单独证明设备是原因。\n保留技术停线；未重跑、未放宽容差、未更换case或规划随机流。\n')
    print({k:summary[k] for k in ('status','case_id','nonrender_fields_bitwise_equal','render_different_channels','render_different_pixels','render_max_abs','official_injected_policy_input_pixel_comparison')},flush=True)
if __name__=='__main__':main()
