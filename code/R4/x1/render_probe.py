"""Cube EGL resource-lifetime TECH check with no actions, CEM or training."""
import gc,hashlib,importlib.metadata as md,os,sys,time
import numpy as np
from . import core
from .evaluate import write_csv

def main():
    core.policy();swm,_,_,_=core.r3('planning').load_official_api();folder=core.ROOT/'technical_render';rows=[];errors=[]
    original=sys.unraisablehook
    sys.unraisablehook=lambda event:errors.append({'type':event.exc_type.__name__,'message':str(event.exc_value)})
    try:
        for trial in range(4):
            began=time.monotonic();world=swm.World(**core.CONFIG['cube']['world'],num_envs=1,max_episode_steps=100,image_shape=(224,224));world.reset(seed=0)
            env=world.envs.envs[0].unwrapped;image=np.asarray(env.render()).copy()
            if image.shape!=(224,224,3) or image.dtype!=np.uint8:raise RuntimeError('Cube render contract differs')
            renderer=env._renderer;renderer.close();renderer.close();world.close()
            del renderer,env,world;gc.collect()
            rows.append({'task':'cube','trial':trial,'backend':os.environ['MUJOCO_GL'],'seed':0,'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                'seconds':time.monotonic()-began,'width':224,'height':224,'raw_actions':0,'CEM_calls':0,'optimizer_updates':0,'cleanup_errors_so_far':len(errors)})
    finally:sys.unraisablehook=original
    passed=not errors and len({r['image_sha256'] for r in rows})==1
    write_csv(folder/'RENDER_CLEANUP_RAW.csv',rows)
    receipt={'status':'PASS' if passed else 'FAIL','trials':4,'same_seed_image_equality':len({r['image_sha256'] for r in rows})==1,
        'unraisable_cleanup_errors':errors,'backend':os.environ['MUJOCO_GL'],'ogbench_version':md.version('ogbench'),
        'mujoco_version':md.version('mujoco'),'numpy_version':np.__version__,'code_sha256':core.sha(__file__),
        'raw_actions':0,'CEM_calls':0,'optimizer_updates':0,'prior_default_close_observation':'gym.Env.close no-op; EGL destructor warnings occurred at process exit; explicit renderer.close tested here'}
    core.atomic(folder/'RENDER_CLEANUP.json',receipt)
    receipt['external_raw_actions']=0;receipt['internal_reset_stabilization_steps_per_trial_source']=2
    core.atomic(folder/'RENDER_CLEANUP.json',receipt)
    (folder/'CONCLUSION_ZH.txt').write_text('\n'.join([
        'Cube EGL 连续 4 次初始化/渲染完成，图像为 224×224 RGB uint8。',
        f'显式 renderer.close 的清理异常数为 {len(errors)}；同 seed 图像一致项为 {receipt["same_seed_image_equality"]}。',
        '固定源码未转发 reset seed；此次直接初始化不能确认复位确定性。',
        '外部动作、CEM 和优化器更新均为 0；源码 reset 内另有每次 2 步随机稳定动作。',
        '正式闭环仍等待真实 source 状态覆盖后的既定 TECH 复位门槛。'])+'\n')
    print(core.canonical(receipt).decode())
    if not passed:raise RuntimeError('Cube renderer-lifetime TECH failed')

if __name__=='__main__':main()
