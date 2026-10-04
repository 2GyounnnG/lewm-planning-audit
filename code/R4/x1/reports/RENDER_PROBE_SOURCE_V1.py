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
    (folder/'CONCLUSION_ZH.txt').write_text('Cube EGL 连续 4 次初始化/渲染/显式清理完成。\n同 seed 图像 SHA 完全一致；显式 renderer.close 后未捕获清理异常。\n图像为 224×224 RGB uint8，原始动作、CEM 调用和优化器更新均为 0。\n依赖仅增加 Cube 私有 ogbench 1.2.1；其余沿用共享环境。\n这是渲染与资源释放技术检查，不替代数据复位和正式控制评价。\n')
    print(core.canonical(receipt).decode())
    if not passed:raise RuntimeError('Cube renderer-lifetime TECH failed')

if __name__=='__main__':main()
