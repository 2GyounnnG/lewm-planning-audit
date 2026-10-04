"""Create an isolated R3 environment; never modify the G1/R2 runtime."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]

def run():
    if os.uname().nodename != '6494ba5e1b1a': raise RuntimeError('Wrong authorized instance')
    if shutil.disk_usage(ROOT).free < 40*(1<<30): raise RuntimeError('Not enough space for isolated dependencies above reserve')
    env = ROOT / '.venv'; python = env / 'bin/python'
    before = subprocess.check_output(['/workspace/g1_pusht_2x5090_v1/.venv/bin/python', '-c',
        "import importlib.metadata as m,json;print(json.dumps(sorted((x.metadata['Name'],x.version) for x in m.distributions())))"], text=True)
    if not python.exists(): subprocess.run(['uv', 'venv', '--python', '/workspace/g1_pusht_2x5090_v1/.venv/bin/python', str(env)], check=True)
    # Existing core versions are deliberately retained. Only dependencies for the
    # two authorized environments, HDF5, and the original CEM API are installed.
    packages = ['torch==2.8.0', 'torchvision==0.23.0', 'transformers==4.51.3',
                'numpy==2.3.5', 'einops==0.8.1', 'pillow==12.3.0', 'pyarrow==25.0.1',
                'pylance==12.0.0', 'huggingface-hub==0.36.2', 'lancedb>=0.30.0',
                'pygame', 'pymunk>=7', 'shapely', 'gymnasium', 'dm-control', 'mujoco<3.12',
                'h5py', 'hdf5plugin', 'zstandard', 'scikit-learn', 'scipy',
                'loguru', 'tabulate', 'rich', 'typer', 'hydra-core', 'omegaconf',
                'opencv-python-headless', 'imageio', 'imageio-ffmpeg', 'psutil', 'pytest']
    start=time.time()
    install_env = dict(os.environ, UV_CONCURRENT_DOWNLOADS='4', UV_HTTP_TIMEOUT='1200',
                       PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['uv', 'pip', 'install', '--python', str(python), *packages], check=True, env=install_env)
    after = subprocess.check_output(['/workspace/g1_pusht_2x5090_v1/.venv/bin/python', '-c',
        "import importlib.metadata as m,json;print(json.dumps(sorted((x.metadata['Name'],x.version) for x in m.distributions())))"], text=True)
    if before != after: raise RuntimeError('Historical G1 environment changed unexpectedly')
    probe = subprocess.check_output([str(python), '-c',
        "import importlib.metadata as m,json,torch,torchvision; print(json.dumps({'python':__import__('sys').version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'torchvision':torchvision.__version__,'packages':dict(sorted((x.metadata['Name'],x.version) for x in m.distributions()))}))"], text=True)
    d = {'status': 'ISOLATED_ENVIRONMENT_READY', 'environment': json.loads(probe), 'requested_packages': packages,
         'old_G1_package_inventory_sha256': hashlib.sha256(before.encode()).hexdigest(),
         'old_environment_unchanged': True, 'seconds': time.time()-start, 'GPU_forward_calls': 0}
    (ROOT/'state/ENVIRONMENT_READY.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d),flush=True)

if __name__ == '__main__': run()
