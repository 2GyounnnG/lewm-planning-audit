"""Read-only hardware, package and deterministic CUDA smoke evidence."""
import hashlib, importlib.metadata, json, os, pathlib, platform, shutil, subprocess
import torch

root = pathlib.Path('/workspace/r4_v23_execution/ops')
rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip().splitlines()
assert len(rows)==8 and all('RTX 5090' in r for r in rows)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
checks=[]
for i in range(8):
    a=torch.arange(16,dtype=torch.float32,device=f'cuda:{i}').reshape(4,4)
    b=(a@a).cpu()
    checks.append({'gpu':i,'finite':bool(torch.isfinite(b).all()),'sum':float(b.sum())})
data={'hostname':platform.node(),'gpu_rows':rows,'cuda_checks':checks,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'python':platform.python_version(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_max':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'disk_free_bytes':shutil.disk_usage('/workspace').free,'packages':dict(sorted((x.metadata['Name'],x.version) for x in importlib.metadata.distributions()))}
(root/'HOST_AUDIT.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='packages'}),flush=True)
