import subprocess, pathlib, json, datetime, os
root=pathlib.Path('/workspace/r5'); log=root/'H1a';log.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='12',MKL_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12')
commands=[]
for task in ('tworoom','cube'):
    commands.append(['/workspace/env/bin/python','/workspace/r5/code/h1a/run.py','--task',task,'--bundle',f'/workspace/x1_{task}/recovery/bundle','--r3-root','/workspace/shared_data/r3','--out',f'/workspace/r5/H1a/{task}','--device','cuda','--threads','12'])
script=log/'sequential.py'
script.write_text('import subprocess,json,pathlib\ncommands='+repr(commands)+'\nfor cmd in commands:\n    r=subprocess.run(cmd)\n    if r.returncode: raise SystemExit(r.returncode)\n')
with (log/'runner.log').open('ab',buffering=0) as f:
    p=subprocess.Popen(['taskset','-c','100-111','/workspace/env/bin/python',str(script)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
receipt={'pid':p.pid,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':['tworoom','cube'],'GPU':7,'CPU':'100-111','root':str(log),'R4_unchanged':True}
(log/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
