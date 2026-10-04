"""Recover only published R3 immutable manifests to the actual guarded external MyProj."""
from pathlib import Path,PurePosixPath
import argparse,base64,fcntl,hashlib,json,os,shlex,stat,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/workspace/r3_official_lewm_predictor_refit'
HOST='root@175.155.64.241'
SSH=['ssh','-p','19171','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/g1_pusht_2x5090_v1/state/known_hosts']
MOUNT=Path('/Volumes/MyProj');TARGET=MOUNT/'r3_official_lewm_predictor_refit/recovery';DEVICE=16777237

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def safe(relative):
 p=PurePosixPath(relative)
 if not relative or p.is_absolute() or '..' in p.parts or str(p)!=relative or any(ord(c)<32 for c in relative) or any(x.startswith('._') for x in p.parts):raise ValueError('Unsafe R3 published relative path')
 return p

def guard():
 if MOUNT.is_symlink() or not MOUNT.is_mount() or MOUNT.stat().st_dev!=DEVICE:raise RuntimeError('Required real MyProj external volume unavailable; no internal fallback')
 current=MOUNT
 for part in TARGET.relative_to(MOUNT).parts:
  current=current/part
  s=current.lstat()
  if stat.S_ISLNK(s.st_mode) or not stat.S_ISDIR(s.st_mode) or s.st_dev!=DEVICE:raise RuntimeError('External recovery root/device changed')

def target(relative,create=False):
 guard();parts=safe(relative).parts;p=TARGET
 for i,part in enumerate(parts):
  parent=p;p=p/part
  if not p.exists() and not p.is_symlink():
   if i<len(parts)-1 and create:
    guard();p.mkdir(exist_ok=True)
   else:continue
  s=p.lstat()
  if stat.S_ISLNK(s.st_mode) or s.st_dev!=DEVICE:raise RuntimeError('Recovery path symlink/device violation')
  if i<len(parts)-1 and not stat.S_ISDIR(s.st_mode):raise RuntimeError('Recovery ancestor is not directory')
 return p

def write(relative,data):
 p=target(relative,True);q=target(relative+'.tmp',True)
 with q.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 guard();os.replace(q,p)

def remote_python(code):
 p=subprocess.run(SSH+[HOST,'python3 -'],input=code,text=True,capture_output=True)
 if p.returncode:raise RuntimeError('Authorized R3 SSH query failed: '+p.stderr[-1000:])
 return json.loads(p.stdout)

def fetch_manifest(relative):
 safe(relative)
 if not (relative.startswith('state/recovery_queue/') or relative.startswith('manifests/RECOVERY_')) or not relative.endswith('.json'):raise ValueError('Only explicit R3 publication manifests may be recovered')
 code="from pathlib import Path\nimport json,hashlib,base64,stat,os\nr=Path(%r);p=r/%r\nassert os.uname().nodename=='6494ba5e1b1a'\nassert p.resolve().is_relative_to(r) and not p.is_symlink()\ns=p.stat();b=p.read_bytes();t=p.stat()\nassert (s.st_size,s.st_mtime_ns,s.st_ino)==(t.st_size,t.st_mtime_ns,t.st_ino) and stat.S_ISREG(s.st_mode)\nprint(json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}))\n"%(REMOTE,relative)
 d=remote_python(code);raw=base64.b64decode(d['base64'],validate=True)
 if len(raw)!=d['bytes'] or hashlib.sha256(raw).hexdigest()!=d['sha256']:raise RuntimeError('Published manifest transport identity differs')
 manifest=json.loads(raw);files=manifest['files']
 if not files:raise ValueError('Empty publication')
 for name,r in files.items():
  safe(name)
  if name.startswith(('recovery/','.venv/','data/source/','data/unpacked/')) or name==relative:raise ValueError('Publication includes forbidden bulk source/cache or recursive path')
  if type(r.get('bytes')) is not int or r['bytes']<0 or len(r.get('sha256',''))!=64:raise ValueError('Invalid file SHA/size')
 return raw,d['sha256'],manifest

def matches(path,record):return path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256']

def recover(relative):
 guard();raw,identity,manifest=fetch_manifest(relative);prefix='recovery_receipts/'+identity
 # Only remote-published completed units may be installed. Existing canonical
 # bytes are immutable: a mismatch stops, rather than replacing old evidence.
 marker=target(prefix+'/VERIFIED.json',True)
 if marker.exists():
  d=json.loads(marker.read_text())
  for name,record in manifest['files'].items():
   if not matches(target(name),record):raise RuntimeError('Previously verified external payload differs: '+name)
  return d
 write(prefix+'/manifest_exact.json',raw)
 missing=[]
 for name,record in manifest['files'].items():
  p=target(name)
  if p.exists():
   if not matches(p,record):raise RuntimeError('Existing external evidence differs; preserve without overwrite: '+name)
  else:missing.append(name)
 if missing:
  need=sum(manifest['files'][x]['bytes'] for x in missing)
  if __import__('shutil').disk_usage(MOUNT).free<need+(1<<30):raise RuntimeError('Insufficient real external space; no internal fallback')
  listing=prefix+'/files0';write(listing,b''.join(x.encode()+b'\0' for x in missing))
  stage=target(prefix+'/staging/placeholder',True).parent
  for name in missing:target(prefix+'/staging/'+name,True)
  command=['rsync','-rtz','--partial','--checksum','--stats','--from0','--files-from='+str(target(listing)),'-e',shlex.join(SSH),HOST+':'+REMOTE+'/','./']
  started=time.perf_counter();p=subprocess.run(command,cwd=stage,text=True,capture_output=True)
  write(prefix+'/rsync_stdout.txt',p.stdout.encode());write(prefix+'/rsync_stderr.txt',p.stderr.encode())
  if p.returncode:raise RuntimeError('R3 rsync failed; external partial and logs retained: '+str(p.returncode))
  for name in missing:
   src=target(prefix+'/staging/'+name);record=manifest['files'][name]
   if not matches(src,record):raise RuntimeError('Recovered file SHA/size differs: '+name)
  for name in missing:
   src=target(prefix+'/staging/'+name);dst=target(name,True)
   if dst.exists():raise RuntimeError('Concurrent canonical writer detected')
   guard();os.replace(src,dst)
  elapsed=time.perf_counter()-started
 else:elapsed=0
 for name,record in manifest['files'].items():
  if not matches(target(name),record):raise RuntimeError('Final recovered content differs: '+name)
 write(relative,raw)
 receipt={'status':'VERIFIED','manifest_path':relative,'manifest_sha256':identity,'files':manifest['files'],'logical_bytes':sum(r['bytes'] for r in manifest['files'].values()),'new_files':len(missing),'seconds':elapsed,'external_target':str(TARGET),'external_device':DEVICE,'remote_deleted':False,'verified_at_unix':time.time()}
 write(prefix+'/VERIFIED.json',(json.dumps(receipt,indent=2)+'\n').encode())
 print(json.dumps({k:receipt[k] for k in ('status','manifest_path','logical_bytes','new_files','seconds')}),flush=True)
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();guard()
 lock=target('recovery_receipts/recovery.lock',True)
 with lock.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);recover(a.manifest)
