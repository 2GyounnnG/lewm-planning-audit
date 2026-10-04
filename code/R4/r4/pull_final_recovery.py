"""Local sequential resumable recovery, independent of lightweight control master."""
import hashlib,json,subprocess,tarfile,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'remote_results/final_recovery'
SSH='ssh -c aes128-ctr -m hmac-sha2-256 -o IPQoS=none -p 46720 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts'
HOST='root@211.72.13.202'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def receive(task,name):
 target=OUT/task;target.mkdir(parents=True,exist_ok=True)
 for attempt in range(12):
  command=['rsync','--partial','--timeout=180','-v','-e',SSH,HOST+':/workspace/r4_'+task+'/recovery_archives/'+name,str(target)+'/' ]
  result=subprocess.run(command)
  if result.returncode==0:return target/name
  print(json.dumps({'task':task,'file':name,'attempt':attempt+1,'returncode':result.returncode}),flush=True)
  time.sleep(min(15*(attempt+1),60))
 raise RuntimeError('Recovery transfer exhausted retries: '+task+'/'+name)

receipts={task:json.loads(receive(task,'RECOVERY_ARCHIVES.json').read_text()) for task in ('pusht','reacher')}
for kind in ('menu_evidence','minimum_trajectories'):
 for task in ('pusht','reacher'):
  record=receipts[task][kind];path=receive(task,kind+'.tar.gz')
  if path.stat().st_size!=record['archive']['bytes'] or digest(path)!=record['archive']['sha256']:raise RuntimeError('Archive hash differs')
  out=OUT/task/kind;out.mkdir(parents=True,exist_ok=True)
  with tarfile.open(path) as archive:
   for member in archive.getmembers():
    if not (member.isfile() or member.isdir()) or not (out/member.name).resolve().is_relative_to(out.resolve()):raise RuntimeError('Unexpected unsafe archive entry')
   archive.extractall(out)
  manifest=out/(kind+'_MANIFEST.json')
  if digest(manifest)!=record['manifest']['sha256']:raise RuntimeError('Manifest hash differs')
  files=json.loads(manifest.read_text())['files']
  for relative,item in files.items():
   value=out/relative
   if value.stat().st_size!=item['bytes'] or digest(value)!=item['sha256']:raise RuntimeError('Recovered member differs: '+str(value))
  result={'status':'RECOVERED_AND_ALL_FILE_HASHES_VERIFIED','task':task,'kind':kind,'files':len(files),'archive':record['archive'],'completed_at_unix':time.time()}
  (OUT/task/(kind+'_LOCAL_VERIFIED.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
for task in ('pusht','reacher'):
 local={kind:json.loads((OUT/task/(kind+'_LOCAL_VERIFIED.json')).read_text()) for kind in ('menu_evidence','minimum_trajectories')}
 index=json.loads((OUT/task/'minimum_trajectories/minimum_recovery/MINIMUM_RECOVERY_INDEX.json').read_text())
 menus=json.loads((OUT/task/'menu_evidence/menu_evidence_MANIFEST.json').read_text())
 final={'version':'R4_V23_LOCAL_RECOVERY_MANIFEST_V1','status':'COMPLETE_AND_ALL_FILE_HASHES_VERIFIED','task':task,'archives':local,'original_source_trajectory_cases':index['completed_cases'],'minimum_derived_npz_bytes':index['derived_npz_bytes'],'s2_initial_proposals':sum(k.endswith('/initial_proposals.npz') for k in menus['files']),'s2_initial_menus':sum(k.startswith('s2/') and k.endswith('/menu.npz') for k in menus['files']),'exact_menu_evidence_files':len(menus['files']),'original_rgb_and_H5_retained_on_host':True,'source_reconstruction_maps':'menu_evidence/source_window_maps/CASE_WINDOWS.json','per_file_SHA_manifests':['menu_evidence/menu_evidence_MANIFEST.json','minimum_trajectories/minimum_trajectories_MANIFEST.json'],'post_primary_second_batch_excluded':True,'CPU_restore_receipt':str(ROOT.parent/'tables/recovery/pusht_first_h0/RECOVERY_CHECK.json') if task=='pusht' else None}
 (OUT/task/'RECOVERY_MANIFEST.json').write_text(json.dumps(final,indent=2)+'\n')
print('ALL_PRIMARY_R4_MINIMUM_RECOVERY_COMPLETE',flush=True)
