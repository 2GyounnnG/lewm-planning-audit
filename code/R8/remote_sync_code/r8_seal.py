from __future__ import annotations
import hashlib,json,pathlib,datetime
ROOT=pathlib.Path('/workspace/r8'); OPS=ROOT/'ops'; REP=ROOT/'reports'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 out=[]
 for p in sorted(pathlib.Path(root).rglob('*')):
  if p.is_file():out.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 return out
def count(root): return sum(1 for p in pathlib.Path(root).rglob('result.json'))
def main():
 seals=[]
 specs=[('M1',['raw/U1','raw/U2'],4800),('M2_X1',['raw/M2/cube/FORMAL','raw/M2/tworoom/FORMAL'],2400),('M2_REACHER_PUSHT',['raw/M2_reacher','raw/M2_pusht'],2400),('M3',['raw/M3/FORMAL/REPEAT3','raw/M3/FORMAL/REAL2','raw/M3/FORMAL/PRED_PAST'],3600)]
 for name,dirs,expected in specs:
  files=[];n=0
  for d in dirs:
   p=ROOT/d;n+=count(p);files.extend(inventory(p))
  if n!=expected:raise RuntimeError(f'{name} count {n}!={expected}')
  out=REP/(name+'_SEAL.json');doc={'module':name,'status':'SEALED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'result_count':n,'expected_count':expected,'files':files,'no_new_training':True,'automatic_destroy':False};out.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');seals.append(str(out))
 rnd=[]
 for p in [ROOT/'raw/M2_RANDOM_REACHER_RAW_VALUES.csv',ROOT/'raw/M2_RANDOM_PUSHT_RAW_VALUES.csv',ROOT/'raw/M2/random/cube_RANDOM_RAW_VALUES.csv',ROOT/'raw/M2/random/tworoom_RANDOM_RAW_VALUES.csv']:
  if not p.exists():raise RuntimeError('missing random '+str(p))
  rnd.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 (REP/'M2_RANDOM_SEAL.json').write_text(json.dumps({'module':'M2_RANDOM','status':'SEALED','files':rnd,'rows_each':300,'seeds':[7001,7002,7003]},ensure_ascii=False,indent=2)+'\n');seals.append(str(REP/'M2_RANDOM_SEAL.json'))
 manifest={'version':'R8_EVIDENCE_MANIFEST_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modules':seals,'reports':inventory(REP),'ops_manifests':inventory(OPS),'r4_r5_r6_r7_untouched':True,'new_training':0,'automatic_destroy':False,'remote_originals_retained':True}
 (REP/'R8_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'SEALED','seals':seals,'report_files':len(manifest['reports']),'ops_files':len(manifest['ops_manifests'])}))
if __name__=='__main__':main()
