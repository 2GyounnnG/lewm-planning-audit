import pathlib,hashlib,json,datetime
root=pathlib.Path('/workspace/r8/m5'); exclude={'M5_SEAL.json'}; files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name not in exclude:
  h=hashlib.sha256(); n=0
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''): h.update(b); n+=len(b)
  files.append({'path':str(p.relative_to(root)),'bytes':n,'sha256':h.hexdigest()})
payload={'version':'R8_M5_SEAL_V1','evidence_label':'PROTOCOL_EXTENSION','module':'M5_FULL_HISTORY','status':'SEALED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eligible_cases':89,'excluded_cases':11,'files':files}
text=json.dumps(payload,ensure_ascii=False,indent=2)+'\n'; (root/'M5_SEAL.json').write_text(text)
h=hashlib.sha256((root/'M5_SEAL.json').read_bytes()).hexdigest(); print('seal_sha256',h,'files',len(files),'bytes',sum(x['bytes'] for x in files))
