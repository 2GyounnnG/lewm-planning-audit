import json,hashlib,pathlib,datetime
root=pathlib.Path('/workspace/r8/m5')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
out={'version':'R8_M5_CONFIG_V1','evidence_label':'PROTOCOL_EXTENSION','module':'M5_FULL_HISTORY','case_manifest':{'path':'M5_CASE_WINDOWS.json','sha256':sha(root/'M5_CASE_WINDOWS.json'),'total_R6_cases':100,'eligible_start_ge_10':89,'excluded_start_lt_10':11},'source':{'path':'/workspace/shared_data/r3/data/unpacked/reacher/reacher.h5','sha256':'85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06'},'arms':['H0','REFIT_103201','REFIT_103202','REFIT_103203'],'streams':['R3_ORIGINAL','R4_ALT_CEM_1','R4_ALT_CEM_2'],'history':'first plan [o_{t-10},o_{t-5},reset_render_t] + source actions t-10:t; subsequent plan uses executing real frame','training_updates':0,'rows':1068,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
json.dump(out,open(root/'M5_CONFIG.json','w'),ensure_ascii=False,indent=2)
