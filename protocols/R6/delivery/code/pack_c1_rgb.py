import numpy as np,json,hashlib,datetime,sys
from pathlib import Path
root=Path('/workspace/r6/c1_rgb_illustration'); r5=Path('/workspace/r5/C1/closed_loop/EVAL/R3_ORIGINAL')
cases=['R3_cube_04ef52ba98506070e25e8bf1','R3_cube_066d4d5266942b23869774ef'];arms=['H0','REFIT_103201','REFIT_103202','REFIT_103203']; rows=[]
for ci,c in enumerate(cases):
 for a in arms:
  src=r5/a/c/'trajectory.npz'; comp=src.parent/'COMPLETE.json'; assert src.is_file() and comp.is_file()
  with np.load(src,allow_pickle=False) as f:
   assert f['raw_pixels'].dtype==np.uint8
   out=root/f'case_{ci+1:02d}_{c}_{a}.npz'; np.savez_compressed(out,raw_pixels=f['raw_pixels'],goal_pixels=f['goal_pixels'])
  rows.append({'case_order':ci+1,'case_id':c,'arm':a,'source_trajectory_path':str(src),'source_trajectory_bytes':src.stat().st_size,'source_trajectory_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_complete_path':str(comp),'illustration_file':out.name,'illustration_bytes':out.stat().st_size,'illustration_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'fields':['raw_pixels','goal_pixels'],'source_R5_C1_raw_table_sha256':'083d4d33f4f23ef3340174a22fef47bdbf17a55ff65990adf0d03bae5f9e6de6'})
(root/'C1_RGB_ILLUSTRATION_MANIFEST.json').write_text(json.dumps({'version':'R6_C1_RGB_ILLUSTRATION_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'R5 C1 sealed EVAL/R3_ORIGINAL; first two case-list entries; four arms each','case_order_source_raw_table_sha256':'083d4d33f4f23ef3340174a22fef47bdbf17a55ff65990adf0d03bae5f9e6de6','cases':cases,'arms':arms,'rows':rows,'R4_R5_evidence_modified':False},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PACKED','files':len(rows),'bytes':sum(x['illustration_bytes'] for x in rows)}))
