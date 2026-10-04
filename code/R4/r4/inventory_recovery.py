"""Inventory completed local recovery without changing scientific outputs."""
import argparse,collections,csv,json,time
from pathlib import Path
from .common import atomic_json,file_record,sha256

def inventory(root,task):
    base=Path(root)/task;current=json.loads((base/'RECOVERY_MANIFEST.json').read_text())
    if current['status']!='COMPLETE_AND_ALL_FILE_HASHES_VERIFIED':raise RuntimeError('Recovery incomplete')
    menu=base/'menu_evidence';minimum=base/'minimum_trajectories/minimum_recovery';groups=collections.Counter()
    for path in minimum.glob('**/MINIMUM_DERIVATION.json'):
        parts=path.parent.relative_to(minimum).parts
        if len(parts)!=4:raise RuntimeError('Unexpected trajectory layout')
        phase,stream,case,arm=parts;groups[(phase,stream,arm)]+=1
    rows=[{'task':task,'phase':p,'stream':s,'arm':a,'cases':n} for (p,s,a),n in sorted(groups.items())]
    csvpath=base/'MINIMUM_TRAJECTORY_COUNTS_RAW.csv'
    with csvpath.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    s3groups=collections.Counter();logical_s3=0
    for path in (menu/'trajectory_menus').glob('FORMAL/*/*/*/menus/replan_*/MENU_LOCK.json'):
        parts=path.relative_to(menu/'trajectory_menus').parts;lock=json.loads(path.read_text())
        if not (path.parent/'menu.npz').exists() or not (path.parent/'selection.json').exists():raise RuntimeError('Actual executed menu missing')
        s3groups[(parts[1],parts[3])]+=1;logical_s3+=64
    initial=list((menu/'s2').glob('*/initial_proposals.npz'));raw=list((menu/'s2').glob('*/candidate_raw.csv'))
    candidate_rows=sum(sum(1 for _ in csv.DictReader(p.open())) for p in raw)
    if len(initial)!=100 or len(raw)!=100 or candidate_rows!=6400:raise RuntimeError('Initial menu coverage differs')
    expected=1384 if task=='pusht' else 1264
    if sum(groups.values())!=expected:raise RuntimeError('Minimum trajectory count differs')
    cpu=Path(__file__).resolve().parents[1]/'tables/recovery'/(task+'_first_h0')/'RECOVERY_CHECK.json'
    if json.loads(cpu.read_text())['status']!='COMPLETE_RECOVERY_COMPARISON':raise RuntimeError('CPU recovery comparison incomplete')
    result={'version':'R4_V23_LOCAL_RECOVERY_INVENTORY_V1','status':'COMPLETE_AND_HASH_VERIFIED','task':task,
        'archive_manifest':file_record(base/'RECOVERY_MANIFEST.json'),
        'per_file_SHA_manifests':{str(p.relative_to(base)):file_record(p) for p in (menu/'menu_evidence_MANIFEST.json',base/'minimum_trajectories/minimum_trajectories_MANIFEST.json')},
        'original_case_derivatives':sum(groups.values()),'formal_trajectory_cases':sum(v for (p,s,a),v in groups.items() if p=='FORMAL'),'technical_trajectory_cases':sum(v for (p,s,a),v in groups.items() if p=='TECH'),
        'trajectory_count_table':file_record(csvpath),'s2_initial_forks':100,'s2_full_four_model_proposal_sets':100,'s2_logical_candidate_rows':candidate_rows,
        's3_actual_menus':sum(s3groups.values()),'s3_logical_candidates':logical_s3,'s3_menu_counts':[{'stream':s,'arm':a,'menus':v} for (s,a),v in sorted(s3groups.items())],
        'CPU_restore_receipt':{'path':str(cpu),**file_record(cpu)},'original_RGB_H5_full300_CEM_logs_retained_on_host':True,
        'reconstruction':'Exact pinned HDF5 CASE_WINDOWS source map plus per-case original reset/actions/states/latents/goals/returned plans/replan RNG and every-frame SHA; derivative identities never substitute for original source hashes.',
        'pixel_reconstruction_tested_on_local_CPU':False,'secondary_batch_excluded':True,'code_sha256':sha256(__file__)}
    atomic_json(base/'FINAL_RECOVERY_INVENTORY.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    for task in ('pusht','reacher'):
        target=Path(a.root)/task/'RECOVERY_MANIFEST.json'
        while a.wait and not target.exists():time.sleep(15)
        result=inventory(a.root,task);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
