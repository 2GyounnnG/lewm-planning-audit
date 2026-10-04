"""Small exact state subset for the already frozen S3 aggregator, not originals."""
import argparse,json,shutil
from pathlib import Path
from .common import atomic_json,atomic_npz,file_record,sha256

def main():
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args();source=Path(a.source);out=Path(a.out);rows=[]
    for path in sorted(source.glob('FORMAL/*/*/*/COMPLETE.json')):
        folder=path.parent;complete=json.loads(path.read_text());target=out/folder.relative_to(source)
        for name,record in complete['files'].items():
            if file_record(folder/name)!=record:raise RuntimeError('Original completed file changed')
        with np.load(folder/'trajectory.npz') as f:
            arrays={k:f[k].copy() for k in ('raw_physical_state','goal_state') if k in f}
        # Deliberate compatible filename is explicitly declared a partial derivative.
        atomic_npz(target/'trajectory.npz',**arrays);shutil.copyfile(folder/'result.json',target/'result.json')
        row={'kind':'DERIVED_STATISTICS_INPUT_NOT_ORIGINAL_TRAJECTORY','source_path':str(folder),'relative_path':str(folder.relative_to(source)),'source_trajectory':complete['files']['trajectory.npz'],'source_result':complete['files']['result.json'],'derived_state_npz':file_record(target/'trajectory.npz'),'retained_arrays':sorted(arrays),'all_retained_values_bitwise_identical':True,'copy_of_result_json_is_original':True}
        atomic_json(target/'DERIVED_STATISTICS_INPUT.json',row);rows.append(row)
        for name in ('DERIVED_METRICS.json','DERIVED_MARGIN.npz'):
            if (folder/name).exists():shutil.copyfile(folder/name,target/name)
    atomic_json(out/'STATISTICS_RECOVERY_MAP.json',{'version':'R4_V23_STATISTICS_STATE_SUBSET_V1','completed_cases':len(rows),'scope':'Exact state/goal subset plus original result JSON for frozen S3 aggregation. These trajectory.npz files are explicitly DERIVED and are not substitutes for original logs or complete recovery archives.','code_sha256':sha256(__file__),'records':rows})
    print(json.dumps({'completed_cases':len(rows),'subset_npz_bytes':sum(r['derived_state_npz']['bytes'] for r in rows)}),flush=True)

if __name__=='__main__':main()
