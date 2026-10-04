"""Summarize already recorded reset-audit differences; execute no simulator steps."""
import argparse
import numpy as np
from . import core
from .evaluate import write_csv

def main(task):
    path=core.ROOT/'reset_audit/RESET_FALLBACK.json';receipt=core.read(path);refs={};rows=[];tested=0
    for trial in receipt['trials']:
        with np.load(core.verify(trial['arrays']),allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        reference=refs.setdefault(trial['case_id'],arrays)
        for field in sorted(set(reference)|set(arrays)):
            tested+=1;a=reference.get(field);b=arrays.get(field)
            exact=a is not None and b is not None and a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
            if exact:continue
            matching=a is not None and b is not None and a.shape==b.shape
            difference=np.abs(a.astype(np.float64)-b.astype(np.float64)) if matching else None
            rows.append({'task':task,'case_id':trial['case_id'],'trial':trial['arrays']['path'],'seed':trial['seed'],
                'repeat':trial['repeat'],'path':trial['path'],'field':field,'reference_shape':str(a.shape) if a is not None else None,
                'candidate_shape':str(b.shape) if b is not None else None,'exact_equal':False,
                'mismatched_elements':int(np.count_nonzero(a!=b)) if matching else None,
                'max_absolute_difference':float(difference.max()) if difference is not None and difference.size else None,
                'mean_absolute_difference':float(difference.mean()) if difference is not None and difference.size else None})
    folder=core.ROOT/'reports'/f'{task}_RESET_DIAGNOSTIC';folder.mkdir(parents=True,exist_ok=True)
    if rows:write_csv(folder/'NONZERO_DIFFERENCES.csv',rows)
    result={'task':task,'audit_status':receipt['status'],'trials':len(receipt['trials']),'field_comparisons':tested,
        'nonzero_field_comparisons':len(rows),'differing_fields':sorted({r['field'] for r in rows}),
        'audit_receipt':core.file_record(path),'new_raw_actions':0,'new_optimizer_updates':0,'tolerance_changes':0}
    core.atomic(folder/'DIAGNOSTIC.json',result);print(core.canonical(result).decode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=core.TASKS);a=p.parse_args();main(a.task)
