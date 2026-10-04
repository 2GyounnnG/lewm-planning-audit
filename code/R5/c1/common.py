from c0.common import canonical,digest,sha,record,read,atomic,freeze,verify,save_npz,csv_write,comparisons
import os
from pathlib import Path
ROOT=Path(os.environ.get('R5_C1_ROOT','/workspace/r5/C1'))
C0=Path('/workspace/r5/C0')
ARMS=('H0','REFIT_103201','REFIT_103202','REFIT_103203')
LABEL='CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT'
EVIDENCE='POST_R4_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES'

def install_reset(world,mj,clear):
    original=world.reset
    def reset(seed=None,options=None):
        if seed is not None:raise ValueError('Official Cube source has no seed; do not invent one')
        result=original(seed=None,options=options)
        clear(world.envs.envs[0].unwrapped,mj)
        return result
    world.reset=reset
