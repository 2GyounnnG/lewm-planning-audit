from c1.common import canonical,digest,sha,record,read,atomic,freeze,verify,save_npz,csv_write,comparisons,install_reset,C0,ARMS,LABEL,EVIDENCE
from pathlib import Path
import os
ROOT=Path(os.environ.get('R5_C2_ROOT','/workspace/r5/C2'))
STREAMS=('R4_ALT_CEM_1','R4_ALT_CEM_2')
