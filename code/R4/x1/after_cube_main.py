"""Queue Cube recovery export after the offline coordinator's successful report."""
import subprocess,sys,time
from . import core

if __name__=='__main__':
    if core.ROOT!=core.Path('/workspace/x1_cube'):raise RuntimeError('Cube only')
    gate=core.ROOT/'state/offline_report_exit.json'
    while not gate.exists():time.sleep(10)
    if core.read(gate)['returncode']!=0:raise RuntimeError('Cube final reporting failed')
    status=core.read(core.ROOT/'reports/cube/MODULE_STATUS.json')
    if status['status']!='COMPLETE_WITH_TECHNICAL_LIMITATIONS':raise RuntimeError('Completed authorized offline scope required')
    subprocess.run([sys.executable,'-u','-m','x1.recovery_pack','cube'],check=True)
