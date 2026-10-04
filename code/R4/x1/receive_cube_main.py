"""Recover only Cube's small completed main interface ahead of large archives."""
import subprocess,time
from . import core

if __name__=='__main__':
    local=core.Path(__file__).parent/'reports/cube_MAIN';local.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S','/private/tmp/r4_v23_control_only','-p','46720','-o','UserKnownHostsFile=/Users/richwang/Documents/ChatGPT/热/r4_v23_execution/ops/known_hosts']
    host='root@211.72.13.202';command='test -f /workspace/x1_cube/state/offline_report_exit.json && test -f /workspace/x1_cube/reports/cube/MODULE_STATUS.json'
    while subprocess.run([*ssh,host,command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:time.sleep(20)
    subprocess.run(['rsync','-rtz','-e',' '.join(ssh),host+':/workspace/x1_cube/reports/cube/',str(local)+'/'],check=True)
    status=core.read(local/'MODULE_STATUS.json')
    if status['status']!='COMPLETE_WITH_TECHNICAL_LIMITATIONS':raise RuntimeError('Unexpected Cube main state')
    for name,rec in status['tables'].items():core.verify(dict(rec,path=str(local/name)))
    core.atomic(local/'MAIN_RECEIVE.json',{'status':'PASS','received_at':time.time(),'module_status':core.file_record(local/'MODULE_STATUS.json'),'tables':{n:core.file_record(local/n) for n in status['tables']}})
    print('CUBE_MAIN_RECOVERED_VERIFIED',flush=True)
