from pathlib import Path
from .common import *
def main():
    c=read('/workspace/r5/C1/C1_CONTRACT.json');gate=ROOT/'MAIN_DELIVERY_GATE.json'
    g=read(gate)
    if g.get('main_results_delivered') is not True:raise RuntimeError('Parent main-delivery gate required')
    for entry in g.get('sources',{}).values():verify(entry)
    from x1 import core
    core.ROOT=Path('/workspace/x1_cube')
    if core.task_contract('cube')!=c['official']:raise RuntimeError('Inherited official configuration changed')
    for arm,entry in c['models'].items():
        if arm=='H0':
            for asset in entry['files'].values():verify(asset)
        else:
            verify(entry['delta']);verify(entry['training_result'])
            if read(core.ROOT/'train'/arm.split('_')[1]/'last.json')!=entry['delta']:raise RuntimeError('Fixed30k pointer changed')
    verify(c['c0_reset']);verify(c['c0_contract']);verify(c['c0_gate'])
    status=Path('/workspace/r5/C1/reports/MODULE_STATUS.json')
    if read(status)['status']!='COMPLETE':raise RuntimeError('C1 incomplete')
    c.update(version='R5_C2_V1',module='C2',stream=list(STREAMS),formal_trajectories=800,
      c1_contract=record('/workspace/r5/C1/C1_CONTRACT.json'),c1_module_status=record(status),
      c1_tech_gate=record('/workspace/r5/C1/TECH_GATE.json'),main_delivery_gate=record(gate),
      initial_equivalence_sources={case['case_id']:record(Path('/workspace/r5/C1/closed_loop/EVAL/R3_ORIGINAL/H0')/case['case_id']/'states.npz') for case in c['cases']['EVAL']},
      case_seed='Inherited x1.evaluate.case_seed(cube,case_id,replan_index,stream): first8 SHA256({R4_ALT_CEM_1|2}/cube/{case_id}/{replan_index}) uint64 big endian',
      technical_design={'inherited_C1_TECH':record('/workspace/r5/C1/TECH_GATE.json'),'new_technical_trials':0},no_new_TECH_trials=True,initial_equivalence_gate='Before each first CEM solve compare full physical/controller state and render bitwise against same-case completed C1 H0; no simulation repeated, no tolerance change',resources={'GPU0':list(range(80,96)),'GPU1':list(range(96,112)),'threads':1},code={p.name:record(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file() and p.suffix in ('.py','.sh')})
    freeze(ROOT/'C2_CONTRACT.json',c)
    print({'status':'FROZEN','contract':record(ROOT/'C2_CONTRACT.json')},flush=True)
if __name__=='__main__':main()
