"""Bounded force-memory reconstruction on saved sealed trajectories only."""
import json,sys
from pathlib import Path
import numpy as np
from contact_gate import ContactGate

def main(run):
 read=lambda p:json.loads(Path(p).read_text());cfg=read(Path(__file__).with_name('contact_gate_protocol.json'))['gate']
 protocol=read(Path(__file__).with_name('alpha_repair_protocol.json'));result={}
 ledger=read(run/'compute-ledger.json');phys=next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']=='physics' and a.get('result_status')=='completed')
 for cond in protocol['conditions']:
  if cond['split']!='sealed_new_gap':continue
  for ctl in ['contact','A','B']:
   key=cond['name']+'__'+ctl;ep=read(phys/(key+'.json'));cs=ep['control'];errors=[]
   for i,c in enumerate(cs):
    gate=ContactGate(cfg)
    for prev in cs[max(0,i-319):i+1]:label=gate.step(prev['measured_force_xyz'])['alpha']
    errors.append(abs(label-c.get('shadow_alpha',c['alpha'])))
   result[key]=dict(force_fifo320_reconstruct_shadow_max=max(errors),frames=len(cs))
 (run/'analysis/saved-memory-diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
