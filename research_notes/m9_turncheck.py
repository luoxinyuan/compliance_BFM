"""Predicted left-turn extension of measured native composition response."""
import json
import torch
from m9_stage import RUN,completed,episode
from m9_cyclic import sample
from locomotion_rollout import spherical
from contact_gate import smooth
from m9_contact import run as contact
def run(env,model,contract,out):
 refs=json.loads((RUN/'native-cycles.json').read_text())['cycles']
 for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
 r=refs['straight'];turn=torch.tensor(json.loads((completed('infer')/'reward-candidates.json').read_text())['rotate-z-0.35-0.65']['z'],device=env.standing.device)
 def native(phase):return model.project_z(sample(r,phase)+.06*(turn-env.standing))
 def cmd(t,e):
  if t<4.5:return spherical(e.standing,native(0),smooth((t-3)/1.5))
  if t<10.5:return native((t-4.5)/r['period'])
  return spherical(native(6/r['period']),e.standing,smooth((t-10.5)/1.5))
 qualification=episode(env,model,contract,out,'composition-positive-predicted',cmd,16.,dict(entry=[3,4.5],execute=[4.5,10.5],exit=[10.5,12],recover=[12,16]))
 if not qualification['complete']:return dict(status='completed',student_optimizer_updates=0,qualification=qualification,contact_skipped='Unsafe proposed endpoint; previous successful finite/contact cases remain available')
 result=contact(env,model,contract,out,cycles=refs,composition=True);result['qualification']=qualification;return result
