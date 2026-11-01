"""One fixed small native-turn composition hypothesis, not a gain/seed sweep."""
import json
import torch
from m9_stage import RUN,completed,episode
from m9_cyclic import sample
from contact_gate import smooth
from locomotion_rollout import spherical
from m1_stage import save
def run(env,model,contract,out):
 r=json.loads((RUN/'native-cycles.json').read_text())['cycles']['straight'];r['tensor']=torch.tensor(r['z'],device=env.standing.device);rewards=json.loads((completed('infer')/'reward-candidates.json').read_text());rows=[]
 for sign,name in [('positive','rotate-z-0.35-0.65'),('negative','rotate-z--0.35-0.65')]:
  turn=torch.tensor(rewards[name]['z'],device=env.standing.device);delta=turn-env.standing
  def native(phase):return model.project_z(sample(r,phase)+.03*delta)
  def cmd(t,e):
   if t<4.5:return spherical(e.standing,native(0),smooth((t-3)/1.5))
   if t<10.5:return native((t-4.5)/r['period'])
   return spherical(native(6/r['period']),e.standing,smooth((t-10.5)/1.5))
  rows.append(episode(env,model,contract,out,'composition-'+sign,cmd,16.,dict(entry=[3,4.5],execute=[4.5,10.5],exit=[10.5,12],recover=[12,16])))
 save(out,'composition-hypothesis.json',dict(gain=.03,rationale='Native positive/negative rotate candidates yielded about +/-3rad/s. A fixed3% anchored direction is a local hypothesis near0.1rad/s, not assumed linear behavior or calibrated servo. Same straight cycle, original actor/PD. No gain scan; actual sign/heading decides.',episodes=rows))
 return dict(status='completed',episodes=rows,student_optimizer_updates=0)
