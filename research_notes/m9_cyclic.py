"""Bounded reuse of native walking references, with actual feedback and no end timer."""
import json,math
import numpy as np,torch
from m9_stage import RUN,episode
from locomotion_rollout import spherical
from contact_gate import smooth
from m1_stage import save
from m9_contact import NativeFeedback

class CyclicFeedback(NativeFeedback):
 def __init__(self,periods,privileged=False):super().__init__();self.periods=periods;self.privileged=privileged
 def step(self,force_world,R,dt=.02,guide=None):
  self.filtered+=dt/(.25+dt)*(np.asarray(force_world)-self.filtered);f=R.T@self.filtered
  target=float(np.clip((f[0]-.25)/2.,0,1));self.alpha=float(np.clip(target,self.alpha-.9*dt,self.alpha+.6*dt))
  if self.privileged:
   assert guide is not None
   desired=R.T@guide;turn=float(np.clip(np.arctan2(desired[1],max(desired[0],.02))/.35,-1,1))
  else:turn=float(np.clip(f[1]/max(abs(f[0]),1.),-1,1))
  self.turn=float(np.clip(turn,self.turn-dt,self.turn+dt));period=(1-abs(self.turn))*self.periods['straight']+abs(self.turn)*self.periods['left' if self.turn>=0 else 'right']
  if self.alpha>.05:self.phase+=dt/period
  return dict(alpha=self.alpha,phase=self.phase,turn=self.turn,force_body_filtered=f.tolist(),reference_exhausted=False,phase_unit='cycles',privileged_expert=self.privileged)

def sample(ref,phase):
 z=ref['tensor'];x=(phase%1)*len(z)+ref['phase_offset'];i=int(x)%len(z);u=x-int(x);return spherical(z[i:i+1],z[(i+1)%len(z):(i+1)%len(z)+1],u)

def run(env,model,contract,out):
 from m9_contact import run as contact
 refs=json.loads((RUN/'native-cycles.json').read_text())['cycles'];rows=[]
 for name,r in refs.items():
  r['tensor']=torch.tensor(r['z'],device=env.standing.device)
  def cmd(t,e):
   if t<4.5:return spherical(e.standing,sample(r,0),smooth((t-3)/1.5))
   if t<10.5:return sample(r,(t-4.5)/r['period'])
   return spherical(sample(r,6/r['period']),e.standing,smooth((t-10.5)/1.5))
  rows.append(episode(env,model,contract,out,'cycle-'+name,cmd,16.,dict(entry=[3,4.5],execute=[4.5,10.5],exit=[10.5,12],recover=[12,16])))
  save(out,'cyclic-native-summary.json',dict(episodes=rows))
 # This is a composition qualification, not a global requirement for all M9 skills.
 if not all(r['complete'] for r in rows):return dict(status='completed',episodes=rows,student_optimizer_updates=0,contact_skipped_reason='A proposed cyclic blend endpoint was unsafe; finite original reference successes retained, no claim all BFM locomotion fails')
 result=contact(env,model,contract,out,cycles=refs);result['native_cycles']=rows;return result
