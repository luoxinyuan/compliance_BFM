"""M7 teacher algorithm candidate: reverse force retraces original native path.
Not an engineering fix. Original positive entry/low-force release retained;
negative evidence has the same debounce/threshold and path rate as positive.
No task clock, anchor, or mode label. Student does NOT call this gate.
"""
from contact_gate import ContactGate,inverse_smooth,smooth
class DirectionalGate(ContactGate):
 def __init__(self,cfg):
  super().__init__(cfg);self.reverse_count=0;self.reversing=False
 def step(self,measured_force_xyz):
  prior_alpha=self.alpha;was_reversing=self.reversing
  out=super().step(measured_force_xyz)
  negative=out['filtered_force_xyz'][0]<=-self.cfg['entry_forward_force_n']
  self.reverse_count=self.reverse_count+1 if negative else 0
  self.reversing=self.reverse_count>=self.cfg['entry_samples']
  if self.reversing:
   self.u=max(0.,inverse_smooth(prior_alpha)-self.cfg['dt']/self.cfg['entry_transition_s']);self.alpha=smooth(self.u)
   self.state='active' if self.alpha>0 else 'standing';self.low_count=0;self.enter_count=0
   out.update(alpha=self.alpha,state=self.state,event='reverse' if not was_reversing else None,low_count=0,enter_count=0)
  out.update(reverse_count=self.reverse_count,reversing=self.reversing)
  return out
