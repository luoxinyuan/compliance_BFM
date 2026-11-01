"""One bounded operating-range test: sustained .25m/s guide, unchanged5N force cap.
It changes the partner profile, not controller gains, BFM, PD, or old evidence.
"""
import json,torch
from m9_stage import RUN
from m9_contact import run as contact
def run(env,model,contract,out):
 refs=json.loads((RUN/'native-cycles.json').read_text())['cycles']
 for r in refs.values():r['tensor']=torch.tensor(r['z'],device=env.standing.device)
 return contact(env,model,contract,out,cycles=refs,composition=True)

def sustained_profile(t,start,vmax=.25,accel=1.,cruise=6.):
 x=t-start
 if x<=0:return 0.,0.
 if x<accel:
  u=x/accel;return vmax*accel*(u**3-.5*u**4),vmax*(3*u*u-2*u**3)
 if x<accel+cruise:return vmax*(.5*accel+x-accel),vmax
 u=min((x-accel-cruise)/accel,1.)
 return vmax*(.5*accel+cruise+accel*(u-u**3+.5*u**4)),vmax*(1-3*u*u+2*u**3) if u<1 else 0.
