"""Post-test saved-trace timing localization; no new rollout or policy update."""
import json
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-alpha-cross-20261031A')
p=json.loads(Path(__file__).with_name('alpha_cross_protocol.json').read_text());out={}
for c in p['conditions']:
 if c['split']!='sealed_final':continue
 for ctl in ['A','B']:
  key=c['name']+'__'+ctl;ep=json.loads((R/'final-attempt-01/outputs'/(key+'.json')).read_text());cs=ep['control'];t=np.array([v['time'] for v in cs]);a=np.array([v['alpha'] for v in cs]);y=np.array([v['shadow_alpha'] for v in cs]);e=a-y
  lo,hi=c['disconnect_start'],c['disconnect_end'];phases={}
  for name,start,end in [('gap',lo,hi),('resume1s',hi,hi+1),('remaining_contact',hi+1,10.4),('release',10.4,12.4)]:
   m=(t>=start-1e-8)&(t<end-1e-8);phases[name]=dict(alpha_rms=float(np.sqrt(np.mean(e[m]**2))),mean_signed_alpha_error=float(e[m].mean()),max_abs_error=float(abs(e[m]).max()),min_alpha=float(a[m].min()))
  m=(t>=lo-1e-8)&(t<hi+1-1e-8);bad=np.flatnonzero(m&(abs(e)>.1))
  indices={'last_before':int(np.flatnonzero(t<hi-1e-8)[-1]),'first_at_or_after':int(np.flatnonzero(t>=hi-1e-8)[0])}
  out[key]=dict(phases=phases,first_abs_error_over_0_1=None if not len(bad) else dict(time=float(t[bad[0]]),seconds_after_reconnect=float(t[bad[0]]-hi),signed_error=float(e[bad[0]])),end_brackets={name:dict(actual_control_time=float(t[i]),alpha=float(a[i]),shadow=float(y[i])) for name,i in indices.items()})
(R/'analysis/resumption-localization.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
