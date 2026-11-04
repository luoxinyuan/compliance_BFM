"""Independent recorded-force/anchor/latent audit; no physics or learning."""
import gzip,json,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);items=[]
for O in [R/'physical-01',R/'physical-02']:
 for p in sorted(O.glob('*.json.gz')):
  if p.name.endswith('-source-prefix.json.gz') or p.name=='timelines.json.gz':continue
  e=json.load(gzip.open(p,'rt'))
  if 'physics' not in e:continue
  s=e['condition'];a=[r for r in e['physics'] if r['active']];ferr=0.;bases=[]
  for r in a:
   c=r['contacts'][0];raw=80*(np.array(c['anchor'])-c['point'])+8*(np.array(c['anchor_velocity'])-c['velocity']);f=raw*min(1.,5/max(np.linalg.norm(raw),1e-12));f=np.zeros(3) if s.get('physical_force_off') else f;ferr=max(ferr,float(np.max(abs(f-c['force']))))
   u=np.clip((r['force_time']-s['onset'])/s['ramp'],0,1);bases.append(np.array(c['anchor'])-[s['distance']*(3*u*u-2*u**3),0,0])
  bs=np.array(bases);drift=float(abs(bs-bs[0]).max());norm=max(abs(np.linalg.norm(c['executed_z'])-16) for c in e['control']);assert ferr<1e-9,(p,ferr);assert drift<1e-8,(p,drift);assert norm<1e-4,(p,norm)
  items.append(dict(label=e['label'],force_recompute_max=ferr,anchor_base_drift=drift,latent_norm_error=norm,physical_force_off=s.get('physical_force_off',False)))
(R/'validation.json').write_text(json.dumps(items,indent=2));print('passed',len(items))
