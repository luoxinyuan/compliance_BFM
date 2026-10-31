"""Saved-trajectory behavioral mechanism audit, no model or simulator execution."""
import gzip,json
from pathlib import Path
import numpy as np
from m6_analysis import RUN,read

def main():
 ledger=read(RUN/'compute-ledger.json');ans={}
 for a in ledger['attempts']:
  if a['stage'] not in ['qualification','projected','final']:continue
  eps={}
  sources=[(p,None) for p in (Path(a['path'])/'outputs').glob('*.json.gz')]
  if a['stage']=='final':sources += [(Path(r['source']),r) for r in read(Path(__file__).with_name('m6_protocol.json'))['reused_exact_physical_controls']]
  for p,reuse in sources:
   e=read(p)
   if reuse:e['controller']=reuse['controller'];e['label']=reuse['condition']+'__'+reuse['controller']
   eps[e['label']]=e
  for label,e in eps.items():
   cs=e['control'];rows=e['physics'];on=e['condition']['onset'];rel=e['condition']['release'];active=[r for r in rows if on-1e-8<=r['force_time']<rel-1e-8 and r['active']]
   q=np.array([c['q_PD'] for c in cs]);q0=np.array([c['standing_q_PD'] for c in cs]);cc=[i for i,c in enumerate(cs) if on<=c['time']<rel];delta=q[cc]-q0[cc]
   x=dict(action_vs_same_state_standing_arm_rms_deg=float(np.degrees(np.sqrt(np.mean(delta[:,22:]**2)))) if len(cc) else 0,action_vs_same_state_standing_nonarm_rms_deg=float(np.degrees(np.sqrt(np.mean(delta[:,:22]**2)))) if len(cc) else 0,progress_max=max((c['native_coords'][0] if c['native_coords'] is not None else c['alpha'] if e['controller'] in ['teacher','rule','projected','force_rule'] else 0) for c in cs),projection_arm_rms_deg_max=max((c['projection']['arm_rms_deg'] for c in cs if c.get('projection')),default=0),projection_nonarm_rms_deg_max=max((c['projection']['nonarm_rms_deg'] for c in cs if c.get('projection')),default=0),controller=e['controller'],condition=e['condition'],complete=e['boundary'] is None)
   if active:
    f=np.array([r['force'] for r in active]);x.update(force_xyz_mean=f.mean(0).tolist(),force_x_positive_fraction=float(np.mean(f[:,0]>0)),force_x_negative_fraction=float(np.mean(f[:,0]<0)))
   base=eps.get(e['condition']['name']+'__native_hold')
   if base:
    times=np.array([r['time'] for r in rows]);ix=int(np.argmin(abs(times-rel)));br=base['physics'][ix];r=rows[ix];x['extra_motion_vs_native_at_release_cm']={k:((np.array(r[k])-br[k])*100).tolist() for k in ['wrist','shoulder_xyz','root']};x['extra_wrist_relative_root_cm']=((np.array(r['wrist'])-r['root']-np.array(br['wrist'])+br['root'])*100).tolist()
   ans[a['stage']+'/'+label]=x
 (RUN/'analysis/mechanism.json').write_text(json.dumps(ans,indent=2));print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
