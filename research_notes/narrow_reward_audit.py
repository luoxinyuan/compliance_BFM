"""Independent reconstruction from recorded physics/control, no reward helper import."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np
def audit(path):
 with gzip.open(path,'rt') as f:e=json.load(f)
 p=e['physics'];cs=e['control'];ss=e['rl_stats'];errors={};cost={};prev=None;total=0.;force_xyz=[]
 for i,c in enumerate(cs):
  rs=p[i*4:(i+1)*4];assert rs
  f=np.array([r['force'] for r in rs]);err=np.array([r['anchor'] for r in rs])-np.array([r['wrist'] for r in rs]);q=np.array(c['q_PD']);nom=np.array(c['nominal_q_PD']);vp=np.array(c['post_partner_velocity']);v=np.array(c['post_velocity']);active=c['post_active'];rv=np.array(c['post_root_velocity']);d=np.array(c['uncapped_demand_xyz']);dq=None if prev is None else q-prev
  mw=min(np.linalg.norm(rv[:2])/.12,1) if active and np.linalg.norm(vp[:2])>.02 else 0.
  terms=dict(force=1.5*np.mean(np.sum(f*f,1))/25,uncapped=.15*min(math.log1p(d@d/25),8),tracking=.15*min(np.mean(np.sum(err*err,1))/.04,16) if active else 0.,relative_velocity=.1*min((vp-v)@(vp-v)/.25,16) if active else 0.,heading=.1*mw*(1-math.cos(c['post_yaw']-math.atan2(vp[1],vp[0]))),slip=.05*min((max(r['contact_tangent_speed_max_mps'] for r in rs)/.15)**2,16),tilt=.1*min((max(0,rs[-1]['tilt_deg']-15)/15)**2,16),continuity=c['reward_terms']['continuity'] if dq is None else .05*min(np.mean(dq*dq)/.01,16),soft_prior=.02*min(np.mean((q-nom)**2)/.04,16),idle=.2*min(rv@rv/.04,16) if not active else 0.)
  for k,v in terms.items():errors[k]=max(errors.get(k,0),abs(v-c['reward_terms'][k]));cost[k]=cost.get(k,0)+.02*v
  total-=.02*sum(terms.values());prev=q
  force_xyz.extend(r['force'] for r in rs if r['active'])
 rl=sum(s['reward'] for s in ss);control=sum(c['reward_integral'] for c in cs);tail=rl-control
 f=np.array(force_xyz);a=[r for r in p if r['active']]
 return dict(label=e['label'],boundary=e['boundary'],terms_max_error=errors,reconstructed_return=total,control_return=control,ppo_reward_sum=rl,terminal_extra=tail,return_error=abs(total-control),cost_integrals=cost,force_xyz_rms=np.sqrt(np.mean(f*f,0)).tolist(),force_rms=float(np.sqrt(np.mean(np.sum(f*f,1)))),error_cm=100*float(np.sqrt(np.mean([r['error']**2 for r in a]))),cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=4.999)),first_continuity_source='logged initial cost only; all subsequent controls independently reconstructed',rl_decisions=len(ss),control_steps=len(cs),physics_steps=len(p))
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();rows=[audit(f) for f in sorted(a.folder.glob('*.json.gz'))];assert rows
 out=dict(rows=rows,discount='0.99 per full 0.1s RL interval, .99**(control_steps/5) for partial; GAE lambda .95; rewards integrated at .02s control dt',source='actual saved physics/control and PPO step rewards; no call to runtime reward function')
 (a.folder/'independent-reward-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
 assert max(max(r['terms_max_error'].values()) for r in rows)<1e-5
 assert max(r['return_error'] for r in rows)<1e-5
if __name__=='__main__':main()
