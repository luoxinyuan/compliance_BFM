"""Saved-state qPD discontinuity attribution; no rollout or parameter changes."""
import json,gzip
import numpy as np
from m9c_stage import RUN

def main():
 items=[]
 for stage,labels in [('qualification',['held_straight__expert','held_straight__arm']),('rl_eval',['held_straight__rl']),('final',['new_straight__expert','new_straight__learned_expert','new_straight__transfer','new_heading__transfer','new_right__transfer','wrist_early__ablation','wrist_early__transfer'])]:
  for label in labels:
   p=RUN/(stage+'-attempt-01')/'outputs'/(label+'.json.gz')
   if not p.exists():continue
   e=json.loads(gzip.decompress(p.read_bytes()));cs=e['control'];dq=np.array([c['q_PD_step_max_rad'] for c in cs]);top=np.argsort(dq)[-5:][::-1];events=[]
   for i in top:
    c=cs[i];prev=cs[max(i-1,0)];s=c['oracle_controller_state'];ps=prev['oracle_controller_state'];z=np.asarray(c['executed_z']);zp=np.asarray(prev['executed_z']);q=np.asarray(c['q_PD']);qp=np.asarray(prev['q_PD']);j=int(np.argmax(abs(q-qp)))
    events.append(dict(time=c['time'],phase=next((k for k,(a,b) in e['phases'].items() if a<=c['time']<b),'other'),qpd_step_deg=float(np.degrees(dq[i])),joint_index=j,qpd_actual_deg=float(np.degrees(abs(np.asarray(c['pre_qpos'])[7:]-np.asarray(prev['pre_qpos'])[7:])).max()),z_step_deg=float(np.degrees(np.arccos(np.clip((z*zp).sum()/256,-1,1)))),oracle_phase=s['phase'],oracle_phase_delta=s['phase']-ps['phase'],cycle_crossing=int(s['phase'])!=int(ps['phase']),alpha=s['alpha'],alpha_delta=s['alpha']-ps['alpha'],turn=s['turn'],turn_delta=s['turn']-ps['turn'],force_xyz=c['measured_force_xyz']))
   items.append(dict(stage=stage,label=label,events=events,jumps_over_30_deg=int((dq>np.radians(30)).sum()),jumps_over_30_at_phase_crossing=sum(int(cs[i]['oracle_controller_state']['phase'])!=int(cs[i-1]['oracle_controller_state']['phase']) for i in range(1,len(cs)) if dq[i]>np.radians(30))))
 (RUN/'jump-audit.json').write_text(json.dumps(items,indent=2));print(json.dumps(items,indent=2))
if __name__=='__main__':main()
