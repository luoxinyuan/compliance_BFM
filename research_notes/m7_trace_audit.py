"""Independent saved-data audit: actual applied force, causal histories, takeover continuity."""
import json,gzip,hashlib
from pathlib import Path
import numpy as np
from m7_analysis import RUN,read

def main():
 ledger=read(RUN/'compute-ledger.json');res={};prefix=[]
 for a in ledger['attempts']:
  for p in sorted((Path(a['path'])/'outputs').glob('*__*.json.gz')):
   e=read(p);rr=e['physics'];cs=e['control'];s=e['condition'];ts=np.array([r['force_time'] for r in rr]);post=np.array([r['time'] for r in rr]);active=np.array([r['active'] for r in rr]);f=np.array([r['force'] for r in rr]);wrench=np.array([r['applied_wrist_wrench'] for r in rr]);expected=(ts>=s['onset'])&(ts<s['release'])&s['enabled'];tr=s.get('transition')
   if tr:expected &= ~((ts>=tr['unload'])&(ts<tr['restart']))
   # Simulator float timing can differ at a single exact boundary: record, inspect, not relabel.
   mismatches=np.flatnonzero(active!=expected).tolist();sensors=[];history=[];fifo=np.zeros((320,3))
   for c in cs:
    i=int(np.argmin(abs(post-c['time'])));last=rr[i]['force'] if abs(post[i]-c['time'])<1e-7 else [0,0,0]
    sensors.append(float(abs(np.array(c['measured_force_xyz'])-last).max()));fifo=np.roll(fifo,-1,axis=0);fifo[-1]=c['measured_force_xyz'];history.append(float(abs(np.array(c['student_input'][465:]).reshape(320,3)-fifo/5).max()))
   v=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_wrench_force_max_error=float(abs(f-wrench[:,:3]).max()),off_force_max=float(abs(f[~active]).max()) if (~active).any() else 0.,active_contract_mismatch_times=ts[mismatches].tolist(),sensor_last_physics_max_error=max(sensors),network_force_history_max_error=max(history),ready_sha=e['ready_physical_state_sha256'],bfm_input_keys=sorted(cs[0]['bfm_obs']),feature_dimension=len(cs[0]['student_input']),boundary=e['boundary'])
   assert v['actual_wrench_force_max_error']==0 and v['off_force_max']==0 and v['sensor_last_physics_max_error']<1e-6 and v['network_force_history_max_error']<1e-6
   assert len(mismatches)==0,v
   if s['name']=='free':
    old=read(RUN.parent/'m5-contact-lpp-20261031A/final-attempt-01/outputs/free__native_hold.json.gz')
    v['free_vs_native']={k:float(abs(np.array([r[k] for r in rr])-np.array([r[k] for r in old['physics']])).max()) for k in ['qpos','qvel','wrist','force']}
   res[p.parent.parent.name+'/'+e['label']]=v
   if '_then_' in e['controller']:
    old=read(RUN/'qualification-attempt-01/outputs'/(s['name']+'__m6_alignment.json.gz'));base=[r for r in old['physics'] if r['force_time']<tr['restart']-1e-8];cur=[r for r in rr if r['force_time']<tr['restart']-1e-8]
    diff={k:float(abs(np.array([r[k] for r in base])-np.array([r[k] for r in cur])).max()) for k in ['qpos','qvel','force','wrist']};assert max(diff.values())==0,diff
    prefix.append(dict(source=str(p),paired=str(RUN/'qualification-attempt-01/outputs'/(s['name']+'__m6_alignment.json.gz')),until=tr['restart'],errors=diff))
 (RUN/'analysis/trace-audit.json').write_text(json.dumps(dict(episodes=res,takeover_prefix=prefix),indent=2));print('AUDITED',len(res),'episodes; takeovers',len(prefix))
if __name__=='__main__':main()
