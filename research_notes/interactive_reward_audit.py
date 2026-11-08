"""Read-only pinned replay fragments and saved native execution; no GPU/physics/model."""
import json,gzip,time
from pathlib import Path
import numpy as np,h5py
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/interactive-contact-pilot-20261108';OLD=P/'experiments/native-rolling-h100-20261107';out=S/'runs/reward-readonly-02';out.mkdir(exist_ok=False);started=time.monotonic()
def body_velocity(q,v):
 w,x,y,z=q[:,3:7].T;R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3);return np.einsum('nji,nj->ni',R,v[:,:3])
def rewards(q,v):
 b=body_velocity(q,v);return np.exp(-((b[:,0]-.15)/.10)**2-(b[:,1]/.10)**2-(v[:,5]/.25)**2)
a=OLD/'runs/reward-inference-01';ix=np.load(a/'sample-indices.npy');info=[]
with h5py.File('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train/buffer.hdf5','r') as f:
 for config in ['forward015','forward022','forward015_arm']:
  rw=np.load(a/(config+'-reward.npy')).reshape(-1);ranks=np.argsort(rw)[::-1];seen=set();n=0
  for index in ranks:
   t,env=map(int,ix[index]);key=(env,t//60)
   if key in seen or t<31 or t>4968:continue
   seen.add(key);q=np.asarray(f['qpos'][t-30:t+31,env]);v=np.asarray(f['qvel'][t-30:t+31,env]);valid=True
   trunc=f.get('truncated');reset_count=int(np.count_nonzero(trunc[t-30:t+30,env])) if trunc is not None else None
   rr=rewards(q,v);b=body_velocity(q,v);center=rewards(q[30:31],v[30:31])[0]
   info.append(dict(config=config,sampled_index=int(index),storage_index=[t,env],original_center_reward=float(rw[index]),forward015_center=float(center),forward015_window_mean=float(rr.mean()),forward015_window_fraction_above_half=float((rr>.5).mean()),window_body_v_mean=b.mean(0).tolist(),window_body_v_std=b.std(0).tolist(),world_root_displacement_cm=(100*(q[-1,:3]-q[0,:3])).tolist(),window_yaw_rate_mean=float(v[:,5].mean()),window_yaw_rate_std=float(v[:,5].std()),truncation_count=reset_count,nominal_window_s=1.2,raw_fragment_not_published=True))
   n+=1
   if n==5:break
summaries=[]
for p in (OLD/'runs/qualification-03').glob('*.json.gz'):
 if 'source-prefix' in p.name:continue
 with gzip.open(p,'rt') as f:ep=json.load(f)
 rs=[r for r in ep.get('physics',[]) if r['active'] and r['force_time']<ep['condition']['onset']+ep['condition']['ramp']]
 if not rs:continue
 q=np.array([r['qpos'] for r in rs]);v=np.array([r['qvel'] for r in rs]);r=rewards(q,v);b=body_velocity(q,v)
 summaries.append(dict(episode=p.name,robot_reward015_mean=float(r.mean()),reward015_p90=float(np.quantile(r,.9)),fraction_above_half=float((r>.5).mean()),body_velocity_mean=b.mean(0).tolist(),mean_yaw_rate=float(v[:,5].mean()),active_3d_error_cm=float(100*np.sqrt(np.mean([r0['error']**2 for r0 in rs]))),force_rms=float(np.sqrt(np.mean([np.dot(r0['force'],r0['force']) for r0 in rs]))),source='saved actual execution; robot-state prompt only not complete contact reward'))
# Apply the identical robot-state prompt to actual successful rolling control and frozen P4000.
for path in [OLD/'runs/rolling-02/rolling_continue.json.gz',P/'experiments/p4000-feedback-ppo-20261108/runs/single5201-independent-32/dev-0000/normal72__mean.json.gz']:
 with gzip.open(path,'rt') as f:ep=json.load(f)
 rs=[r for r in ep['physics'] if r['active'] and r['force_time']<ep['condition']['onset']+ep['condition']['ramp']]
 q=np.array([r['qpos'] for r in rs]);v=np.array([r['qvel'] for r in rs]);r=rewards(q,v);b=body_velocity(q,v)
 summaries.append(dict(episode=str(path),robot_reward015_mean=float(r.mean()),reward015_p90=float(np.quantile(r,.9)),fraction_above_half=float((r>.5).mean()),body_velocity_mean=b.mean(0).tolist(),mean_yaw_rate=float(v[:,5].mean()),active_3d_error_cm=float(100*np.sqrt(np.mean([r0['error']**2 for r0 in rs]))),force_rms=float(np.sqrt(np.mean([np.dot(r0['force'],r0['force']) for r0 in rs]))),source='actual successful motion/contact feedback, fixed robot prompt diagnostic only'))
result=dict(status='completed',wall_s=time.monotonic()-started,read_replay_fragments=15,replay_steps_read=15*61,new_reward_inference=0,new_physics_steps=0,GPU_calls=0,fragments=info,native_actual=summaries,scope='top5 distinct env/time bins per existing prompt; not unbiased replay survey; no proof aggregation is unique failure')
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result)[:1500])
