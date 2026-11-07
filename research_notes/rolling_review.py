"""CPU-only derived summaries and qpos movie inputs. Never calls a controller."""
import argparse,gzip,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def merge(d,label,out):
 path=d/(label+'.json.gz');ep=read(path);prefix=read(d/(label+'-source-prefix.json.gz'))
 assert prefix['physics'][-1]['time']<ep['physics'][0]['time']
 ep['physics']=prefix['physics']+ep['physics'];ep['control']=prefix['control']+ep['control'];ep['initial']=prefix['initial'];ep['movie_provenance']=dict(saved_actual_physics_only=True,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prefix_sha256=hashlib.sha256((d/(label+'-source-prefix.json.gz')).read_bytes()).hexdigest(),no_dynamics_rerun=True,omitted_common_initial_warmup='0-3s standing only; all contact and release included')
 target=out/(label+'.json.gz')
 with gzip.open(target,'wt') as f:json.dump(ep,f)
 return str(target)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();s=a.root;out=s/'runs/analysis';out.mkdir(exist_ok=True);data=s/'runs/video-input';data.mkdir(exist_ok=True)
 details=json.loads((out/'physical-details.json').read_text());compact=[]
 for row in details:
  obj={k:row[k] for k in ['label','run','sha256','condition','boundary','events']};obj['phases']={}
  keys=['seconds','error_cm','error_xyz_rms_cm','signed_error_cm','force_rms','force_xyz_rms','cap_fraction','force_squared_time_average_N2','force_absolute_impulse_Ns','root_delta_cm','root_path_cm','root_forward_speed_mean','root_body_forward_speed_mean','foot_paths_cm','unloaded_fraction','both_unloaded_fraction','pelvis_yaw_delta_deg','torso_yaw_delta_deg','max_tilt_deg','min_height','wrist_relative_shoulder_delta_cm','qpd_peak','qpd_p95_deg','qpd_p99_deg','actual_joint_peak','actual_joint_speed_max_radps','actuator_abs_work_proxy_J','external_work_J','torque_saturation_max','slip_proxy_mean','slip_proxy_p95']
  for ph,x in row['phases'].items():obj['phases'][ph]={k:x[k] for k in keys if k in x} if x else None
  compact.append(obj)
 write(out/'physical-summary.json',compact)
 searches=json.loads((s/'runs/rolling-02/searches.json').read_text());selection={}
 for label in ['rolling_continue','rolling_stop']:
  qs=[x for x in searches if x['label']==label];kinds=Counter(x['candidates'][x['selected']]['kind'] for x in qs);bases=Counter(x['candidates'][x['selected']]['parameters'].get('base','motion') for x in qs if x['candidates'][x['selected']]['kind']=='sequence');segments=[]
  for x in qs:
   c=x['candidates'][x['selected']];key=c['kind']+(':'+c['parameters'].get('base','motion') if c['kind']=='sequence' else '')
   if not segments or segments[-1]['kind']!=key:segments.append(dict(start=x['time'],end=x['time']+.1,kind=key))
   else:segments[-1]['end']=x['time']+.1
  selection[label]=dict(plans=len(qs),kinds=dict(kinds),sequence_bases=dict(bases),segments=segments,median_plan_wall_s=float(np.median([x['wall_s'] for x in qs])),total_selected_predicted_score=float(sum(x['candidates'][x['selected']]['score'] for x in qs)),note='Sum of overlapping predicted horizons is NOT an episode return')
 write(out/'selection-summary.json',selection)
 # Actual matched standing onset against qualification reference, independent of metadata.
 ref=read(s/'runs/qualification-03/C32__speed0.15.json.gz');act=read(s/'runs/rolling-02/rolling_continue.json.gz')
 def fields(ep):return [{k:r[k] for k in ['time','qpos','qvel','force','feet_xyz','pelvis_matrix','torso_matrix']} for r in ep['physics'] if r['force_time']<5-1e-8]
 rc=[r for r in ref['control'] if r['time']<5-1e-8];ac=[r for r in act['control'] if r['time']<5-1e-8]
 write(out/'standing-onset-parity.json',dict(physical_fields_equal=fields(ref)==fields(act),full_recorded_control_prefix_equal=rc==ac,including_BFM_observation_history_action_z=True,ticks_reference=len(fields(ref)),ticks_rolling=len(fields(act)),control_steps=len(rc),fields=['time','qpos','qvel','force','feet_xyz','pelvis_matrix','torso_matrix']))
 q=s/'runs/qualification-03';r=s/'runs/rolling-02';paths={}
 for label in ['motion__speed0.09','motion__speed0.15','motion__speed0.22','I__speed0.15','C32__speed0.15']:paths[label]=merge(q,label,data)
 for label in ['rolling_continue','rolling_stop']:paths[label]=merge(r,label,data)
 def side(label,title,role):return dict(path=paths[label],title=title,role=role)
 pairs=[dict(name='01_native_source_speeds',grid4=True,cross_condition_description='Same72cm but different mean speeds/durations; full episodes at1x. Ended panels hold last frame, not continued control.',sides=[side('motion__speed0.09','Native motion | avg 0.09m/s | source failure','REAL NATIVE SOURCE'),side('motion__speed0.15','Native motion | avg 0.15m/s | source reference','REAL NATIVE SOURCE'),side('motion__speed0.22','Native motion | avg 0.22m/s | source reference','REAL NATIVE SOURCE'),side('I__speed0.15','Old I | avg 0.15m/s | frozen reference','FROZEN I')]),dict(name='02_native_feedback_comparison',grid4=True,cross_condition_description='Same72cm/avg.15/onset5.0/10N, actual standing prefix matched. Motion scheduled entry/return is a qualification reference; MPC has no stop clock.',sides=[side('C32__speed0.15','C32 | old native PPO frozen reference','FROZEN C32'),side('motion__speed0.15','Native motion | qualification source','REAL NATIVE SOURCE'),side('rolling_continue','Native MPC | first command only, then replan','REAL MPC'),side('I__speed0.15','Old I | frozen capability reference','FROZEN I')]),dict(name='03_common_prefix_stop_continue',cross_condition_description='One identical MPC program. Same actual inputs and commands until7.8s; then real partner smoothly stops in second branch. No future stop info given to planner.',sides=[side('rolling_continue','Native MPC | partner continues | fixed world + overhead','REAL MPC'),side('rolling_stop','Same native MPC | partner brakes at 7.8s','REAL MPC')])]
 write(data/'video-pairs.json',pairs)
 # Physics-derived curves for inspection, no policy/reference animation.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(4,1,figsize=(11,11),sharex=True)
 for label,color in [('C32__speed0.15','gray'),('motion__speed0.15','tab:orange'),('rolling_continue','tab:blue'),('I__speed0.15','tab:green')]:
  ep=read(Path(paths[label]));rr=[x for x in ep['physics'] if x['force_time']>=4];tt=np.array([x['force_time'] for x in rr]);f=np.array([x['force'] for x in rr]);pos=np.array([x['root'] for x in rr]);err=np.array([np.linalg.norm(np.array(x['anchor'])-x['wrist'])*100 if x['active'] else np.nan for x in rr]);yaw=np.unwrap([np.arctan2(np.array(x['pelvis_matrix'])[1,0],np.array(x['pelvis_matrix'])[0,0]) for x in rr]);axes[0].plot(tt,err,label=label,color=color);axes[1].plot(tt,np.linalg.norm(f,axis=1),color=color);axes[2].plot(tt,100*(pos[:,0]-pos[0,0]),color=color);axes[3].plot(tt,np.degrees(yaw),color=color)
 for ax,y in zip(axes,['Active3D error (cm)','True force norm (N)','Root X displacement (cm)','Pelvis yaw (deg)']):
  ax.set_ylabel(y);ax.grid(alpha=.3);ax.axvline(5,color='k',ls=':',lw=.8);ax.axvline(9.8,color='k',ls='--',lw=.8);ax.axvline(11.8,color='k',ls=':',lw=.8)
 axes[0].legend(fontsize=8);axes[-1].set_xlabel('Actual physics time (s)');fig.tight_layout();fig.savefig(out/'same_condition_curves.png',dpi=150);plt.close(fig)
 write(out/'review-result.json',dict(status='completed',real_physics_only=True,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),videos_prepared=3))
 print('REVIEW_COMPLETE',len(compact),selection,flush=True)
if __name__=='__main__':main()
