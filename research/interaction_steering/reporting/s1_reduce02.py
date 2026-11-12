import json,time,sys,math,hashlib
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(sys.argv[1])/'runtime'))
S=Path(sys.argv[1]);run=S/'runs'/sys.argv[2];out=S/'analysis02';out.mkdir(exist_ok=False);start=time.monotonic()
rows=json.loads((run/'results.json').read_text());summ=[];phase=[];audit=[]
def stats(ph,co):
 if not ph:return None
 active=[x for x in ph if x['active']];load=[x['stage_loaded_slip_mps'] for x in ph]
 e=np.array([np.array(x['anchor'])-x['wrist'] for x in active]);f=np.array([x['force'] for x in active]);q=np.array([x['qpos'][7:] for x in ph]);dq=np.array([x['qvel'][6:] for x in ph]);pd=np.array([x['q_PD'] for x in co])
 y=math.degrees(math.atan2(ph[-1]['pelvis_matrix'][1][0],ph[-1]['pelvis_matrix'][0][0]))
 peak=max(ph,key=lambda x:x['stage_loaded_slip_mps']);contacts=peak['stage_foot_contacts'];loaded=[c for c in contacts if c['normal_N']>1e-6];event=max(loaded,key=lambda c:c['tangent_speed_mps']) if loaded else None
 return dict(ticks=len(ph),duration_s=len(ph)*.005,error_cm=float(100*np.sqrt(np.mean(np.sum(e*e,axis=1)))) if active else None,force_rms_N=float(np.sqrt(np.mean(np.sum(f*f,axis=1)))) if active else 0.,error_xyz_cm=(100*np.sqrt(np.mean(e*e,axis=0))).tolist() if active else None,force_xyz_N=np.sqrt(np.mean(f*f,axis=0)).tolist() if active else [0.,0.,0.],cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=10-1e-6)) if active else 0.,root_dx_m=ph[-1]['qpos'][0]-ph[0]['qpos'][0],root_mean_vx_mps=float(np.mean([x['qvel'][0] for x in ph])),slip_peak_mps=max(load),slip_p90_mps=float(np.quantile(load,.9)),slip_p95_mps=float(np.quantile(load,.95)),yaw_deg=y,slip_peak_contact=event,q_step_max_deg=float(np.rad2deg(np.max(np.abs(np.diff(q,axis=0))))) if len(q)>1 else 0.,dq_abs_max=float(np.max(np.abs(dq))),qpd_step_max_deg=float(np.rad2deg(np.max(np.abs(np.diff(pd,axis=0))))) if len(pd)>1 else 0.,force_evaluation='actual active ticks only; released not diluting burden')
traces={}
for row in rows:
 if row['rep']!=0:continue
 x=torch.load(run/row['artifact'],map_location='cpu',weights_only=False);ph=x['physics'];co=x['controls'];cut=x['start']+3-1e-8
 s=stats([p for p in ph if p['force_time']<cut],[c for c in co if c['time']<cut]);u=[d['guidance']['vx'] for d in x['decisions'] if d['guidance'] and d['time']<cut]
 summ.append(dict(state=row['state_id'],parent=row['parent_id'],controller=row['controller'],**s,u_mean=float(np.mean(u)) if u else None,actual_seconds=row['actual_seconds'],boundary=row['boundary'],artifact_sha=row['sha256']))
 sp=x['spec'];moveend=sp.get('stop_time',sp['onset']+sp['ramp']);hold=moveend+sp.get('deceleration',0.)
 for name,lo,hi in [('moving',sp['onset'],moveend),('deceleration',moveend,hold),('hold',hold,sp['release']),('released',sp['release'],1e10)]:
  pp=[p for p in ph if lo-1e-8<=p['force_time']<hi-1e-8];cc=[c for c in co if lo-1e-8<=c['time']<hi-1e-8]
  if pp:phase.append(dict(state=row['state_id'],controller=row['controller'],phase=name,**stats(pp,cc)))
 if row['state_id'].startswith('new_stop-') or row['state_id']=='native-standing':traces[(row['state_id'],row['controller'])]=x
 checks=[('measured_contact_force',d['causal']['force_history'][-1]) for d in x['decisions']]
 audit.append(dict(state=row['state_id'],controller=row['controller'],decisions=len(x['decisions']),causal_keys=list(x['decisions'][0]['causal']),phase_own_start=0.,guidance_spec_access=False if row['controller']=='CS' else None))
(out/'summary.json').write_text(json.dumps(summ,indent=2));(out/'phases.json').write_text(json.dumps(phase,indent=2));(out/'permission-audit.json').write_text(json.dumps(audit,indent=2))
# Display 3s matched endpoint and full phase metrics separately.
def f(x):return '—' if x is None else f'{x:.3f}'
lines=['|State|Controller|error cm|force N|root dx m|slip max m/s|yaw deg|mean u m/s|','|---|---|---:|---:|---:|---:|---:|---:|']
for s in summ:lines.append('|'+ '|'.join([s['state'],s['controller'],*[f(s[k]) for k in ['error_cm','force_rms_N','root_dx_m','slip_peak_mps','yaw_deg','u_mean']]])+'|')
(out/'tables.md').write_text('\n'.join(lines)+'\n')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
for sid in sorted({k[0] for k in traces if k[0].startswith('new_stop-')}):
 fig,ax=plt.subplots(4,1,figsize=(9,9),sharex=True)
 for kind in ['BC500','CS','CP']:
  x=traces[(sid,kind)];ph=x['physics'];tt=np.array([p['force_time'] for p in ph]);dec=x['decisions'];dt=[d['time'] for d in dec]
  if kind!='BC500':ax[0].plot(dt,[d['guidance']['vx'] for d in dec],label=kind)
  ax[1].plot(tt,[p['qvel'][0] for p in ph],label=kind)
  ax[2].plot(tt,[100*p['error'] if p['active'] else np.nan for p in ph],label=kind)
  ax[3].plot(tt,[np.linalg.norm(p['force']) if p['active'] else np.nan for p in ph],label=kind)
 for a,label in zip(ax,['u (m/s)','root vx (m/s)','active error (cm)','active force (N)']):a.set_ylabel(label);a.grid(alpha=.2);a.legend()
 ax[-1].set_xlabel('actual physical time (s)');fig.suptitle(sid+' — actual execution');fig.tight_layout();fig.savefig(out/(sid+'.png'),dpi=110);plt.close(fig)
(out/'cost.json').write_text(json.dumps({'job_s':time.monotonic()-start,'physics_steps':0,'training':False,'source_runtime':json.loads((run/'launch.json').read_text())['runtime']},indent=2))
print('\n'.join(lines))
