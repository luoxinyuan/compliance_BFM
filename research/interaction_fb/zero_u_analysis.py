"""CPU-only analysis of frozen matched zero-U qualification; no dynamics or updates."""
import gzip,json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT.parents[2];S=P/'experiments/r2-online-20261114'
OLD=S/'runs/initial-qualification-01';NEW=S/'runs/zero-u-qualification-01'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def angles(rows):return np.unwrap([np.arctan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in rows])
def quality(ph,cs):
 yaw=angles(ph);root=np.asarray([p['root'] for p in ph]);slip=np.asarray([p['stage_loaded_slip_mps'] for p in ph]);q=np.asarray([p['qpos'][7:] for p in ph]);qp=np.asarray([c['q_PD'] for c in cs]);act=[p for p in ph if p['active']]
 e=np.asarray([np.asarray(p['contacts'][0]['anchor'])-p['contacts'][0]['point'] for p in act]);f=np.asarray([p['force'] for p in act])
 return dict(physics=len(ph),active_physics=len(act),force_xyz_rms=np.sqrt((f*f).mean(0)).tolist() if act else None,error_xyz_rms_cm=(100*np.sqrt((e*e).mean(0))).tolist() if act else None,yaw_delta_deg=float(np.degrees(yaw[-1]-yaw[0])),yaw_excursion_deg=float(np.degrees(np.max(abs(yaw-yaw[0])))),root_xy_path_m=float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=-1).sum()),root_delta_m=(root[-1]-root[0]).tolist(),slip_peak_mps=float(slip.max()),slip_p95_mps=float(np.quantile(slip,.95)),qpd_maxstep_rad=float(np.max(abs(np.diff(qp,axis=0)))) if len(qp)>1 else None,actual_q_maxstep_rad=float(np.max(abs(np.diff(q,axis=0)))) if len(q)>1 else None)
def phases(t):
 s=t['spec'];stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0);out={}
 for k,a,b in [('precontact',s['source_until'],s['onset']),('moving',s['onset'],stop),('decelerating',stop,hold),('holding',hold,s['release']),('released',s['release'],s['end'])]:
  ph=[p for p in t['physics'] if a<=p['force_time']<b];cs=[c for c in t['controls'] if a<=c['time']<b]
  if ph:out[k]=quality(ph,cs)
 return out
start=time.monotonic();old=load(OLD/'qualification.json');new=load(NEW/'qualification.json');assert new['zero_u'] and new['optimizer_updates']==0
rows=[]
for i,(a,b) in enumerate(zip(old['results'],new['results'])):
 paths=[d/f'case-{i}'/'actual-0.json.gz' for d in (OLD,NEW)]
 traces=[]
 for p in paths:
  with gzip.open(p,'rt') as f:traces.append(json.load(f))
 x,y=traces;assert x['spec']==y['spec']
 keys=('bfm_obs','pre_qpos','pre_qvel','time')
 assert all(x['controls'][0][k]==y['controls'][0][k] for k in keys)
 c0,c1=x['controls'][0],y['controls'][0]
 r=dict(case=i,family=a['family'],parent_id=a['parent_id'],initial_obs_q_qvel_time_exact=True,source_state_sha_equal=a['source_state_sha']==b['source_state_sha'],source_state_sha_old=a['source_state_sha'],source_state_sha_new=b['source_state_sha'],initial_action_max_abs=float(np.max(abs(np.asarray(c0['action'])-c1['action']))),initial_qpd_max_abs_rad=float(np.max(abs(np.asarray(c0['q_PD'])-c1['q_PD']))),traces_sha256=[sha(p) for p in paths])
 for label,metrics,t in [('A128',a,x),('zero_U',b,y)]:
  r[label]=dict(error_cm=metrics['all']['error_cm'],force_N=metrics['all']['force_rms'],cap_fraction=metrics['all']['cap_fraction'],yaw_abs_final_deg=metrics['yaw_abs_final'],clean_success=metrics['clean_success'],boundary=metrics['boundary'],partner_progress_m=metrics['partner_progress_m'],wrist_progress_m=metrics['wrist_progress_m'],whole=quality(t['physics'],t['controls']),phases=phases(t))
 rows.append(r)
result=dict(rows=rows,cases=len(rows),new_physics=0,new_updates=0,analysis_s=time.monotonic()-start,zero_u_identity=load(NEW/'initial-function.json'),physical_runtime='1a72d0fe4b7b4473f80a50fd47da4148c05889c6',reference_runtime='852c4572f725f176cdab5076c33b5c93a15670a0',qualification_sha256=sha(NEW/'qualification.json'))
(S/'analysis/zero-u-paired-results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(cases=len(rows),source_full_hash_matches=sum(r['source_state_sha_equal'] for r in rows),rows=[{k:r[k] for k in ('family','source_state_sha_equal','A128','zero_U')} for r in rows]),indent=2))
