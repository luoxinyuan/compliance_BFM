"""Read-only saved-trajectory analysis; no model or simulation execution."""
import json,gzip,sys
from pathlib import Path
import numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/narrow-contact-expert-20261102')
paths={
 'native':'audit-attempt-02/outputs/job/audit-physical/single__prior.json.gz',
 'expert':'audit-attempt-02/outputs/job/audit-physical/single__m10_expert.json.gz',
 'gain1_mean80':'train-attempt-02/outputs/job/dev-0080/single__mean.json.gz',
 'gain16_mean80':'train-attempt-03/outputs/job/dev-0080/single__mean.json.gz',
 'gain16_sample80':'train-attempt-03/outputs/job/dev-0080/single__sample.json.gz',
 **{f'command_scale{k}':f'amplitude-attempt-02/outputs/job/amplitude/mean_scale{k}.json.gz' for k in [4,16,-4]},
}
out={}
def rms(x):return np.sqrt(np.mean(np.square(x),axis=0)).tolist()
for name,path in paths.items():
 x=json.load(gzip.open(R/path,'rt'));o={};out[name]=o
 phases=dict(x['phases']);phases['contact']=[x['condition']['onset'],x['condition']['release']]
 for phase,(lo,hi) in phases.items():
  ps=[p for p in x['physics'] if lo<=p['force_time']<hi];cs=[c for c in x['control'] if lo<=c['time']<hi]
  f=np.array([p['force'] for p in ps]);err=np.array([np.array(p['anchor'])-p['wrist'] for p in ps]);q=np.array([c['q_PD'] for c in cs]);nom=np.array([c['nominal_q_PD'] for c in cs]);z=np.array([c['executed_z'][0] for c in cs]);b=np.array([c['prior_z'][0] for c in cs]);ang=np.degrees(np.arccos(np.clip(np.sum(z*b,axis=-1)/256,-1,1)))
  d=np.array([c['residual_uncapped_norm'] for c in cs]);dem=np.array([c['uncapped_demand_xyz'] for c in cs]);dn=np.linalg.norm(dem,axis=-1);vel=np.array([np.array(c['post_partner_velocity'])-c['post_velocity'] for c in cs]);terms={k:sum(c['reward_terms'][k]*.02 for c in cs) for k in cs[0]['reward_terms']}
  o[phase]=dict(error_xyz_cm_rms=(100*np.sqrt(np.mean(err**2,axis=0))).tolist(),error_xyz_mean_cm=(100*err.mean(0)).tolist(),force_xyz_rms=rms(f),force_rms=float(np.sqrt(np.mean(np.sum(f*f,axis=-1)))),force_cap_fraction=float(np.mean(np.linalg.norm(f,axis=-1)>=4.999)),demand_norm_median=float(np.median(dn)),demand_norm_p95=float(np.percentile(dn,95)),latent_angle_mean=float(ang.mean()),latent_angle_p95=float(np.percentile(ang,95)),residual_norm_mean=float(d.mean()),residual_cap_fraction=float(np.mean(d>=16)),qpd_vs_nominal_rms_deg=float(np.sqrt(np.mean(np.degrees(q-nom)**2))),qpd_joint_rms_deg=rms(np.degrees(q-nom)),relative_velocity_xyz_rms=rms(vel),reward_integrals=terms,root_delta=(np.array(ps[-1]['root'])-ps[0]['root']).tolist(),wrist_delta=(np.array(ps[-1]['wrist'])-ps[0]['wrist']).tolist(),max_tilt=max(p['tilt_deg'] for p in ps),torque_saturation_mean=float(np.mean([p['torque_saturation'] for p in ps])),max_qpd_step_deg=float(np.degrees(max(c['q_PD_step_max_rad'] for c in cs))))
print(json.dumps(out,indent=2))
