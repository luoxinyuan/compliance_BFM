"""Read-only completed M3 physical qualification; no model fitting or new rollout."""
import gzip,json
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from m2_analysis import window
from ccl02a_teacher import audit_episode
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m3-continuous-body-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 ledger=read(R/'compute-ledger.json');initial=read(P/'bridge-v2-owned/runs/issue21-ccl02a-20261029T153900Z/attempt-01/outputs/dev-2-initial-state.json');out={};parity={}
 for a in ledger['attempts']:
  if a['stage'] not in ['preparation','qualification','ablation'] or a.get('status')!='finished':continue
  src=Path(a['path'])/'outputs'
  if not (src/'action-contract.json').exists():continue
  contract=read(src/'action-contract.json')
  for p in sorted(src.glob('*.json.gz')):
   e=read(p);q=summarize(e,{});rel=e['condition']['release'];ready=e['control_ready_s'];q.update(source=str(p),configuration_u=e['goal_candidate']['u'],ready_sha=e['ready_physical_state_sha256'],native_last_second=window(e,ready-1,ready),precontact=window(e,ready,e['condition']['onset']),full_recovery=window(e,rel,e['condition']['end']),deployment_audit=audit_episode(e,contract,initial))
   rows=e['physics'];ct=e['control'];q['recovery_curve']=[]
   for dt in np.arange(0,e['condition']['end']-rel+.001,.2):
    r=min(rows,key=lambda r:abs(r['time']-rel-dt));c=min(ct,key=lambda c:abs(c['time']-rel-dt));q['recovery_curve'].append(dict(after_release_s=round(float(dt),2),speed_mps=float(np.linalg.norm(r['qvel'][:3])),wrist=r['wrist'],root=r['root'],feet=r['feet_xyz'],foot_load=r['foot_normal_force_n'],force_xyz=r['force'],tilt_deg=r['tilt_deg']))
   q['recovery_speed_2_4_6_8']=[next(r['speed_mps'] for r in q['recovery_curve'] if r['after_release_s']==t) for t in [2,4,6,8] if t<=e['condition']['end']-rel+1e-8]
   q['qPD_deployed_peak_deg']=float(np.rad2deg(max(c['q_PD_step_max_rad'] for c in ct if c['time']>=ready-1e-8)));q['latent_norm_max_error']=max(abs(np.linalg.norm(c['executed_z'])-16) for c in ct);assert q['latent_norm_max_error']<1e-4
   key=a['stage']+'/'+e['label'];out[key]=q
   if e['ready_physical_state_sha256'] is not None:
    parity.setdefault(e['configuration'],set()).add(e['ready_physical_state_sha256'])
 assert all(len(v)==1 for v in parity.values())
 comparisons=[]
 for u in [.25,.5,.75]:
  config=f'u{round(u*1000):03d}';b=out.get('qualification/'+config+'__native_hold')
  if not b:continue
  for ctl in ['teacher','local_teacher']:
   q=out.get('qualification/'+config+'__'+ctl)
   if not q:continue
   pairs={phase:{metric:dict(native=b['phases'][phase][metric],teacher=q['phases'][phase][metric],relative_gain=1-q['phases'][phase][metric]/b['phases'][phase][metric]) for metric in ['tracking_rms_cm','force_rms_n']} for phase in ['all_contact','hold']}
   comparisons.append(dict(u=u,controller=ctl,physical_boundary=q['boundary'],paired=pairs,practical_evidence_note='Do not train on a teacher with worse full3D hold; improvement alone is not statistical or optimal-body proof. Read continuity/force/body effects beside errors.'))
 result=dict(episodes=out,comparisons=comparisons,ready_parity={k:list(v) for k,v in parity.items()},no_training=True)
 (R/'analysis/qualification-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for k,e in out.items():
  x=e['phases'].get('all_contact',{});print(k,'boundary',e['boundary'],'error',x.get('tracking_rms_cm'),'hold',e['phases'].get('hold',{}).get('tracking_rms_cm'),'force',x.get('force_rms_n'),'recovery',e['recovery_speed_2_4_6_8'],'ready',e['native_last_second']['root_speed_max_mps'])
 print('COMPARISONS',json.dumps(comparisons))
if __name__=='__main__':main()
