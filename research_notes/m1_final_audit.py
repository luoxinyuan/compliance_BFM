"""Final causal comparison: saved physical records only. No models, fits or simulations."""
import gzip,json
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m1-state-response-20261031A')
def read(p):return json.loads(gzip.decompress(p.read_bytes()).decode()) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 attempts=[a for a in read(P/'compute-ledger.json')['attempts'] if a['stage']=='causal_final'];assert attempts and attempts[-1]['status']=='finished','Wait for canonical final termination'
 a=attempts[-1];out=Path(a['path'])/'outputs';episodes={}
 for p in out.glob('*.json.gz'):
  e=read(p)
  if 'physics' in e:episodes[e['label']]=e
 rows=[];negatives=[];parity=[]
 for label,e in episodes.items():
  if e['boundary']:continue
  r=e['physics'];cs=e['control'];times=np.array([x['force_time'] for x in r]);contact=np.array([x['active'] for x in r]);q=np.array([x['qpos'] for x in r]);w=np.array([x['wrist'] for x in r]);ready=e['control_ready_s'];on=e['condition']['onset'];pre=(times>=ready-1e-8)&(times<on-1e-8)
  deployed=[c for c in cs if c['time']>=ready-1e-8]
  row=dict(deployed_qPD_step_max_deg=float(np.rad2deg(max(c['q_PD_step_max_rad'] for c in deployed))),deployed_z_step_max_deg=max(c['z_step_deg'] for c in deployed),label=label,ready_physical_state_sha256=e['ready_physical_state_sha256'],onset_physical_state_sha256=e['physical_state_sha256'],control_ready_s=ready,onset=on,task_onset_handoff=e['task_onset_handoff'],actual_first_active_time=next((x['force_time'] for x in r if x['active']),None),precontact_wrist_delta_cm=((w[pre][-1]-e['ready_initial']['wrist'])*100).tolist(),precontact_root_delta_cm=((q[pre][-1,:3]-e['ready_initial']['root'])*100).tolist(),precontact_force_peak_n=float(np.linalg.norm(np.array([x['force'] for x in r])[pre],axis=-1).max()),anchor_xyz_cm=(np.array(e['initial']['wrist'])*100).tolist())
  rows.append(row);assert not e['task_onset_handoff'] and ready==6.2 and on==6.55 and row['precontact_force_peak_n']==0
  if e['condition']['enabled'] and e['controller'] in ['alignment','distill','rule']:
   other=episodes.get(label+'__free')
   if other and not other['boundary']:
    rr=[x for x in other['physics'] if x['force_time']<on-1e-8];aa=[x for x in r if x['force_time']<on-1e-8];cc=[x for x in cs if x['time']<on-1e-8];dd=[x for x in other['control'] if x['time']<on-1e-8]
    err={k:float(abs(np.array([x[k] for x in aa])-np.array([x[k] for x in rr])).max()) for k in ['qpos','qvel','wrist','force']};err['action']=float(abs(np.array([x['action'] for x in cc])-np.array([x['action'] for x in dd])).max());assert max(err.values())==0;parity.append(dict(controller=e['controller'],configuration=e['configuration'],exact_precontact_parity=err))
  if not e['condition']['enabled']:
   base=episodes.get('causal_heldout_'+e['configuration']+'__no_contact_hold__free')
   if base and not base['boundary']:
    br=base['physics'];bc=base['control'];mask=times>=ready-1e-8;dw=w-np.array([x['wrist'] for x in br]);dr=q[:,:3]-np.array([x['qpos'][:3] for x in br]);cm=np.array([x['time'] for x in cs])>=ready-1e-8;z=np.array([x['executed_z'][0] for x in cs]);zb=np.array([x['executed_z'][0] for x in bc]);angle=np.rad2deg(np.arccos(np.clip((z*zb).sum(-1)/(np.linalg.norm(z,axis=-1)*np.linalg.norm(zb,axis=-1)),-1,1)));dq=np.array([x['q_PD'] for x in cs])-np.array([x['q_PD'] for x in bc])
    negatives.append(dict(label=label,wrist_trajectory_difference_rms_cm=float(np.sqrt(np.mean(np.sum(dw[mask]**2,axis=-1)))*100),wrist_difference_peak_cm=float(np.linalg.norm(dw[mask],axis=-1).max()*100),root_trajectory_difference_rms_cm=float(np.sqrt(np.mean(np.sum(dr[mask]**2,axis=-1)))*100),latent_difference_peak_deg=float(angle[cm].max()),q_PD_difference_rms_deg=float(np.rad2deg(np.sqrt(np.mean(dq[cm]**2)))),force_peak_n=float(np.linalg.norm(np.array([x['force'] for x in r]),axis=-1).max())))
 for config in ['standing','raised']:
  hashes={r['ready_physical_state_sha256'] for r in rows if episodes[r['label']]['configuration']==config};assert len(hashes)<=1
 result=dict(stage_status=a.get('result_status'),complete_episodes=len(rows),episodes=rows,negative_relative_to_native_hold=negatives,positive_negative_precontact_parity=parity,meaning='Exactpairedready6.2; trajectoriesallowedtodivergebeforeonset6.55. Relativeanchorprotocol unchanged, anchorcoordinatesneednotmatchafterpolicyalreadyacted. Negativephysicaldifferences include nativefeedbacksensitivity, notbythemselves intentionlabels. Generic.75sinitializationblend persists independentlyoftaskonset.')
 (P/'analysis/causal-final-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(stage_status=result['stage_status'],negative=negatives,parity_cases=len(parity)),indent=2))
if __name__=='__main__':main()
