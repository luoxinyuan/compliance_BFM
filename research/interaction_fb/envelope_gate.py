"""Read-only frozen qualification gates, original absolute/relative definitions."""
import json,math,sys,hashlib
from pathlib import Path
import numpy as np,torch
S=Path(sys.argv[1]);run=S/'runs'/sys.argv[2];O=S.parent/'interaction-pilot-adapter-off-20261112'
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def metrics(rows):
 slip=np.array([float(np.max(r['stage_loaded_slip_mps'])) for r in rows]);mat=np.asarray(rows[-1]['pelvis_matrix']).reshape(3,3)
 active=[r for r in rows if r['active']]
 result=dict(ticks=len(rows),start=rows[0]['time'],end=rows[-1]['time'],duration=len(rows)*.005,actual_joint_excess_max_rad=max(r['joint_limit_excess_max_rad'] for r in rows),slip_peak=float(slip.max()),slip_p95=float(np.percentile(slip,95)),min_height=min(r['height'] for r in rows),yaw_final_deg=math.degrees(math.atan2(mat[1,0],mat[0,0])),no_support_ticks=sum(not any(r['support']) for r in rows),force_max=max(float(np.linalg.norm(r['force'])) for r in rows),root_displacement=float(np.linalg.norm(np.asarray(rows[-1]['root'])[:2]-np.asarray(rows[0]['root'])[:2])),final_root_speed=float(np.linalg.norm(rows[-1]['qvel'][:3])),active_ticks=len(active),absolute_slip_pass=bool(slip.max()<=.45))
 result['absolute_yaw_pass']=abs(result['yaw_final_deg'])<=20
 if active:
  result.update(error_rms=float(np.sqrt(np.mean([np.dot(r['error'],r['error']) for r in active]))),force_rms=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in active]))),wrist_progress=float(active[-1]['wrist'][0]-active[0]['wrist'][0]),partner_progress=float(active[-1]['anchor'][0]-active[0]['anchor'][0]))
 return result

out={'scope':'exposed frozen qualification; no training or sealed selection','branches':{}}
for typ in ['sample','mean']:
 d=run/('envelope-'+typ)
 summary=d/'qualification.json'
 if not summary.exists() and typ=='sample':summary=S/(sys.argv[2]+'-partial-evidence.json')
 if not summary.exists():out['branches'][typ]={'not_completed':True};continue
 j=json.loads(summary.read_text());t=torch.load(d/'qualification-trace.pt',map_location='cpu',weights_only=False)
 if j['case']['id']=='B19-release':
  od=O/'runs'/('off'+typ+'-B19');oj=json.loads((od/'probe.json').read_text());ot=torch.load(od/'probe-trace.pt',map_location='cpu',weights_only=False)
  controls=[r for r in ot['controls'] if r['time']>=j['case']['time']-1e-8]
  checks=dict(model=j['model_digest']==oj['frozen_agent_digest'],rng=j['rng_digest']==oj['restored_rng_digest'],task=same(t['task_spec'],ot['task_spec']),first_z=same(t['controls'][0]['executed_z'],controls[0]['executed_z']),first_obs=same(t['controls'][0]['bfm_obs'],controls[0]['bfm_obs']))
 else:
  od=run/('off-'+typ);oj=json.loads((od/'qualification.json').read_text());ot=torch.load(od/'qualification-trace.pt',map_location='cpu',weights_only=False)
  checks={k:j[k]==oj[k] for k in ['initial_digest','prefix_digest','rng_digest','model_digest']}
 assert all(checks.values()),checks
 m=metrics(t['physics']);om=metrics(ot['physics']);complete=j['status']=='completed_3s' and m['ticks']==600 and om['ticks']==600
 gates=dict(complete_same_3s=complete,no_hard_boundary=j['status']=='completed_3s')
 if j['case']['id']=='B19-release':gates['actual_joint_excess_zero']=m['actual_joint_excess_max_rad']==0
 if complete:
  gates.update(slip_peak=m['slip_peak']<=om['slip_peak']+max(.10,.20*om['slip_peak']),slip_p95=m['slip_p95']<=om['slip_p95']+max(.020,.20*om['slip_p95']),yaw=abs(m['yaw_final_deg'])<=abs(om['yaw_final_deg'])+5,height=m['min_height']>=om['min_height']-.03,no_support=m['no_support_ticks']<=om['no_support_ticks'])
  if m['active_ticks']:
   assert m['active_ticks']==om['active_ticks']
   gates.update(error=m['error_rms']<=om['error_rms']+max(.005,.1*om['error_rms']),force=m['force_rms']<=om['force_rms']+max(.25,.1*om['force_rms']))
   if m['partner_progress']>0:gates['progress']=m['wrist_progress']>=.9*om['wrist_progress']
 acts=t['actions'];au=[a['audit'] for a in acts];delta=np.array([(a['qexec']-a['qbase'])[0].numpy() for a in au]);rounderr=max(float(a['rounding_error'].max()) for a in au)
 active_controls=[i for i,a in enumerate(acts) if t['task_spec']['enabled'] and t['task_spec']['onset']<=a['time']<t['task_spec']['release']]
 correction=dict(rounding_error_max_rad=rounderr,median_rms=float(np.median(np.sqrt(np.mean(delta**2,axis=1)))),max_abs=float(np.abs(delta).max()),p90_abs=float(np.percentile(np.abs(delta),90)),trigger_rate=float(np.concatenate([a['triggered'].numpy() for a in au]).mean()),active_controls=len(active_controls))
 if active_controls:
  dd=delta[active_controls];qraw=np.array([np.array((a['raw']*5).clip(-5,5))*np.array([r['scale'] for r in json.loads((S/'runtime/research_notes/adapter_envelope_lock_v1.json').read_text())['operator']['mechanical_table']])/5+np.array([r['offset'] for r in json.loads((S/'runtime/research_notes/adapter_envelope_lock_v1.json').read_text())['operator']['mechanical_table']]) for a in au])[:,0,:]
  dr=qraw-np.array([a['qbase'][0].numpy() for a in au]);dr=dr[active_controls];nonzero=np.max(np.abs(dr),axis=1)>=.020
  ratio=np.linalg.norm(dd[nonzero],axis=1)/np.linalg.norm(dr[nonzero],axis=1)
  correction.update(active_median_rms=float(np.median(np.sqrt(np.mean(dd**2,axis=1)))),active_frame_delta_fraction=float(np.mean(np.max(np.abs(dd),axis=1)>=.020)),retention_median=float(np.median(ratio)) if len(ratio) else None)
  gates.update(adaptation_rms=correction['active_median_rms']>=.010,adaptation_frames=correction['active_frame_delta_fraction']>=.20,adaptation_retention=correction['retention_median'] is not None and correction['retention_median']>=.20)
 out['branches'][typ]=dict(gates=gates,pass_all=all(gates.values()),identity=checks,envelope=m,off=om,correction=correction,path_audit=j.get('path_audit'),trace_sha=j['trace_sha256'],model_sha=j['model_digest'])
out['pass_all']=all(r.get('pass_all',False) for r in out['branches'].values())
(S/(sys.argv[2]+'-gate.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
