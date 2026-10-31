"""Read-only saved-trace analysis; never import model, train or simulate."""
import json,gzip,copy
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from m2_analysis import window
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m6-multibehavior-20261031A')
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 ledger=read(RUN/'compute-ledger.json');res={};comparisons={};parity={}
 for a in ledger['attempts']:
  src=Path(a['path'])/'outputs';raw={}
  sources=[(p,None) for p in sorted(src.glob('*.json.gz'))]
  if a['stage']=='final':sources += [(Path(r['source']),r) for r in read(Path(__file__).with_name('m6_protocol.json'))['reused_exact_physical_controls']]
  for p,reuse in sources:
   e=read(p)
   if reuse:
    expected=read(Path(__file__).with_name('m6_protocol.json'))['conditions'][reuse['condition']]
    assert all(e['condition'][k]==expected[k] for k in expected)
    e['controller']=reuse['controller'];e['label']=reuse['condition']+'__'+reuse['controller']
   raw[e['label']]=e
   for c in e['control']:
    c['oracle_alpha']=c['alpha'];c['alpha']=c['native_coords'][0] if c.get('native_coords') is not None else 0. if e['controller']=='native_hold' else c['alpha']
   x=summarize(e,{});rel=e['condition']['release'];end=e['condition']['end'];on=e['condition']['onset'];rows=e['physics'];cs=e['control'];times=np.array([r['time'] for r in rows]);ct=np.array([c['time'] for c in cs]);q=np.array([c['q_PD'] for c in cs]);dq=abs(np.diff(q,axis=0));i,j=np.unravel_index(dq.argmax(),dq.shape)
   x['source']=str(p);x['recovery']=window(e,rel,end);x['late_recovery']=window(e,rel+6.4,end);x['precontact']=window(e,3,on);x['recovery_speed_2_4_6_8']=[float(np.linalg.norm(rows[int(np.argmin(abs(times-rel-t)))]['qvel'][:3])) for t in [2,4,6,8]]
   x['qPD_peak']=dict(time=cs[i+1]['time'],joint_index=int(j),target_step_deg=float(np.degrees(dq[i,j])),actual_next_20ms_deg=float(np.degrees(cs[min(i+2,len(cs)-1)]['pre_qpos'][7+j]-cs[i+1]['pre_qpos'][7+j])))
   z=np.array([c['executed_z'][0] for c in cs]);x['z_norm_error']=float(abs(np.linalg.norm(z,axis=1)-16).max());x['expert_query_valid_fraction']=float(np.mean([c['oracle_query_valid'] for c in cs])) if 'oracle_query_valid' in cs[0] else None
   rc=[c for c in cs if c['time']>=rel-1e-8];aa=np.array([c['alpha'] for c in rc]);late=[c['alpha'] for c in rc if c['time']>=rel+2-1e-8]
   x['release_progress']=dict(max_rebound_after_running_min=float((aa-np.minimum.accumulate(aa)).max()),last_alpha_over_0_01_delay=next((c['time']-rel for c in reversed(rc) if c['alpha']>.01),None),max_alpha_after_2s=max(late,default=0.))
   x['events']=[dict(time=c['time'],**c['rule_record']) for c in cs if c.get('rule_record') and c['rule_record'].get('event')];res[a['stage']+'/'+e['label']]=x
  for label,e in raw.items():
   key=a['stage']+'/'+label;x=res[key];name=e['condition']['name'];base=raw.get(name+'__native_hold');teacher=raw.get(name+'__teacher')
   if base and e['condition']['enabled']:
    b=res[a['stage']+'/'+base['label']]['phases']['all_contact'];v=x['phases']['all_contact'];comparisons[key]=dict(error_gain=1-v['tracking_rms_cm']/b['tracking_rms_cm'],force_gain=1-v['force_rms_n']/b['force_rms_n'])
    if teacher:
     te=res[a['stage']+'/'+teacher['label']]['phases']['all_contact']['tracking_rms_cm'];den=b['tracking_rms_cm']-te;comparisons[key]['teacher_gain_recovered']=(b['tracking_rms_cm']-v['tracking_rms_cm'])/den if abs(den)>1e-8 else None
   free=raw.get('free__'+e['controller'])
   if free and e['condition']['onset']==5:
    n=sum(r['force_time']<e['condition']['onset']-1e-8 for r in e['physics']);parity[key]={k:float(abs(np.array([r[k] for r in e['physics'][:n]])-np.array([r[k] for r in free['physics'][:n]])).max()) for k in ['qpos','qvel','force']}
    t=e['condition']['release'];i=min(range(len(e['physics'])),key=lambda i:abs(e['physics'][i]['time']-t));x['contact_minus_free_end_cm']={k:((np.array(e['physics'][i][k])-free['physics'][i][k])*100).tolist() for k in ['wrist','shoulder_xyz','root']}
  if 'free__native_hold' in raw:
   base=raw['free__native_hold'];bw=np.array([r['wrist'] for r in base['physics']]);bq=np.array([r['qpos'] for r in base['physics']])
   for ctl in ['rule','teacher','alignment','distill','single_negative','m5','force_rule']:
    if 'free__'+ctl not in raw:continue
    e=raw['free__'+ctl];v=np.array([r['wrist'] for r in e['physics']])-bw;dq=np.array([r['qpos'] for r in e['physics']])-bq
    res[a['stage']+'/free__'+ctl]['relative_to_native_free']=dict(wrist_rms_cm=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))*100),wrist_max_cm=float(np.linalg.norm(v,axis=1).max()*100),qpos_max_difference=float(abs(dq).max()))
  for label,e in raw.items():
   key=a['stage']+'/'+label;rule=raw.get(e['condition']['name']+'__rule')
   if key in comparisons and rule:
    comparisons[key]['error_gain_vs_rule']=1-res[key]['phases']['all_contact']['tracking_rms_cm']/res[a['stage']+'/'+rule['label']]['phases']['all_contact']['tracking_rms_cm']
 result=dict(episodes=res,comparisons=comparisons,precontact_pair_parity=parity,ledger=ledger,scope='Only true contact contributes tracking; full8s release; allfailedattempts preserved; one seed with two fixed unseen condition combinations; no unseen body or universal compliance claim')
 d=RUN/'analysis';d.mkdir(exist_ok=True);(d/'physical-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for k,x in res.items():
  c=x['phases'].get('all_contact',{});print(k,'error',c.get('tracking_rms_cm'),'F',c.get('force_rms_n'),'cap',c.get('force_cap_fraction'),'qPD',x['qPD_peak']['target_step_deg'],'recovery',x['recovery_speed_2_4_6_8'],flush=True)
if __name__=='__main__':main()
