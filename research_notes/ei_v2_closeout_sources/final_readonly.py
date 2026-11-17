"""Frozen actual-archive closeout. No dynamics, policy learning or selection."""
import os,sys,time,json,hashlib,traceback,collections,csv,math
from pathlib import Path
START=time.monotonic()
S=Path(__file__).resolve().parents[1];R=S/'eval-runtime'
sys.path[:0]=[str(R),str(R/'research_notes')]
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
torch.set_num_threads(1);torch.set_grad_enabled(False)
from research.interaction_fb.journal import read
from research.interaction_fb.ei_v2_dev import load_core,plan
OUT=S/'runs/final-readonly-01';OUT.mkdir(exist_ok=False)
D=S/'runs/formal-dev-02';results=json.loads((D/'evaluation.json').read_text())
plans={r['index']:r for r in plan(S)}
def jw(name,x):
 (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cnt(x):return dict(collections.Counter(x))
def phase(sp,t,active):
 if not active:return 'precontact' if t<sp['onset'] else 'released'
 stop=sp.get('stop_time',sp['onset']+sp['ramp'])
 return 'moving' if t<stop-1e-8 else ('decelerating' if t<stop+sp.get('deceleration',0)-1e-8 else 'holding')
def kin(ph,cc):
 if not ph or len(cc)<2:return None
 qpd=np.array([c['q_PD']for c in cc]);q=np.array([p['qpos'][7:]for p in ph]);dq=np.array([p['qvel'][6:]for p in ph]);torque=np.array([p['torque']for p in ph]);jump=abs(np.diff(qpd,axis=0))
 root=np.array([p['root']for p in ph]);w=np.array([p['wrist']for p in ph]);rv=np.diff(root,axis=0)/.005;wv=np.diff(w,axis=0)/.005
 rr={'qPD_jump_p95_rad_per_joint':np.quantile(jump,.95,axis=0).tolist(),'qPD_jump_max_rad_per_joint':jump.max(0).tolist(),'qPD_jump_max_rad':float(jump.max()),'actual_dq_p95_rad_s_per_joint':np.quantile(abs(dq),.95,axis=0).tolist(),'actual_dq_max_rad_s_per_joint':abs(dq).max(0).tolist(),'actual_ddq_p95_rad_s2_per_joint':np.quantile(abs(np.diff(dq,axis=0)/.005),.95,axis=0).tolist(),'actual_q_min_rad_per_joint':q.min(0).tolist(),'actual_q_max_rad_per_joint':q.max(0).tolist(),'actual_q_5ms_jump_max_rad':float(abs(np.diff(q,axis=0)).max()),'torque_p95_Nm_per_joint':np.quantile(abs(torque),.95,axis=0).tolist(),'torque_peak_Nm_per_joint':abs(torque).max(0).tolist(),'torque_saturation_max':max(p['torque_saturation']for p in ph),'root_speed_p95_m_s':float(np.quantile(np.linalg.norm(rv,axis=1),.95)),'wrist_speed_p95_m_s':float(np.quantile(np.linalg.norm(wv,axis=1),.95)),'root_accel_p95_m_s2':float(np.quantile(np.linalg.norm(np.diff(rv,axis=0)/.005,axis=1),.95)),'wrist_accel_p95_m_s2':float(np.quantile(np.linalg.norm(np.diff(wv,axis=0)/.005,axis=1),.95))}
 # q_PD is a target. q/dq are independent measured physics values.
 times=np.array([c['time']for c in cc]);pdtimes=np.array([p['force_time']for p in ph]);ix=np.clip(np.searchsorted(times,pdtimes,side='right')-1,0,len(cc)-1)
 rr['PD_target_actual_error_p95_rad_per_joint']=np.quantile(abs(qpd[ix]-q),.95,axis=0).tolist()
 return rr
def gate(r,a):
 sp=a['spec'];ph=a['online_physics'];ins=a['instructions'];ix=0;groups=collections.defaultdict(list)
 for p in ph:
  while ix+1<len(ins)and ins[ix+1]['time']<=p['force_time']+1e-8:ix+=1
  ii=ins[ix];mode='FOLLOW'if ii['goal_mode']==0 else'HOLD'
  if p['active']:groups[mode].append((p,ii))
 failures=[];unmeasured=[]
 if r['hard_boundary']or r['online_quality']['max_joint_excess_rad']>.02:failures.append('hard_boundary')
 if r['online_quality']['unsupported_ticks']:failures.append('support_lost')
 if r['family']in('none','zero'):
  failures.extend(r.get('protection_failed',[]))
  if r['family']=='none'and not r['no_contact_force_zero']:failures.append('none_force_nonzero')
 else:
  if r['loaded_slip_peak']>.45:failures.append('active_slip_peak')
  if r['yaw_abs_final']>20:failures.append('final_abs_yaw')
  for mode,gg in groups.items():
   if mode=='HOLD':
    errs=np.array([np.linalg.norm(np.array(p['wrist'])-ii['goal_position_world'])for p,ii in gg]);fr=math.sqrt(np.mean([np.dot(p['force'],p['force'])for p,ii in gg]))
    if math.sqrt(np.mean(errs**2))>.04:failures.append('HOLD_public_target_RMS')
    if np.quantile(errs,.95)>.08:failures.append('HOLD_public_target_p95')
    if fr>7:failures.append('HOLD_force_RMS')
   else:
    # Only current FOLLOW intervals enter FOLLOW outcome and useful-progress checks.
    pp=[p for p,ii in gg];er=math.sqrt(np.mean([np.dot(np.array(p['anchor'])-p['wrist'],np.array(p['anchor'])-p['wrist'])for p in pp]));fr=math.sqrt(np.mean([np.dot(p['force'],p['force'])for p in pp]));bound=.04 if r['family']=='near'else .08;fb=4 if r['family']=='near'else 7
    if er>bound:failures.append('FOLLOW_error_RMS')
    if fr>fb:failures.append('FOLLOW_force_RMS')
    # Original progress formula within each contiguous FOLLOW period. A HOLD span is not asked to advance.
    spans=[];span=[];last=None
    for p in pp:
     if last is not None and p['force_time']-last>.0051:spans.append(span);span=[]
     span.append(p);last=p['force_time']
    if span:spans.append(span)
    for ss in spans:
     desired=ss[-1]['anchor'][0]-ss[0]['anchor'][0];actual=ss[-1]['wrist'][0]-ss[0]['wrist'][0]
     if actual<desired-bound:failures.append('FOLLOW_useful_progress')
    if r['family']=='stop':
     ss=[p for p in pp if phase(sp,p['force_time'],True)=='holding']
     if ss:
      e=math.sqrt(np.mean([np.dot(np.array(p['anchor'])-p['wrist'],np.array(p['anchor'])-p['wrist'])for p in ss]));f=math.sqrt(np.mean([np.dot(p['force'],p['force'])for p in ss]))
      if e>.08:failures.append('stop_holding_error')
      if f>6.5:failures.append('stop_holding_force')
  rec=r['recovery']
  if rec['status'].startswith('UNMEASURED'):unmeasured.append(rec['status'])
  failures.extend(rec.get('failed',[]))
 return dict(task_quality_failures=sorted(set(failures)),unmeasured=unmeasured,pass_task_quality_no_continuity=not failures and not unmeasured,progress_note='original X progress formula restricted to contiguous current FOLLOW spans; HOLD is not penalized for not following')
report={};cases=[];stages=[];probe_counts={};continuity={};probe_records=[]
try:
 for r in results:
  i=r['case_index'];a=read(D/f'case-{i:03d}-actual.ptz',r['archive']);ph=a['online_physics'];sp=a['spec'];rec=plans[i];assert sp['parent_id']==r['parent_id'] and len(a['rows'])*4==len(ph)
  gg=gate(r,a);kk=kin(ph,a['controls']);continuity[str(i)]=kk
  cr=dict(case=i,method=r['method'],role=r['role'],parent=r['parent_id'],ancestor=sp['execution_source_parent_id'],family=r['family'],source=sp['source'],K=sp['h0_stiffness'],distance_cm=sp['distance']*100,goal=r['goal_stream'],J=r['goal_aware']['J'],error_cm=r['all']['error_cm'],force_N=r['all']['force_rms'],cap_fraction=r['all']['cap_fraction'],partner_progress_cm=100*r['partner_progress_m'],wrist_progress_cm=100*r['wrist_progress_m'],loaded_slip_peak=r['loaded_slip_peak'],loaded_slip_p95=r['online_quality']['slip_p95'],yaw_abs_final=r['yaw_abs_final'],yaw_excursion=r['online_quality']['yaw_excursion_deg'],yaw_net=r['online_quality']['yaw_net_deg'],root_path_m=r['online_quality']['root_path_m'],recovery_status=r['recovery']['status'],recovery_failures=r['recovery'].get('failed',[]),**gg)
  cases.append(cr)
  for win,v in r['goal_aware']['windows'].items():
   st,mode=win.split(':');stages.append(dict(case=i,method=r['method'],role=r['role'],parent=r['parent_id'],family=r['family'],stage=st,goal=mode,**{k:v[k]for k in ('duration_s','error_xyz_rms_m','signed_error_xyz_m','force_xyz_rms_N','force_3D_rms_N','error_3D_rms_m','saturation_fraction','J')},public_goal_error_rms_m=v.get('public_goal_error_rms_m'),public_goal_error_p95_m=v.get('public_goal_error_p95_m'),quality=v['quality']))
  # Pre-registered label support first. All eligible 20ms starts, frozen I primary actual trajectories.
  if r['method']=='I-final'and r['role']=='primary':
   for h,n in ((.5,100),(1.,200)):
    for row in a['rows']:
     b=row['physics_begin']
     if b+n>len(ph)or np.linalg.norm(ph[b]['force'])<1:continue
     ss=ph[b:b+n];yaw=np.unwrap([math.atan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0])for p in ss]);fr=math.sqrt(np.mean([np.dot(p['force'],p['force'])for p in ss]));good=fr<=4 and max(p['stage_loaded_slip_mps']for p in ss)<=.45 and np.degrees(abs(yaw-yaw[0])).max()<=20 and max(p['joint_limit_excess_max_rad']for p in ss)<=.02 and all(any(p['support'])for p in ss)
     probe_records.append(dict(case=i,parent=r['parent_id'],ancestor=sp['execution_source_parent_id'],horizon=h,row_index=row['physics_begin']//4,label=int(good),start_force_N=float(np.linalg.norm(ph[b]['force']))))
  jw('progress.json',dict(cases=i+1,planned=100));print('ARCHIVE',i,flush=True)
 # Comparative continuity uses matching fixed Goal-only case; BC500 is a historical FOLLOW reference, not a Goal-conditioned arm.
 by={(x['method'],x['parent'],x['role'],x['goal'],x['family']):x for x in cases}
 for c in cases:
  c['continuity_degradation']=[]
  baseline=by.get(('Goal-only',c['parent'],c['role'],c['goal'],c['family']))
  if baseline is not None and c['method']in('E-final','I-final'):
   for field,ab in (('qPD_jump_p95_rad_per_joint',.03),('actual_dq_p95_rad_s_per_joint',.5)):
    v=np.array(continuity[str(c['case'])][field]);b=np.array(continuity[str(baseline['case'])][field]);bad=np.where((v>1.25*b)&(v-b>ab))[0].tolist()
    if bad:c['continuity_degradation'].append(dict(field=field,joints=bad))
  c['joint_task_quality_pass']=c['pass_task_quality_no_continuity']and not c['continuity_degradation']
 for h in (.5,1.):
  rr=[x for x in probe_records if x['horizon']==h];pos={x['ancestor']for x in rr if x['label']};neg={x['ancestor']for x in rr if not x['label']};allg={x['ancestor']for x in rr}
  probe_counts[str(h)]=dict(rows=len(rr),positive_rows=sum(x['label']for x in rr),negative_rows=sum(not x['label']for x in rr),positive_ancestors=len(pos),negative_ancestors=len(neg),ancestors=len(allg),status='ELIGIBLE_FOR_GROUPED_PROBE'if len(pos)>=4 and len(neg)>=4 and len(allg)>=8 else'UNDERPOWERED',positive_parent_ids=sorted({x['parent']for x in rr if x['label']}),negative_parent_ids=sorted({x['parent']for x in rr if not x['label']}))
 jw('probe-label-records.json',probe_records);jw('probe-label-support.json',probe_counts)
 jw('case-results.json',cases);jw('stage-results.json',stages);jw('continuity.json',continuity)
 family={};rng=np.random.default_rng(660304)
 for method in dict.fromkeys(c['method']for c in cases):
  rr=[c for c in cases if c['method']==method and c['role']=='primary'];pp=collections.defaultdict(list)
  for c in rr:pp[c['parent']].append(c)
  family[method]={'parents':len(pp),'episodes':len(rr),'joint_pass_parents':sum(all(c['joint_task_quality_pass']for c in vv)for vv in pp.values()),'failures':cnt(f for c in rr for f in c['task_quality_failures']),'continuity_degraded_cases':sum(bool(c['continuity_degradation'])for c in rr),'families':{}}
  for f in ('near','mid','stop-continue','none','zero'):
   vv={p:cs for p,cs in pp.items()if ('stop-continue'if cs[0]['family']in('stop','continue','continue_pair')else cs[0]['family'])==f}
   js=[np.mean([c['J']for c in cs if c['J']is not None])for cs in vv.values()if any(c['J']is not None for c in cs)]
   family[method]['families'][f]=dict(parents=len(vv),joint_pass=sum(all(c['joint_task_quality_pass']for c in cs)for cs in vv.values()),J=float(np.mean(js))if js else None)
 # Paired parent bootstrap, not frames or sibling branches.
 bootstrap={}
 for method in ('E-final','I-final'):
  bootstrap[method]={}
  for fam in ('near','mid','stop-continue'):
   cur=family[method]['families'][fam];values=[]
   for c in cases:
    if c['method']!=method or c['role']!='primary':continue
    f='stop-continue'if c['family']in('stop','continue','continue_pair')else c['family']
    if f!=fam:continue
    b=by[('Goal-only',c['parent'],c['role'],c['goal'],c['family'])];values.append((c['parent'],c['J'],b['J']))
   pts=collections.defaultdict(list)
   for p,v,b in values:pts[p].append((v,b))
   arr=np.array([np.mean(v,axis=0)for v in pts.values()]);draw=rng.integers(len(arr),size=(1000,len(arr)));samples=arr[draw].mean(1);imp=1-samples[:,0]/samples[:,1]
   bootstrap[method][fam]=dict(independent_parents=len(arr),mean_relative_improvement=1-float(arr[:,0].mean()/arr[:,1].mean()),bootstrap95_relative_improvement=np.quantile(imp,[.025,.975]).tolist(),seed=660304,resamples=1000)
 jw('joint-success.json',family);jw('parent-bootstrap.json',bootstrap)
 # Actual training checkpoints. Read original checked core and all referenced immutable segments, exactly once per manifest path.
 training={}
 for arm in ('E','I'):
  cp=S/f'runs/formal-{arm}-06/checkpoints/block-node-32000';m=json.loads((cp/'manifest.json').read_text());state,identity=load_core(cp);own=state['own'];episodes=own['episodes'];current_spec=state['physical']['task'].get('spec');del state
  coverage=collections.Counter();options=collections.Counter();catdraw=collections.Counter();seen=set();rowcount=0;phase_rows=collections.Counter();goaltrans=collections.Counter();complete=collections.Counter();native=0;physical=0
  for ep in episodes:
   sp=ep['spec'];complete[(sp['source'],sp['demand_family'])]+=1
  for seg in m['segments']:
   assert seg['path']not in seen;seen.add(seg['path']);part=read(cp.parent/seg['path'],seg);native+=len(part['native']);physical+=len(part['physical_stream']);ph=part['physical_stream'];off=rowcount*4
   for row in part['rows']:
    assert row['eligible_for_joint_PPO'];rowcount+=1;phase_rows[row['behavior_phase']]+=1;parent=bytes(row['provenance']['parent_sha256'][0].tolist()).hex();t=float(row['provenance']['time'][0,0])
    # Resolve exact episode using stored online episode identity, never infer the sibling from time.
    epi=int(row['episode']);sp=episodes[epi]['spec'] if epi<len(episodes)else current_spec
    assert sp is not None and sp['parent_id']==parent,('episode parent mismatch',epi,parent)
    pidx=row['physics_begin']-off
    if not 0<=pidx<len(ph):
     # A checkpoint slice can start mid-episode; absolute global physics index still follows four ticks per online row.
     raise AssertionError(('physical index',row['physics_begin'],off,len(ph)))
    actual=ph[pidx];st=phase(sp,t,actual['active']);mode='FOLLOW'if row['goal_mode']==0 else'HOLD';coverage[(sp['source'],sp['demand_family'],row['behavior_phase'],st,mode)]+=1;options[(row['behavior_phase'],str(row['option']))]+=1
    if row['cat_boundary']:catdraw[(row['behavior_phase'],str(row['option']))]+=1
  assert rowcount==32000 and physical==128000
  study=json.loads((S/f'runs/formal-{arm}-06/result.json').read_text())['study'];steps=[ss for rr in study['records']for ss in rr['metrics']['steps']];accepted=[ss for ss in steps if ss['accepted']];trials=[t for ss in steps for t in ss.get('trials',[])];kk=[t['KL']for ss in accepted for t in ss['trials']if t.get('acceptable')and t.get('alpha')==ss.get('selected_alpha')]
  tr=dict(identity=identity,actual_rows=rowcount,physics_ticks_in_onpolicy_rows=physical,completed_episodes=len(episodes),independent_parents=len({ep['spec']['parent_id']for ep in episodes}),native_reference_rows_not_PPO=native,phase_rows=dict(phase_rows),option_rows=[dict(phase=k[0],option=int(k[1]),rows=v)for k,v in sorted(options.items())],selector_draws=[dict(phase=k[0],option=int(k[1]),draws=v)for k,v in sorted(catdraw.items())],coverage=[dict(source=k[0],family=k[1],learning_phase=k[2],physical_phase=k[3],goal=k[4],rows=v,physical_duration_s=.02*v)for k,v in sorted(coverage.items())],completed_episode_coverage=[dict(source=k[0],family=k[1],episodes=v)for k,v in sorted(complete.items())],accepted_actor_updates=len(accepted),attempts=len(steps),selected_alpha=cnt(str(ss.get('selected_alpha'))for ss in accepted),max_old_logprob_error=max(rr['metrics']['old_logprob_max_error']for rr in study['records']),accepted_joint_KL_p50_p95_max=np.quantile([k['joint']for k in kk],[.5,.95,1]).tolist(),accepted_categorical_KL_p50_p95_max=np.quantile([k['categorical']for k in kk],[.5,.95,1]).tolist(),blocks=len(study['records']))
  training[arm]=tr;jw('training-coverage.json',training);print('TRAINING',arm,rowcount,flush=True)
 report=dict(status='FINAL_READONLY_COMPLETE',cases=100,case_source_sha256=hashlib.sha256((D/'evaluation.json').read_bytes()).hexdigest(),training=training,probe_label_support=probe_counts,joint_success=family,bootstrap=bootstrap,physics_ticks=0,policy_updates=0,representation_updates=0,sealed_read=False)
except BaseException as ex:
 report=dict(status='FINAL_READONLY_FAILED',error=repr(ex),traceback=traceback.format_exc(),completed_cases=len(cases),physics_ticks=0,policy_updates=0,sealed_read=False)
finally:
 report['wall_s']=time.monotonic()-START;report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();jw('result.json',report)
 lp=S/'ledger.json';ledger=json.loads(lp.read_text());ledger['charged_s']+=report['wall_s'];ledger['jobs'].append(dict(name=OUT.name,status=report['status'],wall_s=report['wall_s'],physics=dict(actual=0,reserved=0),source_sha256=report['source_sha256']));tmp=lp.with_suffix('.pending');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,lp)
 print(json.dumps({k:v for k,v in report.items()if k not in('training','joint_success','bootstrap')}),flush=True)
if report['status']!='FINAL_READONLY_COMPLETE':raise RuntimeError(report['error'])
