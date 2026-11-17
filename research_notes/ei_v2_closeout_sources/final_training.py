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
OUT=S/'runs/final-training-01';OUT.mkdir(exist_ok=False)
D=S/'runs/formal-dev-02'
def jw(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cnt(x):return dict(collections.Counter(x))
def phase(sp,t,active):
 if not active:return 'precontact' if t<sp['onset'] else 'released'
 stop=sp.get('stop_time',sp['onset']+sp['ramp'])
 return 'moving' if t<stop-1e-8 else ('decelerating' if t<stop+sp.get('deceleration',0)-1e-8 else 'holding')
report={}
try:
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
  study=json.loads((S/f'runs/formal-{arm}-06/result.json').read_text())['study'];all_steps=[ss for rr in study['records']for ss in rr['metrics']['steps']];steps=[ss for ss in all_steps if not ss.get('auxiliary',False)];accepted=[ss for ss in steps if ss['accepted']];trials=[t for ss in steps for t in ss.get('trials',[])];kk=[t['KL']for ss in accepted for t in ss['trials']if t.get('acceptable')and t.get('alpha')==ss.get('selected_alpha')]
  tr=dict(identity=identity,actual_rows=rowcount,physics_ticks_in_onpolicy_rows=physical,completed_episodes=len(episodes),independent_parents=len({ep['spec']['parent_id']for ep in episodes}),native_reference_rows_not_PPO=native,phase_rows=dict(phase_rows),option_rows=[dict(phase=k[0],option=int(k[1]),rows=v)for k,v in sorted(options.items())],selector_draws=[dict(phase=k[0],option=int(k[1]),draws=v)for k,v in sorted(catdraw.items())],coverage=[dict(source=k[0],family=k[1],learning_phase=k[2],physical_phase=k[3],goal=k[4],rows=v,physical_duration_s=.02*v)for k,v in sorted(coverage.items())],completed_episode_coverage=[dict(source=k[0],family=k[1],episodes=v)for k,v in sorted(complete.items())],accepted_actor_updates=len(accepted),attempts=len(steps),selected_alpha=cnt(str(ss.get('selected_alpha'))for ss in accepted),max_old_logprob_error=max(rr['metrics']['old_logprob_max_error']for rr in study['records']),accepted_joint_KL_p50_p95_max=np.quantile([k['joint']for k in kk],[.5,.95,1]).tolist(),accepted_categorical_KL_p50_p95_max=np.quantile([k['categorical']for k in kk],[.5,.95,1]).tolist(),blocks=len(study['records']),auxiliary_updates_accepted=sum(bool(ss.get('auxiliary')) and ss['accepted'] for ss in all_steps))
  training[arm]=tr;jw('training-coverage.json',training);print('TRAINING',arm,rowcount,flush=True)
 report=dict(status='FINAL_TRAINING_COVERAGE_COMPLETE',training=training,physics_ticks=0,policy_updates=0,sealed_read=False)
except BaseException as ex:
 report=dict(status='FINAL_READONLY_FAILED',error=repr(ex),traceback=traceback.format_exc(),physics_ticks=0,policy_updates=0,sealed_read=False)
finally:
 report['wall_s']=time.monotonic()-START;report['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();jw('result.json',report)
 lp=S/'ledger.json';ledger=json.loads(lp.read_text());ledger['charged_s']+=report['wall_s'];ledger['jobs'].append(dict(name=OUT.name,status=report['status'],wall_s=report['wall_s'],physics=dict(actual=0,reserved=0),source_sha256=report['source_sha256']));tmp=lp.with_suffix('.pending');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,lp)
 print(json.dumps({k:v for k,v in report.items()if k not in('training','joint_success','bootstrap')}),flush=True)
if report['status']!='FINAL_TRAINING_COVERAGE_COMPLETE':raise RuntimeError(report['error'])
