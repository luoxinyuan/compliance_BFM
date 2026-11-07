"""Issue51: exact real-command replay, independent privileged/restricted BC, real rollout.
No search prediction is used as a label; no PPO or online old policy in learned actors.
"""
import argparse,copy,gzip,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
S=ROOT.parent;OLD=S.parent/'native-rolling-h100-20261107'
PLAN=json.loads((ROOT/'research_notes/native_transfer_protocol.json').read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def study(e,m,contract,out):
 import numpy as np,torch,mujoco
 import native_env
 from native_env import Task,install
 from clean_policy import Expert,direct_command
 from geometry_policy import expand
 from locomotion_rollout import spherical
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 torch.set_num_threads(1);torch.set_grad_enabled(False);install()
 write(out/'runtime-speed.json',speed(e));write(out/'backend.json',backend(e,m,S/'compatibility-prefix'))
 native_env.RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106')
 task=Task(e,m,contract,5001);dev=e.standing.device
 assert set(e.keys)=={'state','last_action','history_actor'}
 cases=json.loads((ROOT/'research_notes/native_transfer_cases.json').read_text())
 src=[]
 for label in ['rolling_continue','rolling_stop']:src.append(dict(label=label,folder=OLD/'runs/rolling-02',split='train',group='old-prefix'))
 for c in cases:src.append(dict(label=c['name'],folder=S/'runs/teacher-coverage-01',split=c['split'],group=c['group']))
 if ARGS.mode=='prepare':
  examples=[];manifest=[];checks=[];counts=dict(actual_replay_physics_steps=0,reset_source_physics_steps=0,prediction_label_count=0)
  for item in src:
   label=item['label'];path=item['folder']/(label+'.json.gz');ep=read(path)
   queries=[q for q in read(item['folder']/'searches.json') if q['label']==label]
   if ep['boundary'] is not None:
    manifest.append(dict(episode=label,split=item['split'],group=item['group'],source=str(path),source_sha256=sha(path),samples=0,valid=0,condition=ep['condition'],boundary=ep['boundary'],excluded='unsafe source retained for evaluation, not labels'))
    continue
   controls=ep['control'];assert len(controls)==5*len(queries)
   task.reset(ep['condition'],record=True,label='replay_'+label);counts['reset_source_physics_steps']+=600+len(task.prefix_data['physics'])
   errmax=dict(privileged=0.,restricted=0.,previous_z=0.,qpos=0.,qvel=0.,qpd=0.,target_z=0.)
   n0=len(examples)
   for j,q in enumerate(queries):
    assert abs(float(e.tether.time)-q['time'])<1e-6
    px=task.features().detach().clone();sx=task.student_features().detach().clone()
    assert px.shape==(1,1088) and sx.shape==(1,1639) and task.pro.shape[-1]==465
    prev=task.last_command.detach().clone();z=torch.tensor(q['selected_first_z'],device=dev,dtype=torch.float32)[None]
    errmax['privileged']=max(errmax['privileged'],float((px-torch.tensor(q['privileged_input'],device=dev)).abs().max()))
    errmax['restricted']=max(errmax['restricted'],float((sx-torch.tensor(q['restricted_ideal_force_input'],device=dev)).abs().max()))
    errmax['previous_z']=max(errmax['previous_z'],float((prev-torch.tensor(q['previous_executed_z'],device=dev)).abs().max()))
    saved=controls[5*j];errmax['qpos']=max(errmax['qpos'],float(np.max(np.abs(e.sim.data.qpos-saved['pre_qpos']))));errmax['qvel']=max(errmax['qvel'],float(np.max(np.abs(e.sim.data.qvel-saved['pre_qvel']))))
    # Stored physics/solver and real histories are metadata only, never network input.
    integ=np.empty(mujoco.mj_stateSize(e.sim.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(e.sim.model,e.sim.data,integ,mujoco.mjtState.mjSTATE_INTEGRATION)
    point=task.read_point_state()[0];current_error=float(np.linalg.norm(e.tether.anchor-point)) if e.tether.active else 0.
    all_bad=all(c['metrics']['force_rms']>=9.99 for c in q['candidates'])
    valid=not(current_error>.25 and all_bad)
    row=dict(privileged=px[0].cpu(),restricted=sx[0].cpu(),z=z[0].cpu(),raw=(z[0]/16).cpu(),previous_z=prev[0].cpu(),obs={k:v.detach().cpu().clone() for k,v in e.obs.items() if k in e.keys},integration=torch.tensor(integ),qacc_warmstart=torch.tensor(e.sim.data.qacc_warmstart.copy()),qfrc_constraint=torch.tensor(e.sim.data.qfrc_constraint.copy()),episode=label,group=item['group'],split=item['split'],time=q['time'],valid=valid,invalid_reason=None if valid else 'all predictions full-cap and current error>25cm',source='actual first command, exact full physical replay',control=copy.deepcopy(controls[5*j:5*j+5]),contacts=copy.deepcopy(e.tether.geometry_rows),support=np.asarray(e.last_support).tolist())
    _,_,_,st=task.step(z/16);counts['actual_replay_physics_steps']+=st['physics_steps']
    errmax['qpd']=max(errmax['qpd'],float(np.max(np.abs(np.array([c['q_PD'] for c in task.controls[-5:]])-np.array([c['q_PD'] for c in controls[5*j:5*j+5]])))))
    errmax['target_z']=max(errmax['target_z'],float((task.last_command-z).abs().max()))
    # Same backend deterministic replay: do not silently relax this into a fit.
    assert max(errmax[k] for k in ['privileged','restricted','previous_z','qpos','qvel','qpd'])<1e-6,(label,j,errmax)
    assert errmax['target_z']<1e-4
    examples.append(row)
   checks.append(dict(episode=label,max_errors=errmax,samples=len(examples)-n0,actual_source_sha256=sha(path),search_file_sha256=sha(item['folder']/'searches.json')))
   manifest.append(dict(episode=label,split=item['split'],group=item['group'],source=str(path),source_sha256=sha(path),samples=len(examples)-n0,valid=sum(r['valid'] for r in examples[n0:]),condition=ep['condition'],boundary=ep['boundary']))
   write(out/'replay-checks.json',checks);print('EXACT_REAL_REPLAY',label,len(examples)-n0,errmax,flush=True)
  group_splits={}
  for item in src:
   group_splits.setdefault(item['group'],set()).add(item['split'])
  assert all(len(v)==1 for v in group_splits.values()),group_splits
  write(out/'split-audit.json',{k:sorted(v) for k,v in group_splits.items()})
  torch.save(dict(rows=examples,standing=e.standing[0].cpu(),manifest=manifest,scope='private actual executed commands only; metadata never concatenated to actor'),out/'dataset.pt')
  write(out/'manifest.json',manifest);write(out/'counts.json',counts)
  write(out/'observation-audit.json',dict(privileged_dim=1088,restricted_dim=1639,bfm_keys=list(e.keys),proprio=465,own_actual_z=256,real_force_history=900,known_mounts=18,restricted_excluded=['anchor/error','partner true velocity','future','task or episode ID','clock','active truth','MPC phase/kind/scores'],source_rows='whole episode/group split, never adjacent random split',ideal_force_not_G1_sensor=True,queries=len(examples),predicted_labels=0))
  return dict(status='completed',mode='prepare',samples=len(examples),manifest=manifest,counts=counts)
 data=torch.load(S/'runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True)
 if ARGS.mode=='eval':
  summaries=[];models={};fitout=S/'runs/bc-fit-01'
  def run(label,s,net,permission,collect=False):
   task.reset(s,record=True,label=label);query_rows=[]
   while not task.done:
    x=task.features() if permission=='P' else task.student_features()
    z=direct_command(net.distribution(x).mean,e.standing)
    if collect:query_rows.append(dict(time=float(e.tether.time),input=x[0].cpu().tolist(),target_z=z[0].cpu().tolist(),previous_z=task.last_command[0].cpu().tolist()))
    task.step(z/16)
   row=task.save(out)
   # Reuse the unchanged Issue50 physical-unit reporter, not old anchor after release.
   row.update(phase_report(task));row['controller']=permission;row['model_sha256']=models.get(id(net))
   summaries.append(row);write(out/'summaries.json',summaries)
   if collect:
    with gzip.open(out/(label+'-actual-queries.json.gz'),'wt') as f:json.dump(query_rows,f)
   print('BC_REAL',label,row['all']['error_cm'],row['all']['force_rms'],row['boundary'],flush=True)
   return row
  def load(role,step):
   ck=torch.load(fitout/(role+'-'+str(step)+'.pt'),map_location=dev,weights_only=True);net=Expert(ck['input_dim'],'direct').to(dev);net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False);models[id(net)]=sha(fitout/(role+'-'+str(step)+'.pt'));return net
  devspec=next(c['spec'] for c in cases if c['split']=='dev');selection={}
  for role in ['P','S']:
   scored=[]
   for step in PLAN['bc']['checkpoints']:
    net=load(role,step);row=run(role+'_dev_'+str(step),devspec,net,role)
    a=row['all'];score=a['force_rms']+.03*(a['error_cm'] or 0)+a['cap_fraction']+20*bool(row['boundary'])
    scored.append(dict(step=step,score=score,sha256=models[id(net)],metrics=a))
   selected=min(scored,key=lambda x:x['score']);selection[role]=dict(selected=selected,candidates=scored);write(out/'selection.json',selection)
   net=load(role,selected['step'])
   evalitems=[(x['label'],read(x['folder']/(x['label']+'.json.gz'))['condition']) for x in src]
   no=copy.deepcopy(cases[0]['spec']);no.update(enabled=False,distance=0.,ramp=4.,release=11.2,end=15.2);evalitems.append(('no_actual_contact',no))
   for name,s in evalitems:run(role+'_selected_'+name,s,net,role,True)
   # Same Teacher-visited state, actual five BFM observations; replay is not physics rollout.
   checks=[]
   rows=[r for r in data['rows'] if r['valid']][::max(1,len(data['rows'])//80)]
   for row in rows:
    x=row['privileged' if role=='P' else 'restricted'].to(dev)[None];zs=direct_command(net.distribution(x).mean,e.standing);zt=row['z'].to(dev)[None];prev=row['previous_z'].to(dev)[None];ds=[];parity=[]
    for k,c in enumerate(row['control']):
     ob={key:torch.tensor(c['bfm_obs'][key],device=dev,dtype=torch.float32) for key in e.keys}
     qs=task.qof(m.act(ob,spherical(prev,zs,(k+1)/5),mean=True));qt=task.qof(m.act(ob,spherical(prev,zt,(k+1)/5),mean=True))
     ds.append(float(torch.rad2deg(qs-qt).square().mean().sqrt()));parity.append(float((qt-torch.tensor(c['q_PD'],device=dev)).abs().max()))
    checks.append(dict(episode=row['episode'],time=row['time'],split=row['split'],angle_deg=float(torch.rad2deg(torch.acos(((zs*zt).sum()/256).clamp(-1,1)))),five_substep_qpd_rms_deg=ds,teacher_recompute_max_rad=max(parity)))
   write(out/(role+'-same-state-alignment.json'),checks)
  # Old frozen policies only references, never actor inputs or default training labels.
  from native_policy_coordinate import Expert as CoordinateExpert
  refs=[('I','m39a-demand-20261104/B-ppo-03/job/best.pt','72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a',Expert),('C0','prior-recovery-20261105/correct-C0/C0.pt','ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c',Expert),('C32','native-contact-20261106/ppo-coordinate/last.pt','39cd0db55fddb174de3f77104acbd8d7574fb3a5051f743e4ca39bb6b7d24ad0',CoordinateExpert)]
  base=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
  for name,path,digest,cls in refs:
   p=base/path;assert sha(p)==digest;ck=torch.load(p,map_location=dev,weights_only=True);net=cls(1088,'direct').to(dev)
   if name=='C32':net.load_state_dict(ck['state_dict'])
   else:expand(net,ck)
   net.eval().requires_grad_(False);models[id(net)]=digest
   for c in cases:run(name+'_'+c['name'],c['spec'],net,'P')
  return dict(status='completed',mode='eval',selection=selection,episodes=len(summaries),student_PPO=0)
 raise ValueError(ARGS.mode)
def metrics(rows,controls):
    import numpy as np
    active=[r for r in rows if r['active']]
    f=np.array([r['force'] for r in active]);err=np.array([np.array(r['anchor'])-r['wrist'] for r in active])
    root=np.array([r['root'] for r in rows]);vel=np.array([r['qvel'][:3] for r in rows]);feet=np.array([r['feet_xyz'] for r in rows]);q=np.array([r['qpos'][7:] for r in rows]);qd=np.array([r['q_PD'] for r in controls])
    yaw=np.unwrap(np.array([np.arctan2(np.array(r['pelvis_matrix'])[1,0],np.array(r['pelvis_matrix'])[0,0]) for r in rows]))
    ans=dict(physics_steps=len(rows),error_cm=float(100*np.sqrt(np.mean(np.sum(err*err,1))))) if len(active) else dict(physics_steps=len(rows),error_cm=None)
    ans.update(force_rms=float(np.sqrt(np.mean(np.sum(f*f,1)))) if len(active) else 0.,force_xyz_rms=np.sqrt(np.mean(f*f,0)).tolist() if len(active) else [0]*3,signed_error_cm=(100*err.mean(0)).tolist() if len(active) else [0]*3,force_xyz_mean=f.mean(0).tolist() if len(active) else [0]*3,cap_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=9.999)) if len(active) else 0.,root_delta_cm=(100*(root[-1]-root[0])).tolist(),root_vx_mean=float(vel[:,0].mean()),root_speed_mean=float(np.linalg.norm(vel[:,:2],axis=1).mean()),pelvis_yaw_change_deg=float(np.degrees(yaw[-1]-yaw[0])),foot_paths_cm=(100*np.linalg.norm(np.diff(feet,axis=0),axis=2).sum(0)).tolist(),slip_proxy_mean=float(np.mean([r['contact_tangent_speed_max_mps'] for r in rows])),qpd_max_step_deg=float(np.degrees(np.abs(np.diff(qd,axis=0))).max()) if len(qd)>1 else 0.,q_actual_max_step_deg=float(np.degrees(np.abs(np.diff(q,axis=0))).max()) if len(q)>1 else 0.,min_height=float(min(r['height'] for r in rows)),max_tilt=float(max(r['tilt_deg'] for r in rows)),torque_saturation_max=float(max(r['torque_saturation'] for r in rows)),work_J=float(sum(r['external_power']*.005 for r in rows)))
    return ans

def phase_report(task):
 import numpy as np
 s=task.spec;out=dict(all=metrics(task.physics,task.controls));stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0)
 for name,a,b in [('moving',s['onset'],stop),('decelerating',stop,hold),('holding',hold,s['release']),('released',s['release'],s['end'])]:
  rs=[r for r in task.physics if a<=r['force_time']<b];cs=[r for r in task.controls if a<=r['time']<b]
  if rs:out[name]=metrics(rs,cs)
 return out
def fit(out):
 import numpy as np,torch
 from clean_policy import Expert,direct_command
 torch.set_num_threads(1);dev='cuda';d=torch.load(S/'runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True)
 rows=[r for r in d['rows'] if r['valid'] and r['split']=='train'];assert rows
 groups=[np.array([i for i,r in enumerate(rows) if r['episode']==name]) for name in sorted(set(r['episode'] for r in rows))]
 rng=np.random.default_rng(PLAN['bc']['seed']);order=[rng.choice(groups[int(rng.integers(len(groups)))],size=PLAN['bc']['batch'],replace=True) for _ in range(PLAN['bc']['steps'])]
 np.save(out/'matched-batch-indices.npy',np.stack(order));z=torch.stack([r['z'] for r in rows]).to(dev);standing=d['standing'].to(dev)[None];logs={}
 for role,key in [('P','privileged'),('S','restricted')]:
  torch.manual_seed(PLAN['bc']['seed']);x=torch.stack([r[key] for r in rows]).to(dev);net=Expert(x.shape[-1],'direct').to(dev);net.center.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.05));net.actor[-1].bias.data.copy_(standing[0]/16);net.requires_grad_(False);net.actor.requires_grad_(True);opt=torch.optim.Adam(net.actor.parameters(),lr=PLAN['bc']['lr'])
  def save(step):
   torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},input_dim=x.shape[-1],kind='direct',role=role,steps=step,seed=PLAN['bc']['seed'],runtime=RUNTIME,teacher='native rolling real executed first command',dataset_sha256=sha(S/'runs/data-prepare-01/dataset.pt'),no_old_actor_initialization=True),out/(role+'-'+str(step)+'.pt'))
  save(0);curve=[];start=time.monotonic()
  for j,ix in enumerate(order,1):
   ix=torch.tensor(ix,device=dev);pred=direct_command(net.distribution(x[ix]).mean,standing);loss=(pred-z[ix]).square().mean();opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(net.actor.parameters(),1));opt.step()
   if j%100==0:
    with torch.no_grad():
     zz=direct_command(net.distribution(x).mean,standing);angles=torch.rad2deg(torch.acos(((zz*z).sum(-1)/256).clamp(-1,1)));full=float((zz-z).square().mean())
    curve.append(dict(step=j,minibatch_loss=float(loss.detach()),train_loss=full,train_angle_mean=float(angles.mean()),train_angle_max=float(angles.max()),gradient=grad,elapsed_s=time.monotonic()-start))
    write(out/(role+'-curve.json'),curve);print('BC_UPDATE',role,j,full,float(angles.mean()),flush=True)
   if j in PLAN['bc']['checkpoints']:save(j)
  logs[role]=dict(valid_samples=len(rows),episodes=len(groups),input_dim=x.shape[-1],trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),steps=len(order),elapsed_s=time.monotonic()-start,curve=curve)
 write(out/'training.json',logs);return dict(status='completed',mode='fit',updates=2*PLAN['bc']['steps'],models=logs,student_PPO=0)
def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','fit','eval'],required=True);p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args();out=ARGS.output;out.mkdir(exist_ok=False)
 import subprocess
 assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();write(out/'launch.json',dict(runtime=RUNTIME,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'research_notes/native_transfer_protocol.json'),mode=ARGS.mode))
 if ARGS.mode=='fit':
  write(out/'result.json',fit(out));return
 os.environ['BFM_RECOVERY_OUTPUT']=str(out);os.chdir(out)
 import m10_physics,m10_stage
 m10_physics.OUT=out;m10_physics.MODEL=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view');m10_stage.run_study=study;m10_physics.main()
if __name__=='__main__':main()
