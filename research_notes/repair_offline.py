"""Frozen teacher-forced vs student-visited inference. No rollout or optimizer."""
import hashlib,json,os,time
from pathlib import Path
from locomotion_rollout import UnsafePhysics


def run_study(env,model,contract,out):
 import numpy as np
 import torch
 from alpha_memory import ForceMemory
 from alpha_student import load_student,student_input,path_z
 notes=Path(__file__).parent;read=lambda p:json.loads(Path(p).read_text());protocol=read(notes/'alpha_repair_protocol.json')
 runs=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');run=runs/'issue21-alpha-repair-20261031A';physical=run/'physics-attempt-01/outputs'
 ledger=read(run/'compute-ledger.json');training=next(Path(a['path'])/'outputs' for a in ledger['attempts'] if a['stage']=='train' and a.get('result_status')=='completed');alpha_protocol=protocol;fit=read(training/'paired-fit-result.json');dev=env.standing.device;assert dev.type=='cuda'
 saved=read(physical/'native-command.json');goal=torch.tensor(saved['z'],device=dev);assert torch.equal(env.standing,torch.tensor(saved['standing_z'],device=dev))
 dataset=torch.load(training/'training-dataset.pt',weights_only=True,map_location='cpu');tx=dataset['x'].to(dev);ty=dataset['target_alpha'].to(dev).reshape(-1)
 nets={};meta={}
 for name in ['A','B']:
  p=training/alpha_protocol['final_checkpoints'][name];assert hashlib.sha256(p.read_bytes()).hexdigest()==fit['full_'+name]['checkpoint_sha256']
  nets[name],meta[name]=load_student(p,dev);assert meta[name]['updates']==2000
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 q_of=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 results={};keys=alpha_protocol['memory']['keys']
 for condition in protocol['conditions']:
  if condition['split']!='sealed_new_gap':continue
  start=condition['disconnect_start']
  for source_ctl in ['contact','A','B']:
   label=condition['name']+'__'+source_ctl;p=physical/(label+'.json');ep=read(p);cs=ep['control'];fifo=ForceMemory(320)
   history=[fifo.append(c['measured_force_xyz']) for c in cs]
   obs={k:torch.tensor([c['bfm_obs'][k][0] for c in cs],device=dev,dtype=torch.float32) for k in keys}
   x=student_input(model,obs,history,keys);target=torch.tensor([c.get('shadow_alpha',c['alpha']) for c in cs],device=dev)
   tq=torch.tensor([c.get('shadow_q_PD',c['q_PD']) for c in cs],device=dev);times=np.asarray([c['time'] for c in cs])
   for name in (['A','B'] if source_ctl=='contact' else [source_ctl]):
    net=nets[name];pred=[];qs=[]
    for ids in torch.arange(len(x),device=dev).split(64):
     assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
     alpha=net(x[ids]);z=path_z(env.standing,goal,alpha);a=model.act({k:v[ids] for k,v in obs.items()},z,mean=True)
     pred.append(alpha[:,0]);qs.append(q_of(a))
    pred=torch.cat(pred);qs=torch.cat(qs);namekey=label+'__predict_'+name
    result=dict(source_controller=source_ctl,student=name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),phases={})
    for phase,lo,hi in [('pre_gap',3,start),('gap',start,condition['disconnect_end']),('resume1s',condition['disconnect_end'],condition['disconnect_end']+1),('recovery',10.4,12.4),('gap_resume1s',start,condition['disconnect_end']+1)]:
     m=torch.tensor((times>=lo-1e-8)&(times<hi-1e-8),device=dev)
     result['phases'][phase]=dict(alpha_rmse=float((pred[m]-target[m]).square().mean().sqrt()),alpha_max=float(abs(pred[m]-target[m]).max()),qPD_rms_deg=float(torch.rad2deg((qs[m]-tq[m]).square().mean().sqrt())))
    if source_ctl==name:
     recorded=torch.tensor([c['alpha'] for c in cs],device=dev);qa=torch.tensor([c['q_PD'] for c in cs],device=dev)
     result['deployment_replay_alpha_max']=float(abs(pred-recorded).max());result['deployment_replay_qPD_max_rad']=float(abs(qs-qa).max())
     assert result['deployment_replay_alpha_max']<2e-6 and result['deployment_replay_qPD_max_rad']<2e-5
    xt=(tx-net.center)/net.scale;xx=(x-net.center)/net.scale;ds=(xx.square().sum(1)[:,None]+xt.square().sum(1)[None,:]-2*xx@xt.T).clamp_min(0)/x.shape[1]
    dist,ix=ds.min(1);roi=torch.tensor((times>=start-1e-8)&(times<condition['disconnect_end']+1-1e-8),device=dev)
    result['training_input_nn_rms_quantiles_gap_resume']=torch.quantile(dist[roi].sqrt(),torch.tensor([.5,.9,1],device=dev)).tolist()
    result['nearest_train_teacher_alpha_gap_quantiles']=torch.quantile(abs(target[roi]-ty[ix[roi]]),torch.tensor([.5,.9,1],device=dev)).tolist()
    result['alpha_landmarks']={str(round(t,3)):dict(pred=float(pred[int(abs(times-t).argmin())]),target=float(target[int(abs(times-t).argmin())])) for t in [start-.02,start+.06,condition['disconnect_end'],condition['disconnect_end']+.12,condition['disconnect_end']+.5,condition['disconnect_end']+1]}
    results[namekey]=result
    (out/(namekey+'.json')).write_text(json.dumps(dict(time=times.tolist(),alpha=pred.tolist(),target_alpha=target.tolist(),q_PD=qs.tolist()),indent=2))
    print('FROZEN_DIAGNOSTIC',namekey,json.dumps(result),flush=True)
 for name in nets:
  assert all(p.grad is None and not p.requires_grad for p in nets[name].parameters())
  assert hashlib.sha256((training/alpha_protocol['final_checkpoints'][name]).read_bytes()).hexdigest()==fit['full_'+name]['checkpoint_sha256']
 summary=dict(status='completed',frozen_diagnostics=results,policy_updates=0,new_physical_branches=0,no_model_changes=True)
 (out/'frozen-diagnosis.json').write_text(json.dumps(summary,indent=2));return summary
