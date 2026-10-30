"""Saved real-force suffix identifiability and frozen teacher/own-state probes."""
import hashlib,json,os,time
from pathlib import Path

def diagnose(model,env,goal,contract,students,study,out):
 import numpy as np
 import torch
 from alpha_memory import ForceMemory
 from alpha_student import student_input,path_z
 from contact_gate import ContactGate
 read=lambda p:json.loads(Path(p).read_text())
 cfg=read(Path(__file__).with_name('contact_gate_protocol.json'))['gate']
 dev=env.standing.device;keys=study['memory']['keys'];result={}
 scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
 q_of=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
 conditions=study['matrix'] if os.environ['ALPHA_STAGE']=='diagnostic' else [c for c in study['conditions'] if c['split']=='sealed_final']
 for c in conditions:
  source=Path(c['reference_dir']) if os.environ['ALPHA_STAGE']=='diagnostic' else out
  for ctl in ['contact','A','B']:
   path=source/(c['name']+'__'+ctl+'.json');ep=read(path);cs=ep['control'];fifo=ForceMemory(320)
   histories=[fifo.append(v['measured_force_xyz']) for v in cs]
   obs={k:torch.tensor([v['bfm_obs'][k][0] for v in cs],device=dev,dtype=torch.float32) for k in keys}
   x=student_input(model,obs,histories,keys);target=torch.tensor([v.get('shadow_alpha',v['alpha']) for v in cs],device=dev)
   tq=torch.tensor([v.get('shadow_q_PD',v['q_PD']) for v in cs],device=dev);ts=np.array([v['time'] for v in cs])
   roi=torch.tensor((ts>=c['disconnect_start']-1e-8)&(ts<c['disconnect_end']+1-1e-8),device=dev)
   # Reconstruction uses only the allowed measured-force suffix; labels never enter inference.
   memory_error=0.
   for i,v in enumerate(cs):
    gate=ContactGate(cfg)
    for previous in cs[max(0,i-319):i+1]:a=gate.step(previous['measured_force_xyz'])['alpha']
    memory_error=max(memory_error,abs(a-v.get('shadow_alpha',v['alpha'])))
   for name in (['A','B'] if ctl=='contact' else [ctl]):
    al=[];qs=[]
    for ids in torch.arange(len(cs),device=dev).split(64):
     assert time.monotonic()<float(os.environ['BFM_STAGE_DEADLINE_MONOTONIC'])
     a=students[name](x[ids]);action=model.act({k:v[ids] for k,v in obs.items()},path_z(env.standing,goal,a),mean=True)
     al.append(a[:,0]);qs.append(q_of(action))
    al=torch.cat(al);qs=torch.cat(qs)
    r=dict(alpha_rms=float((al[roi]-target[roi]).square().mean().sqrt()),qPD_rms_deg=float(torch.rad2deg((qs[roi]-tq[roi]).square().mean().sqrt())),force_fifo_reconstruction_max=memory_error,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if ctl==name:
     r.update(replay_alpha_max=float(abs(al-torch.tensor([v['alpha'] for v in cs],device=dev)).max()),replay_qPD_max=float(abs(qs-torch.tensor([v['q_PD'] for v in cs],device=dev)).max()))
     assert r['replay_alpha_max']<2e-6 and r['replay_qPD_max']<2e-5
    result[c['name']+'__'+ctl+'__'+name]=r
 (out/'cross-frozen-probe.json').write_text(json.dumps(result,indent=2))
 return result
