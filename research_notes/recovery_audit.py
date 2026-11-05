"""Saved-trajectory and fixed-data gradient audit; no physical interventions here."""
import argparse,json,gzip,hashlib,time,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--gpu');p.add_argument('--max-seconds',type=float);a,_=p.parse_known_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;a.output.mkdir();start=time.monotonic()
import torch,numpy as np
sys.path.insert(0,'/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/prior-recovery/repo/research_notes')
from clean_policy import Expert,direct_command
from geometry_policy import expand
ROOT=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/prior-recovery/repo/research_notes');OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/latent-prior-audit-20261105')
def read(p):return json.load(gzip.open(p,'rt'))
def write(n,x):(a.output/n).write_text(json.dumps(x,indent=2))
torch.set_num_threads(1);torch.set_grad_enabled(False);cfg=json.loads((ROOT/'recovery_protocol.json').read_text());nets={}
for k,v in cfg['models'].items():
 assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'];net=Expert(1088,'direct').cuda();expand(net,torch.load(v['path'],map_location='cuda',weights_only=True));nets[k]=net.eval().requires_grad_(False)
# exact same datasets as fit2-B, including source manifests, not names inferred from current runs
manifest=json.loads((OLD/'fit2-B/initialization.json').read_text())['manifest'];rows=[];groups={}
for m in manifest:
 rr=read(Path(m['source']));ix=np.linspace(0,len(rr)-1,m['samples']).round().astype(int);groups[m['kind']]=np.arange(len(rows),len(rows)+len(ix));rows += [rr[i] for i in ix]
x=torch.tensor([r['x'] for r in rows],device='cuda');z=torch.tensor([r['z'] for r in rows],device='cuda');stand=z[:1]
identical={};conf=[]
for i,r in enumerate(rows):
 key=np.asarray(r['x'],np.float32).tobytes()
 if key in identical and np.max(abs(np.asarray(r['z'])-rows[identical[key]]['z']))>1e-4:conf.append([identical[key],i])
 identical[key]=i
pairs=[]
for mode in ['all','without_previous_z']:
 xx=nets['I'].normalized(x);xx=xx if mode=='all' else torch.cat([xx[:,:565],xx[:,821:]],-1) # verify layout separately, do not assert hidden-state conflict from distance
 for g,ii in groups.items():
  if g=='walk':continue
  jj=groups['walk'];d=torch.cdist(xx[ii],xx[jj])/xx.shape[-1]**.5;vals,ix=torch.topk(d.flatten(),5,largest=False)
  for val,k in zip(vals,ix):
   i=int(ii[int(k)//len(jj)]);j=int(jj[int(k)%len(jj)]);angle=float(torch.rad2deg(torch.acos((z[i]*z[j]).sum().div(256).clamp(-1,1))))
   pairs.append(dict(mode=mode,groups=[g,'walk'],indices=[i,j],times=[rows[i]['time'],rows[j]['time']],normalized_input_rms=float(val),target_z_angle_deg=angle,scope='neighbor description, not proof of ambiguous input'))
gradients={};cosines={}
for name,net in nets.items():
 net.actor.requires_grad_(True);params=list(net.actor.parameters());gg={};detail={}
 with torch.enable_grad():
  for kind,ii in groups.items():
   raw=net.distribution(x[ii]).mean;zz=direct_command(raw,stand);targetnorm=nets['I'].distribution(x[ii]).mean.norm(dim=-1).detach();lz=(zz-z[ii]).square().mean();ln=(raw.norm(dim=-1)-targetnorm).square().mean();loss=lz+.05*ln
   g=torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,params)]);gg[kind]=g;detail[kind]=dict(samples=len(ii),loss=float(loss),latent_loss=float(lz),norm_loss=float(ln),gradient_norm=float(g.norm()))
 gradients[name]=detail;cosines[name]={f'{g}:walk':float(torch.nn.functional.cosine_similarity(v,gg['walk'],dim=0)) for g,v in gg.items()};net.actor.requires_grad_(False)
write('data-gradient.json',dict(rows=len(rows),manifest=manifest,exact_conflicts=conf,neighbors=pairs,gradients=gradients,cosines=cosines,scope='gradient geometry under exact fixed corpus; no causal inference from cosine alone'))
# first / second round actual traces, aligned real clocks; different-state descriptive only
alltrace={};models={'I':OLD/'qualify-02/10_I_walk.json.gz','A':OLD/'fit2-A/A_walk.json.gz','B':OLD/'fit2-B/B_walk.json.gz','B_first':OLD/'fit-B/B_walk.json.gz'}
for k,path in models.items():alltrace[k]=read(path)
base=alltrace['I'];out={}
for k,e in alltrace.items():
 ph=e['physics'];cs=e['control'];rows2=[]
 for t in [4.,4.2,4.3,4.5,5.,5.5,6.,6.5,7.,8.]:
  p0=min(base['physics'],key=lambda r:abs(r['time']-t));q=min(ph,key=lambda r:abs(r['time']-t));c0=min(base['control'],key=lambda r:abs(r['time']-t));c=min(cs,key=lambda r:abs(r['time']-t));za=np.array(c0['executed_z']);zb=np.array(c['executed_z'])
  rows2.append(dict(time=t,z_angle_deg=float(np.degrees(np.arccos(np.clip((za*zb).sum()/256,-1,1)))),qpd_rms_deg=float(np.sqrt(np.mean(np.degrees(np.array(c['q_PD'])-c0['q_PD'])**2))),q_rms_deg=float(np.sqrt(np.mean(np.degrees(np.array(q['qpos'][7:])-p0['qpos'][7:])**2))),root_delta_from_I_cm=(100*(np.array(q['root'])-p0['root'])).tolist(),wrist_shoulder_cm=(100*(np.array(q['wrist'])-q['shoulder_xyz'])).tolist(),force=q['force'],anchor=q['anchor'],root=q['root'],wrist=q['wrist'],shoulder=q['shoulder_xyz'],feet=q['feet_xyz'],qvel=q['qvel'],support=q['support']))
 sat=next((r['force_time'] for r in ph if any(np.linalg.norm(c['force'])>=c['cap']-.001 for c in r['contacts'])),None)
 sustained={}
 for label,seq,reference,key,threshold in [('z_deg',cs,base['control'],'executed_z',.01),('qPD_deg',cs,base['control'],'q_PD',.01),('q_deg',ph,base['physics'],'qpos',.1)]:
  vals=[]
  for rr,bb in zip(seq,reference):
   aa=np.array(rr[key]);b=np.array(bb[key]);v=float(np.degrees(np.arccos(np.clip((aa*b).sum()/256,-1,1)))) if label=='z_deg' else float(np.sqrt(np.mean(np.degrees(aa[7:]-b[7:] if key=='qpos' else aa-b)**2)));vals.append(v)
  n=5 if key!='qpos' else 20;sustained[label]=next((seq[i]['time'] for i in range(len(vals)-n+1) if all(v>threshold for v in vals[i:i+n])),None)
 out[k]=dict(path=str(models[k]),first_sustained=sustained,thresholds='z/qPD .01deg, actual q .1deg for100ms, descriptive detection only',first_cap=sat,timeline=rows2)
write('saved-divergence.json',out);write('result.json',dict(status='completed',elapsed_s=time.monotonic()-start,rl=0,updates=0));print('AUDIT completed',flush=True)
