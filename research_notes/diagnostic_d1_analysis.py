"""Archive-only development-versus-training coverage. No substitute training NN for dev traces."""
import csv,gzip,json,math,time
from pathlib import Path
import numpy as np
import torch
from dual_track_high import BC,write
from dual_track_returns import advantages
OLD=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/dual-track-formal-20261109')

def run(folder,out):
 start=time.monotonic();torch.set_num_threads(2);out.mkdir(exist_ok=False)
 data={};table=[];own={};inputs={}
 for p in folder.glob('*-complete.pt'):
  if p.name.startswith('3ffd2fc10497') or p.name.startswith('42f40dfe854d') or p.name.startswith('1d3ee72c35d3'):
   d=torch.load(p,map_location='cpu',weights_only=False);label=p.name.removesuffix('-complete.pt');data[label]=d
   for h in d['high']:
    table.append(dict(label=label,time=h['time'],raw_l2=h['raw_l2'],z_angle_deg=h['z_angle_deg'],
     action_rms=h['action_rms'],qpd_rms_deg=h['qpd_rms_deg'],qpd_max_deg=h['qpd_max_deg'],
     **{f'qpd_substep{i+1}_max_deg':v for i,v in enumerate(h['qpd_substep_max_deg'])}))
 with (out/'same-state-five-substeps.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 # First observed own-trajectory divergence. Full continuous values retained.
 for prefix in ['3ffd2fc10497','42f40dfe854d','1d3ee72c35d3']:
  a=data[prefix+'-i0'];b=data[prefix+'-i48'];rows=[]
  for aa,bb in zip(a['high'],b['high']):
   assert abs(aa['time']-bb['time'])<1e-7
   raw0=aa['raw'][0];raw48=bb['raw'][1];z0=aa['z'][0];z48=bb['z'][1]
   rows.append(dict(time=aa['time'],raw_l2=float((raw48-raw0).norm()),
      z_angle_deg=math.degrees(math.acos(float(((z0*z48).sum()/256).clamp(-1,1)))),
      same_state_qpd_onI_deg=aa['qpd_max_deg'],same_state_qpd_on48_deg=bb['qpd_max_deg'],
      qpos_max_abs=float(np.max(np.abs(a['states'][len(rows)]['mjdata_arrays']['qpos']-b['states'][len(rows)]['mjdata_arrays']['qpos'])))))
  own[prefix]=rows
 write(out/'own-trajectory-divergence.json',own)
 # Actual newly rebuilt dev actor inputs, by frame; full train1:48 is independently indexed.
 norm=torch.load(BC,map_location='cpu',weights_only=True)['state_dict'];center=norm['center'];scale=norm['scale']
 print('normalizer',center.shape,scale.shape,flush=True)
 def transform(x):
  x=x.float().clone();x[:,:1639]=((x[:,:1639]-center)/scale).clamp(-10,10);return x
 trains=[];metadata=[]
 for batch in range(1,49):
  z=np.load(OLD/f'runs/expert-kl002/rollout-{batch}.npz');stats=json.loads((OLD/f'runs/expert-kl002/rollout-{batch}-stats.json').read_text())
  with torch.no_grad():
   ad,_=advantages(*[torch.tensor(z[k]) for k in ['reward','value','next_value','discount','terminated','truncated']],.95)
  trains.append(transform(torch.from_numpy(z['obs'])))
  for j,s in enumerate(stats):metadata.append(dict(batch=batch,index=j,parent=s['parent_id'],source=s['source'],family=s['family'],phase=s['phase'],advantage=float(ad[j]),reward=float(z['reward'][j])))
 train=torch.cat(trains);devs=[];ids=[]
 for label,d in data.items():
  inputs[label]=transform(torch.cat([r['input'] for r in d['high']]))
  for j,h in enumerate(d['high']):ids.append(dict(label=label,time=h['time'],frame=j))
  devs.append(inputs[label])
 dev=torch.cat(devs);blocks=dict(proprio=(0,465),last_z=(465,721),force_history=(721,1621),mount=(1621,1639),privileged=(1639,2727))
 results=[]
 for name,(lo,hi) in blocks.items():
  a=dev[:,lo:hi];b=train[:,lo:hi];normb=(b*b).sum(1)[None];nearest=[]
  for begin in range(0,len(a),128):
   x=a[begin:begin+128];dist=((x*x).sum(1)[:,None]+normb-2*x@b.T).clamp_min(0)/(hi-lo)
   values,indices=dist.topk(5,largest=False)
   for vals,ixs in zip(values,indices):nearest.append([(float(v.sqrt()),int(ix)) for v,ix in zip(vals,ixs)])
  for idx,nn in enumerate(nearest):
   r=dict(**ids[idx],block=name,rms_distance=nn[0][0],nearest=[dict(distance=v,**metadata[ix]) for v,ix in nn]);results.append(r)
 write(out/'development-to-training-block-coverage.json',results)
 # Independent dev-to-dev block distances: contains real per-frame trajectories.
 overlaps={}
 for label,a in inputs.items():
  if not label.startswith('3ffd'):continue
  for other,b in inputs.items():
   if other.startswith('3ffd'):continue
   overlaps[label+'->'+other]={}
   for name,(lo,hi) in blocks.items():
    d=torch.cdist(a[:,lo:hi],b[:,lo:hi])/math.sqrt(hi-lo);v,j=d.min(1)
    overlaps[label+'->'+other][name]=dict(minimum=float(v.min()),median=float(v.median()),p95=float(torch.quantile(v,.95)),
      nearest_frame_indices=j.tolist(),distances=v.tolist())
 write(out/'actual-development-overlap.json',overlaps)
 write(out/'analysis.json',dict(status='completed',training_frames=len(train),actual_dev_frames=len(dev),blocks=blocks,
  limitation='Distances are descriptive; no arbitrary distance threshold proves coverage/label validity. Advantages are original on-policy samples, not counterfactual development advantages.',wall_s=time.monotonic()-start))

if __name__=='__main__':
 import sys
 run(Path(sys.argv[1]),Path(sys.argv[2]))
