"""CPU-only tensor/data audit; no BFM/Student forward, simulation, or fit."""
import json
from pathlib import Path
import torch,numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');M3=R/'m3-continuous-body-20261031A/train-attempt-01/outputs';OUT=R/'m4-continuous-feedback-20261031A/analysis'
def main():
 torch.set_num_threads(1);d=torch.load(M3/'dataset.pt',map_location='cpu',weights_only=True);ck=torch.load(M3/'projected-z-final.pt',map_location='cpu',weights_only=True)['state_dict'];z=torch.load(M3/'projected-global-z.pt',map_location='cpu',weights_only=True);h=(d['h']-ck['center'])/ck['scale'];meta=d['metadata'];pairs=[];byconfig={s:torch.tensor([i for i,m in enumerate(meta) if m['configuration']==s]) for s in {m['configuration'] for m in meta}}
 for name,ix in sorted(byconfig.items()):
  others=torch.tensor([i for i,m in enumerate(meta) if m['configuration']!=name]);rows=[]
  for st in range(0,len(ix),64):
   ii=ix[st:st+64];distance=torch.cdist(h[ii],h[others],compute_mode='donot_use_mm_for_euclid_dist')/h.shape[1]**.5;vals,cols=distance.min(1)
   for v,i,jj in zip(vals,ii,cols):
    i=int(i);j=int(others[jj]);ang=float(torch.rad2deg(torch.acos(((z[i]*z[j]).sum()/(z[i].norm()*z[j].norm())).clamp(-1,1))));e=d['q'][i]-d['q'][j];rows.append(dict(distance=float(v),i=i,j=j,source_i=meta[i],source_j=meta[j],global_z_angle_deg=ang,arm_q_deg=float(torch.rad2deg(e[22:].square().mean().sqrt())),nonarm_q_deg=float(torch.rad2deg(e[:22].square().mean().sqrt()))))
  pairs+=sorted(rows,key=lambda r:r['distance'])[:5]
  pairs+=sorted([r for r in rows if r['distance']<=.1],key=lambda r:r['arm_q_deg'],reverse=True)[:3]
 result=dict(samples=len(h),configurations={k:len(v) for k,v in byconfig.items()},pairs=pairs,scope='Crossconfiguration actualglobal-z/qPD nearestinput audit. .1 used only to expose nearby differingtargets, not a scientific conflict or OOD threshold. No arbitrary coordinate-label comparison; finite local data cannot prove universal identifiability. All forcehistory+nativeproprio/action features retained.')
 OUT.mkdir(exist_ok=True);(OUT/'cross-configuration-labels.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
