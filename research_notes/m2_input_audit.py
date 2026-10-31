"""Saved feature/state audit only; CPU tensor statistics, no BFM/Student forward or fit."""
import gzip,json
from pathlib import Path
import numpy as np,torch
from m2_rule import infer_coordinate
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs');R=P/'m2-unseen-body-20261031A';M=P/'m1-state-response-20261031A'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 torch.set_num_threads(1);d=torch.load(M/'aggregate-attempt-01/outputs/dataset.pt',map_location='cpu',weights_only=True);ck=torch.load(M/'aggregate-attempt-01/outputs/distill-final.pt',map_location='cpu',weights_only=True)['state_dict'];train=(d['h']-ck['center'])/ck['scale'];configs=[m['configuration'] for m in d['metadata']];assert set(configs)=={'standing','raised'}
 out=R/'comparison-attempt-01/outputs';refs=read(out/'references.json')['proprio_refs'];records=[]
 for c in ['distill','alignment']:
  for free in [False,True]:
   ep=read(out/('midpoint__'+c+('__free' if free else '')+'.json.gz'));cs=ep['control'];on=ep['condition']['onset'];before=min(cs,key=lambda x:abs(x['time']-on));ready=min(cs,key=lambda x:abs(x['time']-ep['control_ready_s']));wdelta=(np.array(before['pre_wrist'])-ready['pre_wrist'])*100
   selected=[]
   for t in [10.4,11.4,12.4,14.4,16.8,18.4,21.4,23.3]:
    r=min(cs,key=lambda x:abs(x['time']-t));h=torch.tensor(r['student_input']);x=(h-ck['center'])/ck['scale'];dist=(train-x).square().mean(-1).sqrt();prop=(train[:,:465]-x[:465]).square().mean(-1).sqrt();nearest=int(dist.argmin());selected.append(dict(time=r['time'],native_coords=r['native_coords'],actual_proprio_coordinate=infer_coordinate(r['pre_qpos'][29:36],refs),force_current_xyz=r['measured_force_xyz'],force_history_max_abs_normalized=float(h[465:].abs().max()),gate=r['local_gate'],nearest_training_group=d['metadata'][nearest],nearest_normalized_input_rms=float(dist[nearest]),minimum_proprio_rms_by_config={k:float(prop[torch.tensor([v==k for v in configs])].min()) for k in ['standing','raised']}))
   records.append(dict(label=ep['label'],precontact_wrist_delta_xyz_cm=wdelta.tolist(),precontact_wrist_displacement_cm=float(np.linalg.norm(wdelta)),records=selected))
 result=dict(training_samples=len(configs),training_configurations=sorted(set(configs)),records=records,meaning='Feature distances are descriptive only; no global identifiability or causality proof. u is native parameter coordinate, not an exact body state. Actual proprio coordinate is projection on prior physical templates, not ground-truth latent. All zero-force precontact action uses no contact information. No fitting.')
 (R/'analysis/input-state-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
