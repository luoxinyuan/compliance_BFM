"""Descriptive deployed-input conflict audit, saved tensors only; no model forward or fit."""
import json
from pathlib import Path
import torch
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m1-state-response-20261031A')
torch.set_num_threads(1)
d=torch.load(P/'aggregate-attempt-01/outputs/dataset.pt',map_location='cpu',weights_only=True);s=torch.load(P/'fit-attempt-01/outputs/alignment-final.pt',map_location='cpu',weights_only=True)['state_dict']
ix=torch.arange(0,len(d['h']),5);x=(d['h'][ix]-s['center'])/s['scale'];q=d['q'][ix];metadata=[d['metadata'][int(i)] for i in ix];dist=torch.cdist(x,x)/x.shape[-1]**.5;cross=torch.tensor([[a['configuration']!=b['configuration'] for b in metadata] for a in metadata]);near=[]
for i,a in enumerate(metadata):
 for j,b in enumerate(metadata):
  if i==j or (a['group']==b['group'] and abs(a['time']-b['time'])<.2):dist[i,j]=float('inf')
 vals,ids=dist[i].min(0);j=int(ids);target_rms=float((q[i]-q[j]).square().mean().sqrt());near.append(dict(i=i,j=j,input_standardized_rms=float(vals),teacher_q_PD_rms_rad=target_rms,cross_config=a['configuration']!=metadata[j]['configuration']))
report=dict(samples=len(ix),sampling='Every5th50Hzsample; excludesamegroupneighbors<.2s; no fitting',minimum_cross_configuration_input_rms=float(dist[cross].min()),closest_pairs=sorted(near,key=lambda r:r['input_standardized_rms'])[:20],near_pairs_below_0_1=sum(r['input_standardized_rms']<.1 for r in near),near_pairs_with_q_conflict_above_0_02rad=sum(r['input_standardized_rms']<.1 and r['teacher_q_PD_rms_rad']>.02 for r in near),exact_input_pairs_with_q_conflict=sum(r['input_standardized_rms']<1e-6 and r['teacher_q_PD_rms_rad']>.001 for r in near),interpretation='Finiteobservedcoveragecheck only, notglobalidentifiability proof. Zero-force maintainedgrasp and release canremainambiguous. No-contact labels excludeartificialclockreturn. Differentnaturalstates provideproprioinformation; thecriteriondoesnotestablishforcefeaturecausalnecessity.')
(P/'analysis/input-identifiability.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='closest_pairs'},indent=2))
