"""Saved control records: reference context and teacher clipping; no models or physics."""
import gzip,json
from pathlib import Path
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m2-unseen-body-20261031A')
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 out=[];o=R/'comparison-attempt-01/outputs';e=read(o/'midpoint__teacher.json.gz');contract=read(o/'action-contract.json');rows=[c for c in e['control'] if 12.4-1e-8<=c['time']<17.4-1e-8]
 for i,n in enumerate(contract['dof_names']):
  hits=[c for c in rows if abs(c['action'][i]*5)>=contract['clip']]
  if hits:out.append(dict(joint=n,control_frames=len(hits),total_contact_frames=len(rows),first_s=hits[0]['time'],last_s=hits[-1]['time'],base_also_clipped=sum(abs(c['base_action'][i]*5)>=contract['clip'] for c in hits),action_peak=max(abs(c['action'][i]) for c in hits),clip=contract['clip']))
 refs=[]
 for config in ['standing','raised']:
  ep=read(R/'reference-attempt-01/outputs'/f'{config}__distill.json.gz');cs=ep['control'];rr=[]
  for t in [10.4,12.4,19.4,21.4,23.3]:
   c=min(cs,key=lambda c:abs(c['time']-t));rr.append(dict(time=c['time'],native_coords=c['native_coords'],pre_wrist=c['pre_wrist'],force_current_xyz=c['measured_force_xyz'],force_history_max_abs_normalized=max(abs(x) for x in c['student_input'][465:])))
  refs.append(dict(configuration=config,records=rr))
 result=dict(teacher_contact_action_clipping=out,known_body_reference=refs,scope='No model forward or fitting; descriptive saved controls. Clipping is not independently proven to cause teacher task error.')
 (R/'analysis/extra-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
