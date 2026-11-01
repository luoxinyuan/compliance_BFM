"""Real saved-sensor contract test, NOT a new physical contact-switch episode."""
import gzip,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from m9_history import ContactHistory
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
RUN=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A'
OLD=P/'bridge-v2-owned/runs/m8-contact-body-20261101A/final-attempt-01/outputs'
def run():
 h=ContactHistory(320);sources=[];segments=[]
 for name,link,point in [('wrist_early__distill','right_wrist_yaw_link',[0,0,0]),('down_test__distill','torso_link',[0,0,.25])]:
  path=OLD/(name+'.json.gz')
  with gzip.open(path,'rt') as f:e=json.load(f)
  sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  onset=e['condition']['onset'];cs=[c for c in e['control'] if onset+.4<=c['time']<onset+3.6][:160];assert len(cs)>=150
  for c in cs:
   q=np.asarray(c['pre_qpos'][3:7]);R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();h.append(c['measured_force_xyz'],link,point,R)
  segments.append(len(cs))
 x=h.features(R);a=sum(r['link']=='right_wrist_yaw_link' for r in h.rows);b=sum(r['link']=='torso_link' for r in h.rows);assert a>0 and b>0
 for row,feature in zip(h.rows,x[-len(h.rows):]):
  assert feature[3]==(row['link']=='right_wrist_yaw_link') and feature[4]==(row['link']=='torso_link')
  np.testing.assert_allclose(feature[:3]*5,R.T@row['force_world'])
 G=Rotation.from_euler('z',73,degrees=True).as_matrix();rot=ContactHistory(320)
 for r in h.rows:rot.append(G@r['force_world'],r['link'],r['local_point'],G@r['world_R_body'],r['measurement_valid'])
 err=float(abs(x-rot.features(G@R)).max());assert err<1e-12
 np.savez_compressed(RUN/'history-real-sensor-replay.npz',features=x)
 report=dict(status='passed',scope='Concatenated real sensor snippets for history contract only, NOT physical same-episode switching evidence and NOT retrained M8 input',sources=sources,segment_samples=segments,wrist_samples_preserved=a,torso_samples_preserved=b,global_heading_rotation_max_error=err,feature_shape=list(x.shape),past_samples_relabelled=0,anchor_or_task_id_inputs=False)
 (RUN/'history-real-sensor-replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
