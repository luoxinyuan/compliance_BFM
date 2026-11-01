"""Causal per-sample contact provenance; no anchor or task labels.

World force is a simulator measurement; a robot needs calibrated force sensing
and orientation/odometry to form the equivalent coordinates. Historical link and
local point are immutable. Timestamps are for audits, never network features.
"""
from collections import deque
import numpy as np
class ContactHistory:
 def __init__(self,length=320):self.length=length;self.rows=deque(maxlen=length)
 def append(self,force_world,link,local_point,world_R_body,valid=True):
  f=np.asarray(force_world,dtype=float);R=np.asarray(world_R_body,dtype=float).reshape(3,3)
  if not (np.isfinite(f).all() and np.isfinite(R).all()):raise ValueError('nonfinite sensor')
  detected=bool(valid and np.linalg.norm(f)>1e-8)
  row=dict(force_world=f.tolist(),link=str(link) if detected else None,local_point=np.asarray(local_point).tolist() if detected else [0.,0.,0.],world_R_body=R.tolist(),frame='world',measurement_valid=bool(valid),contact_detected=detected)
  self.rows.append(row);return row
 def features(self,current_world_R_body):
  R=np.asarray(current_world_R_body).reshape(3,3);rows=[]
  for r in self.rows:
   # All old forces represented in the SAME current body frame; historical
   # orientation columns separately retain their relationship to current frame.
   force=R.T@np.asarray(r['force_world']);rel=R.T@np.asarray(r['world_R_body']);link=[r['link']=='right_wrist_yaw_link',r['link']=='torso_link']
   rows.append(np.r_[force/5,link,r['local_point'],r['measurement_valid'],r['contact_detected'],rel[:2,:2].ravel()])
  a=np.zeros((self.length,14));
  if rows:a[-len(rows):]=rows
  return a
