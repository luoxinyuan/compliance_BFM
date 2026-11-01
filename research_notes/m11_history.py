"""Vectorized existing contact-history features, unchanged sample provenance/coordinates."""
import numpy as np
from m9_history import ContactHistory
class FastHistory(ContactHistory):
    def features(self,current_world_R_body):
        R=np.asarray(current_world_R_body).reshape(3,3);a=np.zeros((self.length,14));rows=list(self.rows)
        if not rows:return a
        f=np.asarray([r['force_world'] for r in rows]);valid=np.asarray([r['measurement_valid'] for r in rows]);past=np.asarray([r['world_R_body'] for r in rows])
        rel=np.matmul(R.T,past);out=a[-len(rows):]
        out[:,:3]=np.matmul(R.T,f[:,:,None])[:,:,0]/5*valid[:,None]
        out[:,3:5]=[[r['link']=='right_wrist_yaw_link',r['link']=='torso_link'] for r in rows]
        out[:,5:8]=[r['local_point'] for r in rows];out[:,8]=valid;out[:,9]=[r['contact_detected'] for r in rows];out[:,10:]=rel[:,:2,:2].reshape(-1,4)
        return a
