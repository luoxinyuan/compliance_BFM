"""Physical partner path continuity and no hidden activation as policy input."""
import numpy as np
from heading_env import pose

def test_path():
 base=dict(onset=4.25,ramp=7.8,distance=.82,curvature=0,change_time=8.,after_curvature=.35)
 for k in [-.35,.35]:
  s=dict(base,after_curvature=k)
  for t in [4.25,7.,8.,9.,12.05,14.]:
   p,v=pose(s,t);a,_=pose(s,t-1e-5);b,_=pose(s,t+1e-5)
   assert np.linalg.norm((b-a)/2e-5-v)<1e-5
  a,v=pose(s,8-1e-7);b,w=pose(s,8+1e-7);assert np.linalg.norm(a-b)<1e-6 and np.linalg.norm(v-w)<1e-6
 for t in [3.,4.25,7.,8.]:
  a,v=pose(base,t);b,w=pose(dict(base,after_curvature=-.35),t);assert np.array_equal(a,b) and np.array_equal(v,w)
 assert pose(base,10)[0][1]>0 and pose(dict(base,after_curvature=-.35),10)[0][1]<0
 print('PASS analytic velocity, continuous branch, identical prefix, real opposite path')
if __name__=='__main__':test_path()
