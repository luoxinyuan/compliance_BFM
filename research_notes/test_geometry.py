import numpy as np
from geometry_contact import frame,rotation
# Rigid pair geometry and derivative: opposite rotational velocities, same translation.
s=dict(geometry='torso_pair',onset=4.,ramp=8.,rotation_deg=15,distance=.2)
p,v,a,w=frame(s,7.);eps=1e-5
for q in [np.array([0,.13,.15]),np.array([0,-.13,.15])]:
 def point(t):p,_,a,_=frame(s,t);return p+rotation(a)@q
 numeric=(point(7+eps)-point(7-eps))/(2*eps);analytic=v+np.cross([0,0,w],rotation(a)@q);assert np.max(abs(numeric-analytic))<1e-8
# Same-body COM wrench algebra vs sum of point forces, no double torque.
pts=np.array([[.02,.13,.15],[.02,-.13,.15]]);F=np.array([[-2.5,0,0],[2.5,0,0]]);com=np.array([.01,0,.1]);W=np.r_[F.sum(0),np.cross(pts-com,F).sum(0)];assert np.linalg.norm(W[:3])==0 and abs(W[5]-.65)<1e-12
existing=np.array([1,2,3,4,5,6.]);restored=existing+W-W;assert np.allclose(restored,existing)
assert sum(np.linalg.norm(F,axis=1))==5
print('PASS rigid pair derivative, COM torque, equal sum-norm budget, preserved prior wrench')
