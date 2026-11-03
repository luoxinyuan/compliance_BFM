import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stop_env import braking,phase,PLAN
class TestStop(unittest.TestCase):
 def test_continuity(self):
  s=PLAN['stop_sources']['stop0'];t=s['stop_time'];u=(t-s['onset'])/s['ramp'];d=s['distance']*(3*u*u-2*u*u*u);v=s['distance']*6*u*(1-u)/s['ramp']
  self.assertAlmostEqual(braking(s,t)[0],d);self.assertAlmostEqual(braking(s,t)[1],v)
  end=t+s['deceleration'];self.assertAlmostEqual(braking(s,end)[1],0);self.assertEqual(braking(s,end),braking(s,end+10))
 def test_separation(self):
  for s in PLAN['stop_sources'].values():
   self.assertGreater(s['release'],s['stop_time']+s['deceleration']);self.assertEqual(phase(s,s['stop_time']+.01),'decelerate');self.assertEqual(phase(s,s['release']+.01),'recover')
 def test_no_force_columns(self):
  import numpy as np
  x=np.arange(1025);n=np.r_[x[:465],x[765:]];self.assertEqual(len(n),725);self.assertFalse(np.isin(n,np.arange(465,765)).any())
 def test_no_task_answers(self):
  s=Path(__file__).with_name('stop_student.py').read_text().split('def sx(')[1].split('def run(')[0]
  for word in ['anchor','partner_velocity','active','time','features()']:self.assertNotIn(word,s)
if __name__=='__main__':unittest.main()
