"""Reuse the existing real-state renderer for matched M11 continuation videos."""
import os,json
from pathlib import Path
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-continuation-20261102')
os.environ['M11_VIDEO_RUN']=str(R)
from m11_render import main
if __name__=='__main__':
 pairs=[]
 for task in ['sealed_straight','sealed_left','sealed_local']:
  sides=[]
  for arm,title in [('prior','M10 deployed prior'),('fixed','M11 fixed exploration training'),('annealed','M11 gradual exploration reduction')]:
   sides.append(dict(path=str(R/('final-'+arm+'-attempt-01/outputs/job/evaluation')/(task+'__mean.json.gz')),title=title+' | '+task+' | deterministic execution'))
  pairs.append(dict(name=task,sides=sides))
 (R/'video-pairs.json').write_text(json.dumps(pairs,indent=2));main()
