"""Issue51 compact science summaries and final actual-qpos video inputs; CPU only."""
import argparse,json,gzip,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
S=ROOT.parent
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 from rolling_analysis import stats,events
 from rolling_review import merge
 import numpy as np
 names=read(S/'runs/teacher-coverage-01/action-contract.json')['dof_names'];allrows=[]
 for folder in ['teacher-coverage-01','bc-eval-01','diagnostic-01']:
  for row in read(S/'runs'/folder/'summaries.json'):
   ep=read(S/'runs'/folder/(row['label']+'.json.gz'));s=ep['condition'];stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0)
   allrows.append(dict(label=row['label'],run=folder,boundary=ep['boundary'],condition=s,events=events(ep),phases={k:stats(ep,lo,hi,names) for k,lo,hi in [('pre',4,s['onset']),('moving',s['onset'],stop),('braking',stop,hold),('holding',hold,s['release']),('released',s['release'],s['end']),('active',s['onset'],s['release'])]}))
 write(a.output/'all-physical-details.json',allrows)
 align=read(S/'runs/diagnostic-01/main4000-train-alignment.json');groups={}
 for role in ['P','S']:
  for kind in sorted(set(r['source_kind'] for r in align)):
   rr=[r for r in align if r['role']==role and r['source_kind']==kind]
   groups[role+':'+kind]=dict(frames=len(rr),angle_mean=float(np.mean([r['z_angle_deg'] for r in rr])),angle_max=max(r['z_angle_deg'] for r in rr),qpd_mean_deg=float(np.mean([r['qpd_mean_deg'] for r in rr])),qpd_max_deg=max(r['qpd_max_deg'] for r in rr),worst=sorted(rr,key=lambda r:r['qpd_max_deg'],reverse=True)[:3])
 write(a.output/'alignment-by-source.json',groups)
 # Formal selection is fixed. Videos show S4000 as an explicitly non-selected failure.
 sel=read(S/'runs/bc-eval-01/selection.json');vin=S/'runs/video-final-input';vin.mkdir(exist_ok=False)
 def side(folder,label,title,role,overhead=False):return dict(path=merge(S/'runs'/folder,label,vin),title=title,role=role,overhead=overhead)
 videos=[]
 videos.append(dict(name='01_short08_real_BC',grid4=True,cross_condition_description='Same8cm task; main S4000 is a frozen diagnostic, formal dev selection remains new S-init, not old historical S0.',sides=[side('teacher-coverage-01','short08','Search Teacher | 8cm request','REAL MPC TEACHER'),side('bc-eval-01','I_short08','Frozen old I | 8cm request','FROZEN I'),side('bc-eval-01','P_selected_short08','P4000 | privileged BC','PRIVILEGED BC'),side('diagnostic-01','S4000_short08','S4000 | non-selected diagnostic','IDEAL-FORCE BC DIAGNOSTIC')]))
 videos.append(dict(name='02_speed022_real_BC',grid4=True,cross_condition_description='Same72cm/avg.22 task; fixed main4000-step P and S shown. S4000 is not promoted as selected baseline.',sides=[side('teacher-coverage-01','speed022','Search Teacher | avg.22m/s','REAL MPC TEACHER'),side('bc-eval-01','I_speed022','Frozen old I | avg.22m/s','FROZEN I'),side('bc-eval-01','P_selected_speed022','P4000 | privileged BC','PRIVILEGED BC'),side('bc-eval-01','S_dev_4000','S4000 | held fixed failure','IDEAL-FORCE BC DIAGNOSTIC')]))
 videos.append(dict(name='03_new_prefix_Teacher_P',grid4=True,cross_condition_description='New66cm continue/stop. Top identical MPC program fixed oblique; bottom one P4000 checkpoint privileged BC fixed overhead. Not a restricted Student success.',sides=[side('teacher-coverage-01','prefix_continue66','Search Teacher | continue','REAL MPC TEACHER'),side('teacher-coverage-01','prefix_stop66','Same search Teacher | stop8.1s','REAL MPC TEACHER'),side('bc-eval-01','P_selected_prefix_continue66','Same P4000 | continue (overhead)','PRIVILEGED BC',True),side('bc-eval-01','P_selected_prefix_stop66','Same P4000 | stop (overhead)','PRIVILEGED BC',True)]))
 write(vin/'video-pairs.json',videos);write(a.output/'result.json',dict(status='completed',actual_episodes=len(allrows),videos_prepared=3,formal_selection_unchanged=True,Student_PPO=0))
if __name__=='__main__':main()
