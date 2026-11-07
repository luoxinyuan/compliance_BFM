"""Issue51 post-run review from saved real qpos, never controller/prediction animation."""
import argparse,json,gzip,hashlib,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
S=ROOT.parent;OLD=S.parent/'native-rolling-h100-20261107'
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(out):
 import numpy as np,torch
 from rolling_analysis import stats,events
 from rolling_review import merge
 names=read(OLD/'runs/qualification-03/action-contract.json')['dof_names'];details=[]
 for folder in ['teacher-coverage-01','bc-eval-01']:
  base=S/'runs'/folder
  for row in read(base/'summaries.json'):
   label=row['label'];path=base/(label+'.json.gz');ep=read(path);s=ep['condition'];stop=s.get('stop_time',s['onset']+s['ramp']);hold=stop+s.get('deceleration',0)
   details.append(dict(label=label,folder=folder,sha256=sha(path),condition=s,boundary=ep.get('boundary'),events=events(ep),phases={key:stats(ep,a,b,names) for key,a,b in [('pre',4,s['onset']),('moving',s['onset'],stop),('braking',stop,hold),('holding',hold,s['release']),('released',s['release'],s['end']),('active',s['onset'],s['release'])]}))
 write(out/'physical-details.json',details)
 # Verify actual paired physical/command prefixes; no inference from task names.
 pairs={};limit=8.1
 for role,a,b,folder in [('Teacher','prefix_continue66','prefix_stop66','teacher-coverage-01'),('P','P_selected_prefix_continue66','P_selected_prefix_stop66','bc-eval-01'),('S','S_selected_prefix_continue66','S_selected_prefix_stop66','bc-eval-01')]:
  e1=read(S/'runs'/folder/(a+'.json.gz'));e2=read(S/'runs'/folder/(b+'.json.gz'))
  p1=[r for r in e1['physics'] if r['force_time']<limit-1e-8];p2=[r for r in e2['physics'] if r['force_time']<limit-1e-8]
  c1=[r for r in e1['control'] if r['time']<limit-1e-8];c2=[r for r in e2['control'] if r['time']<limit-1e-8]
  pairs[role]=dict(physical_equal=p1==p2,control_equal=c1==c2,physical_ticks=[len(p1),len(p2)],control_steps=[len(c1),len(c2)],fork=limit)
 write(out/'common-prefix.json',pairs)
 # Exact restricted-input duplicates; absence of a conflict is not sufficiency proof.
 data=torch.load(S/'runs/data-prepare-01/dataset.pt',map_location='cpu',weights_only=True);seen={};conflicts=[]
 for r in data['rows']:
  if not r['valid']:continue
  k=hashlib.sha256(r['restricted'].numpy().tobytes()).hexdigest()
  for prev in seen.get(k,[]):
   angle=float(torch.rad2deg(torch.acos(((r['z']*prev['z']).sum()/256).clamp(-1,1))))
   if angle>1.:conflicts.append(dict(episode_a=prev['episode'],time_a=prev['time'],episode_b=r['episode'],time_b=r['time'],z_angle_deg=angle,interpretation='same restricted input, different target direction; action consequence not inferred from z angle alone'))
  seen.setdefault(k,[]).append(r)
 write(out/'exact-input-conflicts.json',dict(pairs=conflicts,scope='exact float32 restricted input only; no conflict is not sufficient observability evidence'))
 # Time-aligned divergence is descriptive; policies are in different closed-loop states.
 divergence=[]
 for role in ['P','S']:
  for c in read(ROOT/'research_notes/native_transfer_cases.json'):
   label=c['name'];te=read(S/'runs/teacher-coverage-01'/(label+'.json.gz'));st=read(S/'runs/bc-eval-01'/(role+'_selected_'+label+'.json.gz'))
   a={round(x['time']*50):x for x in te['control']};b={round(x['time']*50):x for x in st['control']};times=sorted(set(a)&set(b));angles=[]
   for k in times:
    z=np.array(a[k]['executed_z']).reshape(-1);w=np.array(b[k]['executed_z']).reshape(-1);angles.append(float(np.degrees(np.arccos(np.clip(z@w/256,-1,1)))))
   first=None
   for i in range(max(0,len(times)-4)):
    if min(angles[i:i+5])>5:first=times[i]/50;break
   divergence.append(dict(role=role,episode=label,first_executed_z_angle_gt5deg_for100ms=first,max_angle_deg=max(angles) if angles else None,interpretation='time matched closed-loop difference, not same-state causal action effect'))
 write(out/'divergence.json',divergence)
 vin=S/'runs/video-input';vin.mkdir(exist_ok=False);paths={}
 def source(folder,label):return merge(S/'runs'/folder,label,vin)
 sel=read(S/'runs/bc-eval-01/selection.json')
 def side(folder,label,title,role,overhead=False):return dict(path=source(folder,label),title=title,role=role,overhead=overhead)
 videos=[]
 for number,label,title in [(1,'short08','8cm request'),(2,'speed022','72cm / avg0.22mps')]:
  videos.append(dict(name=f'0{number}_'+label,grid4=True,cross_condition_description='Same actual physical task and frozen BFM/PD; separate real MPC/I/P/S executions. No predicted trajectories.',sides=[side('teacher-coverage-01',label,'Rolling search Teacher | '+title,'REAL MPC'),side('bc-eval-01','I_'+label,'Frozen I | '+title,'FROZEN I'),side('bc-eval-01','P_selected_'+label,'Privileged BC step'+str(sel['P']['selected']['step']),'PRIVILEGED BC'),side('bc-eval-01','S_selected_'+label,'Ideal-force BC step'+str(sel['S']['selected']['step']),'IDEAL-FORCE BC')]))
 videos.append(dict(name='03_prefix_stop_continue',grid4=True,cross_condition_description='New66cm common prefix, real partner brakes8.1s in stop branch. Top MPC fixed oblique, bottom one selected S checkpoint fixed overhead. Ended panels hold final frame explicitly.',sides=[side('teacher-coverage-01','prefix_continue66','Teacher | partner continues','REAL MPC'),side('teacher-coverage-01','prefix_stop66','Same Teacher | partner stops','REAL MPC'),side('bc-eval-01','S_selected_prefix_continue66','Same S | partner continues','IDEAL-FORCE BC',True),side('bc-eval-01','S_selected_prefix_stop66','Same S | partner stops','IDEAL-FORCE BC',True)]))
 write(vin/'video-pairs.json',videos)
 lines=['# Issue51 actual-qpos videos','','All are 25fps, normal1x simulation speed, saved real qpos only. Predictions and reference animations are excluded. Common0-3s standing warmup is omitted; all contact and full saved recovery remain.','',
 'C=actual contact point (cyan); A=partner anchor(red); yellow=true force; blue=pelvis heading/root path. Released anchor geometry is not a tracking score.','',
 '01_short08: TL MPC, TR frozen I, BL privileged BC, BR ideal-force BC.','02_speed022: same layout;72cm/mean.22m/s.','03_prefix_stop_continue: top MPC continue/stop oblique; bottom one S checkpoint continue/stop overhead. Different-duration panels explicitly hold last saved frame after episode ends.','',
 'P selected model SHA256: '+sel['P']['selected']['sha256'],'S selected model SHA256: '+sel['S']['selected']['sha256'],'',
 'No claim of new PPO, real-time MPC, generalization beyond tested cases, or real G1 force sensing. Full phases and negative outcomes: physical-details.json. Mac copy/decoding still requires verification.']
 (vin/'INDEX.md').write_text('\n'.join(lines)+'\n');write(out/'result.json',dict(status='completed',real_episodes=len(details),videos_prepared=3,Student_PPO=0))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['analyze','render'],required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 if a.mode=='analyze':analyze(a.output)
 else:
  os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),BFM_ROLLING_VIDEO_RUN=str(S/'runs/video-input'),BFM_RENDER_ASSET_ROOT=str(ROOT))
  import rolling_movie;rolling_movie.main()
if __name__=='__main__':main()
