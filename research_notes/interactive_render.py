"""Only previously saved true qpos. Never render a candidate prediction as Student execution."""
import argparse,json,gzip,os,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);merged=a.output/'merged';merged.mkdir();items=[]
 def side(folder,label,title,role):
  path=folder/(label+'.json.gz');ep=read(path);prefix=folder/(label+'-source-prefix.json.gz');pre=read(prefix);ep['physics']=pre['physics']+ep['physics'];ep['render_provenance']=dict(actual_suffix=str(path),suffix_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_prefix=str(prefix),prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest(),no_physics_rerun=True)
  dest=merged/(folder.name+'__'+label+'.json.gz')
  with gzip.open(dest,'wt') as f:json.dump(ep,f)
  return dict(path=str(dest),title=title,role=role)
 D=S/'runs/diagnose-01'
 items.append(dict(name='01-exact-prefix-teacher-recovery',sides=[side(D,'Student_continue_normal72_4.7','S4000 CONTINUES | FAILED REFERENCE','RESTRICTED STUDENT'),side(D,'first_Teacher_then_Student_normal72_4.7','TEACHER 4.7-4.8s ONLY; S4000 ELSEWHERE','TRUE FIRST ACTION INTERVENTION'),side(D,'Teacher_replan_2s_normal72_4.7','TEACHER 4.7-6.7s; S4000 ELSEWHERE','TRUE 2s TAKEOVER; NOT FULL TEACHER')]))
 E=S/'runs/bc-eval-01'
 if (E/'selection.json').exists():
  sel=json.loads((E/'selection.json').read_text());sides=[]
  for case in ['short08','normal72']:
   for arm,title in [('old_data_only','OLD-DATA BC'),('interactive','INTERACTIVE BC')]:
    j=1000;label=arm+'_'+str(j)+'_dev_'+case;selected=sel[arm]['selected']['step']
    sides.append(side(E,label,title+' '+case+' | fixed1000 / selected'+str(selected),'RESTRICTED STUDENT; FIXED ENDPOINT DIAGNOSTIC'))
  items.append(dict(name='02-matched-student-short-and-long',sides=sides,grid4=True,cross_condition_description='Top short08, bottom normal72; each row is paired same condition. Different durations explicitly hold ended panel. No checkpoint splicing: each arm fixed dev-selected checkpoint for both tasks.'))
 else:
  items.append(dict(name='02-short-request-recovery',sides=[side(D,'Student_continue_short08_5.4','S4000 CONTINUES | SHORT08','RESTRICTED STUDENT'),side(D,'Teacher_replan_2s_short08_5.4','TEACHER 5.4-7.4s; S4000 ELSEWHERE','TRUE TAKEOVER; NOT FULL TEACHER')]))
 # Optional fixed8 PPO comparison only if both authorized seeds actually completed.
 paths=[S/'runs/student-ppo5301-01',S/'runs/student-ppo5302-01']
 if all((f/'candidate-8.pt').exists() and (f/'dev8_normal72.json.gz').exists() for f in paths):
  sel=json.loads((E/'selection.json').read_text());j=sel['interactive']['selected']['step'];label='interactive_'+str(j)+'_dev_normal72'
  sides=[side(E,label,'FROZEN INTERACTIVE BC','RESTRICTED IDEAL-FORCE STUDENT')]
  for seed,folder in zip([5301,5302],paths):sides.append(side(folder,'dev8_normal72','STUDENT PPO8 | seed'+str(seed),'FIXED8, NOT FINAL-SELECTED'))
  items.append(dict(name='03-student-own-ppo-fixed8',sides=sides))
 (a.output/'video-pairs.json').write_text(json.dumps(items,indent=2));os.environ.update(BFM_ROLLING_VIDEO_RUN=str(a.output),BFM_RECOVERY_OUTPUT=str(a.output),BFM_RENDER_ASSET_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2')
 import rolling_movie;rolling_movie.main()
 import imageio_ffmpeg
 for row in json.loads((a.output/'manifest.json').read_text()):
  path=Path(row['path']);assert path.stat().st_size>10000;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],stdout=subprocess.DEVNULL,check=True)
 (a.output/'verified.json').write_text(json.dumps(dict(fully_decoded=True,no_new_physics=True,all_sources_actual=True),indent=2))
if __name__=='__main__':main()
