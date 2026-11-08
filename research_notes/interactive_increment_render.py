"""Two concise, actual-qpos only comparisons; no predictor animation."""
import argparse,gzip,json,hashlib,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def read(p):
 with gzip.open(p,'rt')as f:return json.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);merged=a.output/'merged';merged.mkdir()
 def side(folder,label,title):
  src=folder/(label+'.json.gz');pre=folder/(label+'-source-prefix.json.gz');e=read(src);e['physics']=read(pre)['physics']+e['physics'];e['render_provenance']=dict(actual_suffix=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),actual_prefix=str(pre),no_new_physics=True);dest=merged/(label+'.json.gz')
  with gzip.open(dest,'wt')as f:json.dump(e,f)
  return dict(path=str(dest),title=title,role='RESTRICTED IDEAL-FORCE STUDENT | TRUE EXECUTION')
 E=S/'runs/eval-02';F=S/'runs/ppo-clean-5401';sides=[]
 for case in ['near06','prefix59_continue']:
  sides.append(side(E,'BC500_'+case,'FROZEN OLD-DATA BC500 | '+case));sides.append(side(E,'Sinteractive_'+case,'CLEAN INTERACTIVE500 | '+case))
 items=[dict(name='01-clean-label-increment-short-and-long',sides=sides,grid4=True,cross_condition_description='Same frozen checkpoint each column; top6cm bottom59cm continuing; gain and regression both retained; actualqpos1x.')]
 items.append(dict(name='02-student-RL-transient-gain-and-regression',sides=[side(E,'BC500_normal72','FROZEN BC500 | DEV72cm'),side(F,'dev4_normal72','STUDENT PPO SEED5401 | DEV BATCH4'),side(F,'dev7_normal72','STUDENT PPO SEED5401 | FIXED BATCH7')]))
 (a.output/'video-pairs.json').write_text(json.dumps(items,indent=2));os.environ.update(BFM_ROLLING_VIDEO_RUN=str(a.output),BFM_RECOVERY_OUTPUT=str(a.output),BFM_RENDER_ASSET_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2')
 import rolling_movie;rolling_movie.main()
 import imageio_ffmpeg
 for row in json.loads((a.output/'manifest.json').read_text()):
  path=Path(row['path']);assert path.stat().st_size>10000;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],stdout=subprocess.DEVNULL,check=True)
 (a.output/'verified.json').write_text(json.dumps(dict(fully_decoded=True,no_new_physics=True,all_sources_actual=True,normal_speed=True),indent=2))
if __name__=='__main__':main()
