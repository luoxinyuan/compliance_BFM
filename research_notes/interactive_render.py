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
 C=S/'runs/repair-short-01'
 items.append(dict(name='02-corrected-postcontact-recovery',sides=[side(C,'Student_continue_correct','S4000 CONTINUES | SHORT08','UNINTERRUPTED SOURCE MATCHED EXACTLY'),side(C,'Teacher_replan_correct_1p4s','TEACHER 5.4-6.8s; S4000 ELSEWHERE','CORRECTED CALLBACK/STATE RESTORE')]))
 E=S/'runs/bc-eval-01';sides=[]
 for case in ['short08','normal72']:
  sides.append(side(E,'S4000_dev_'+case,'S4000 INITIAL | '+case,'RESTRICTED IDEAL-FORCE STUDENT'))
  sides.append(side(E,'old_data_only_500_dev_'+case,'OLD-DATA BC +500 | '+case,'VALID DEV ONLY; NO NEW LABELS / NO RL'))
 items.append(dict(name='03-valid-old-data-bc-development',sides=sides,grid4=True,cross_condition_description='Top short08, bottom normal72; each row paired same condition, both tasks same frozen old-data-only500 checkpoint. Development evidence only; no heldout/general claim. Mixed affected model is not displayed as success.'))
 (a.output/'video-pairs.json').write_text(json.dumps(items,indent=2));os.environ.update(BFM_ROLLING_VIDEO_RUN=str(a.output),BFM_RECOVERY_OUTPUT=str(a.output),BFM_RENDER_ASSET_ROOT=str(ROOT),CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2')
 import rolling_movie;rolling_movie.main()
 import imageio_ffmpeg
 for row in json.loads((a.output/'manifest.json').read_text()):
  path=Path(row['path']);assert path.stat().st_size>10000;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],stdout=subprocess.DEVNULL,check=True)
 (a.output/'verified.json').write_text(json.dumps(dict(fully_decoded=True,no_new_physics=True,all_sources_actual=True),indent=2))
if __name__=='__main__':main()
