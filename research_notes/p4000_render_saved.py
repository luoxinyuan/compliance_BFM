"""Two complete saved-qpos comparisons; no simulation or policy rerun."""
import argparse,gzip,hashlib,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 root=Path(__file__).resolve().parents[1];stage=root.parent;merged=a.output/'merged';merged.mkdir();items=[];identities={}
 models=[('initial',stage/'runs/single5201-independent-32/dev-0000','P4000 INIT'),('joint32',stage/'runs/single5201-joint-32/dev-0032','PPO32 / JOINT CRITIC'),('independent32',stage/'runs/single5201-independent-32/dev-0032','PPO32 / INDEPENDENT CRITIC')]
 for number,case in [(1,'normal72'),(2,'stop66')]:
  sides=[]
  for name,folder,title in models:
   source=folder/f'{case}__mean.json.gz';prefix=folder/f'{case}__mean-source-prefix.json.gz'
   with gzip.open(source,'rt') as f:episode=json.load(f)
   with gzip.open(prefix,'rt') as f:pre=json.load(f)
   episode['physics']=pre['physics']+episode['physics'];episode['render_provenance']=dict(suffix=str(source),suffix_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prefix=str(prefix),prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest(),no_physics_rerun=True)
   dest=merged/f'{name}-{case}.json.gz'
   with gzip.open(dest,'wt') as f:json.dump(episode,f)
   sides.append(dict(path=str(dest),title=title+' | '+case,role='DETERMINISTIC PRIVILEGED EXPERT'))
  items.append(dict(name=f'{number:02d}-{case}-fixed32-comparison',sides=sides))
 (a.output/'video-pairs.json').write_text(json.dumps(items,indent=2))
 os.environ.update(BFM_ROLLING_VIDEO_RUN=str(a.output),BFM_RECOVERY_OUTPUT=str(a.output),BFM_RENDER_ASSET_ROOT=str(root),CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2')
 import rolling_movie
 rolling_movie.main()
 import imageio_ffmpeg
 ff=imageio_ffmpeg.get_ffmpeg_exe();manifest=json.loads((a.output/'manifest.json').read_text())
 for entry in manifest:
  path=Path(entry['path']);assert path.stat().st_size>10000
  subprocess.run([ff,'-v','error','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
 write=dict(status='completed',no_physics_rerun=True,fully_decoded=True,videos=manifest,role='privileged Expert initialization and two fixed32 PPO methods; no Student')
 (a.output/'result.json').write_text(json.dumps(write,indent=2))
if __name__=='__main__':main()
