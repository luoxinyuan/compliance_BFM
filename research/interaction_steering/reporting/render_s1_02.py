import os,sys,time,json,gzip,subprocess,hashlib,fcntl,shutil
from pathlib import Path
start=time.monotonic();S=Path(sys.argv[1]);R=S/'runtime';P=S.parents[1];run=S/'runs/physical-01';out=S/'videos02';out.mkdir(exist_ok=False)
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(P).free>=100*(1<<30)
job=dict(status='running',pid=os.getpid(),started=time.time(),physics_steps=0,training=False)
side=S/'jobs/video-02.json';side.write_text(json.dumps(job,indent=2))
try:
 sys.path.insert(0,str(R))
 import torch
 stop='new_stop-sustained-5f6380cdec75'
 configurations=[('s1-sustained-negative',[('P1-13',k,k+' | matched sustained demand') for k in ['BC500','CS','CP']]),('s1-stop-response',[(stop,k,k+' | stop, hold and release') for k in ['BC500','CS','CP']]),('s1-cs-stop-continue',[(stop,'CS','CS | STOP'),(stop+'-continue','CS','CS | CONTINUE')])]
 items=[]
 for name,entries in configurations:
  sides=[]
  for sid,kind,title in entries:
   path=run/(sid+'-'+kind+'-0.pt');x=torch.load(path,map_location='cpu',weights_only=False);assert x['identity']=='actual_controller_execution'
   dst=out/(sid+'-'+kind+'.json.gz')
   if not dst.exists():
    with gzip.open(dst,'wt') as f:json.dump(dict(physics=x['physics'],condition=x['spec'],anchor=x['physics'][0]['anchor'],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),f)
   sides.append(dict(path=str(dst),title=title,role='ACTUAL EXECUTION; frozen controller'))
  item=dict(name=name,sides=sides)
  if name.endswith('stop-continue'):item['cross_condition_description']='One parent, common physical prefix then actual stop/continue. Shorter side explicitly holds final frame after its recorded end.'
  items.append(item)
 (out/'video-pairs.json').write_text(json.dumps(items,indent=2))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BFM_ROLLING_VIDEO_RUN=str(out),BFM_RECOVERY_OUTPUT=str(out),BFM_RENDER_ASSET_ROOT=str(R))
 subprocess.run([sys.executable,str(R/'research_notes/rolling_movie.py')],env=env,check=True,timeout=600)
 import imageio_ffmpeg
 ff=shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
 for p in out.glob('*.mp4'):subprocess.run([ff,'-v','error','-i',str(p),'-f','null','-'],check=True,timeout=90,stdout=subprocess.DEVNULL)
 job.update(status='completed',decode_all_frames=True,videos=3)
except BaseException as exc:job.update(status='failed_preserved',error=repr(exc));raise
finally:job['job_s']=time.monotonic()-start;side.write_text(json.dumps(job,indent=2))
