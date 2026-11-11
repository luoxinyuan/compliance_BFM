"""Serial GPU rendering of existing executed qpos only. No dynamics sampling."""
import argparse,fcntl,gzip,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
START=time.monotonic()
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args();S=a.stage;P=S.parents[1];R=S/'runtime-l2';OUT=S/'runs/video-01';OUT.mkdir(exist_ok=False)
lock=(P/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
job=dict(layer='L2',status='running',pid=os.getpid(),started=time.time(),runtime=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),physics_steps=0,training=False)
side=S/'jobs/video-render-01.json'
try:
 assert json.loads((S/'runs'/a.run/'complete.json').read_text())['status']=='completed'
 assert shutil.disk_usage(S).free>100*(1<<30)
 busy=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not busy,('GPU0 occupied',busy)
 side.write_text(json.dumps(job,indent=2))
 import torch
 manifest=json.loads((P/'experiments/f1-l0-array-audit-20261111/runs/l0-complete-02/state-manifest.json').read_text());specs={}
 for r in manifest:
  if r.get('role')!='source_positive':specs[r['state_id']]=torch.load(P/'experiments/f1-l0-array-audit-20261111'/r['state_snapshot'],map_location='cpu',weights_only=False)['spec']
 sys.path.insert(0,str(R/'research_notes'))
 from frl_v2_task import continuation
 videos=[]
 pairs=[('f1-p1-13-feedback',[('P1-13','CS','BC500 fixed ideal-force policy'),('P1-13','CP','Causal privileged replanning')]),('f1-stop-continue',[('new_stop-sustained-5f6380cdec75','CP','STOP: causal privileged replanning'),('new_stop-sustained-5f6380cdec75-continue','CP','CONTINUE: same causal planner')]),('f1-tail-hold-release',[('new_tail-hold-97ff10c4dcd9','CS','BC500 fixed ideal-force policy'),('new_tail-hold-97ff10c4dcd9','CP','Causal privileged replanning')])]
 for name,entries in pairs:
  sides=[]
  for state,mode,title in entries:
   source=S/'runs'/a.run/(state+'-L2-'+mode+'.pt');x=torch.load(source,map_location='cpu',weights_only=False)
   assert x['identity'] in ('fixed_restricted_execution','real_causal_feedback_execution')
   spec=specs[state.removesuffix('-continue')].copy()
   if state.endswith('-continue'):spec=continuation(spec)
   dst=OUT/(state+'-'+mode+'.json.gz');payload=dict(physics=x['physics'],condition=spec,anchor=x['physics'][0]['anchor'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),diagnostic_window_only=True)
   with gzip.open(dst,'wt') as f:json.dump(payload,f)
   sides.append(dict(path=str(dst),title=title+' | actual 3s window',role='ACTUAL EXECUTION; offline compute'))
  item=dict(name=name,sides=sides)
  if name=='f1-stop-continue':item['cross_condition_description']='Matched common prefix, then stop versus continue. One related parent, same planner; not two independent conditions.'
  videos.append(item)
 (OUT/'video-pairs.json').write_text(json.dumps(videos,indent=2))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BFM_ROLLING_VIDEO_RUN=str(OUT),BFM_RECOVERY_OUTPUT=str(OUT),BFM_RENDER_ASSET_ROOT=str(R))
 subprocess.run([sys.executable,str(R/'research_notes/rolling_movie.py')],env=env,check=True,timeout=600)
 ffmpeg=shutil.which('ffmpeg')
 if ffmpeg is None:
  import imageio_ffmpeg
  ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 for path in OUT.glob('*.mp4'):subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True,timeout=60,stdout=subprocess.DEVNULL)
 job.update(status='completed',videos=len(videos),decode_all_frames=True)
except BaseException as exc:
 job.update(status='failed_preserved',error=repr(exc));raise
finally:
 job['elapsed_s']=time.monotonic()-START;side.write_text(json.dumps(job,indent=2));fcntl.flock(lock,fcntl.LOCK_UN)
