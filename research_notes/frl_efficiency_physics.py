"""Exposed-only physical parity and sampling profile; never trains or reads sealed data."""
import argparse,cProfile,contextlib,gzip,hashlib,json,os,pstats,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc

def study(e,m,contract,out):
 import torch
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from frl_v2_task import configure,manifests
 import frl_v2_high
 from frl_fast_archive import installed
 install();speed(e);torch.set_num_threads(1);torch.set_grad_enabled(False)
 (out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');configure()
 allgroups=manifests();selected=[allgroups['dev'][i] for i in (0,8,16)]
 assert all(x['family'] in ('near','mid','stop') for x in selected)
 frl_v2_high.parent_manifests=lambda:{'dev':selected}
 profile=cProfile.Profile();started=time.perf_counter()
 cm=installed(1) if ARGS.archive_mode=='fast' else contextlib.nullcontext()
 if ARGS.profile:profile.enable()
 with cm:result=frl_v2_high.evaluate(e,m,contract,out,ARGS)
 if ARGS.profile:
  profile.disable();profile.dump_stats(str(out/'sampling.pstats'))
  stats=pstats.Stats(profile)
  top=sorted(stats.stats.items(),key=lambda x:x[1][2],reverse=True)[:60]
  (out/'profile-self.json').write_text(json.dumps([dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_s=v[2],inclusive_s=v[3]) for k,v in top],indent=2))
 elapsed=time.perf_counter()-started
 counts=dict(high_level_decisions=0,control_50Hz=0,physics_200Hz=0,source_physics_200Hz=0)
 assets={}
 for path in sorted(out.glob('*.json.gz')):
  content=gzip.decompress(path.read_bytes());obj=json.loads(content)
  assets[path.name]=dict(uncompressed_sha256=hashlib.sha256(content).hexdigest(),compressed_bytes=path.stat().st_size)
  if path.name.endswith('-source-prefix.json.gz'):counts['source_physics_200Hz']+=len(obj['physics'])
  else:
   counts['high_level_decisions']+=len(obj.get('decisions',[]));counts['control_50Hz']+=len(obj['control']);counts['physics_200Hz']+=len(obj['physics'])
 micro=[]
 if ARGS.profile:
  # Inference-only scaling is reported separately from actual MuJoCo throughput.
  # No claim that this establishes a vectorized contact environment.
  ob={k:v.detach().clone() for k,v in e.obs.items()};z=e.standing.detach().clone();reference=m.act(ob,z,mean=True).detach()
  for size in (1,4,8):
   try:
    obs={k:v.repeat((size,)+(1,)*(v.ndim-1)) for k,v in ob.items()};zz=z.repeat(size,1)
    for _ in range(3):a=m.act(obs,zz,mean=True)
    torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(30):a=m.act(obs,zz,mean=True)
    torch.cuda.synchronize();wall=time.perf_counter()-t
    micro.append(dict(batch=size,calls=30,wall_s=wall,rows_per_s=30*size/wall,maximum_action_difference=float((a-reference).abs().max()),FP32=True,physics_steps=0))
   except Exception as exc:micro.append(dict(batch=size,status='unsupported',error=repr(exc),physics_steps=0))
  (out/'inference-only-scaling.json').write_text(json.dumps(micro,indent=2))
 result.update(archive_mode=ARGS.archive_mode,profiled=ARGS.profile,evaluation_wall_s=elapsed,counts=counts,archive_assets=assets,
  sampled_training_decisions=0,actor_updates=0,FB_updates=0,sealed_evaluations=0,reference_checkpoint_sha256=ARGS.sha256)
 (out/'efficiency-result.json').write_text(json.dumps(result,indent=2));return result

def main():
 global ARGS
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive-mode',choices=['legacy','fast'],required=True);p.add_argument('--profile',action='store_true')
 ARGS=p.parse_args();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
 ARGS.checkpoint=STAGE/'runs/frl-formal-A-01/candidate-0.pt';ARGS.sha256='112b3da596960e7064ba3a31f55f7c9da976101adcd866bedb15038501bbff1f'
 assert hashlib.sha256(ARGS.checkpoint.read_bytes()).hexdigest()==ARGS.sha256
 ARGS.arm='A';ARGS.split='dev';ARGS.iteration=0;ARGS.smoke=False;ARGS.record=True
 ARGS.runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 ARGS.output.mkdir(parents=True,exist_ok=False)
 (ARGS.output/'launch.json').write_text(json.dumps(dict(runtime=ARGS.runtime,pid=os.getpid(),started=time.time(),archive_mode=ARGS.archive_mode,profiled=ARGS.profile,training=False),indent=2))
 os.environ.update(CLEAN_KIND='direct',P4000_CANONICAL_ROW_ACTOR='1',BFM_RECORD_NOMINAL_DIAGNOSTIC='0',BFM_RECOVERY_OUTPUT=str(ARGS.output))
 os.environ.setdefault('TMPDIR',str(STAGE/'tmp'));os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(STAGE/'torch-cache'))
 os.chdir(ARGS.output)
 import m10_stage,m10_physics
 m10_physics.OUT=ARGS.output;m10_stage.run_study=study
 m10_physics.main()
if __name__=='__main__':main()
