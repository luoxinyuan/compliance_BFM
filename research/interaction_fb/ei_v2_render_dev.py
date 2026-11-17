"""Render three fixed actual Dev comparisons, no new physics or policy execution."""
import argparse,bisect,fcntl,gzip,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,type=Path);a=parser.parse_args()
    S=a.stage.resolve();P=S.parents[1];R=S/'eval-runtime';M=Path(__file__).resolve().parents[2];out=S/'runs/dev-videos-01';side=S/'jobs/dev-videos-01.json'
    assert not out.exists() and not side.exists();out.mkdir();merged=out/'merged';merged.mkdir()
    start=time.monotonic();locks=[];job=dict(name='dev-videos-01',status='preflight',pid=os.getpid(),start_epoch=time.time(),physics=dict(actual=0,reserved=0),training_updates=0,no_new_physics=True,source_sha256=sha(__file__),renderer_sha256=sha(Path(__file__).with_name('ei_v2_movie.py')));charged=False
    try:
        for path in (P/'.dual-track-gpu0.lock',P/'gpu0-experiment.lock'):
            f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-3ae548cd-2131-2b5d-42ad-5d0f40ff9da0'
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(x.split(',')[0].strip()==uuid for x in active.splitlines())
        assert int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True))==0
        assert shutil.disk_usage(S).free>6*2**30
        ledger=json.loads((S/'ledger.json').read_text());assert ledger['charged_s']+1800<56000
        job.update(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip(),asset_runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),gpu_uuid=uuid)
        side.write_text(json.dumps(job,indent=2)+'\n')
        sys.path[:0]=[str(R),str(R/'research_notes')]
        from research.interaction_fb.journal import read
        rows=json.loads((S/'runs/formal-dev-02/evaluation.json').read_text());actual=S/'runs/formal-dev-02';exported={};provenance=[]
        labels={0:'standing',1:'arm5',2:'arm15',3:'walking proposal'}
        def entry(ix,title):
            row=rows[ix];assert row['complete'] and row['hard_boundary'] is None
            if ix not in exported:
                path=actual/f'case-{ix:03d}-actual.ptz';x=read(path,row['archive']);assert x['role']=='actual_frozen_policy_independent_dev_not_training'
                prefix=x['source']['physics'];physical=prefix+x['physics'];assert physical
                assert prefix[-1]['time']<x['physics'][0]['time'];assert max(abs((b['time']-aa['time'])-.005)for aa,b in zip(physical,physical[1:]))<1e-6
                ins=x['instructions'];ts=[v['time']for v in ins]
                for v in physical:
                    j=bisect.bisect_right(ts,v.get('force_time',v['time'])+1e-8)-1
                    if j<0:v.update(video_goal='SOURCE PREFIX',video_command=x['spec']['source'])
                    else:v.update(video_goal='FOLLOW'if ins[j]['goal_mode']==0 else'HOLD',video_command=labels[ins[j]['option']])
                cond=dict(x['spec'],handoff_time=x['spec']['source_until']);dest=merged/f'actual-{ix:03d}.json.gz'
                payload=dict(physics=physical,condition=cond,anchor=physical[0]['anchor'],provenance=dict(actual_source=str(path),source_sha256=sha(path),actual_case=ix,method=row['method'],role='saved_actual_full_source_prefix_and_episode',no_new_physics=True,start_time=physical[0]['time'],end_time=physical[-1]['time'],no_unrecorded_settle_reconstruction=True))
                with gzip.open(dest,'wt')as f:json.dump(payload,f)
                exported[ix]=str(dest);provenance.append(dict(case_index=ix,method=row['method'],parent_id=row['parent_id'],goal_stream=row['goal_stream'],source_sha256=sha(path),merged_sha256=sha(dest),start_s=physical[0]['time'],end_s=physical[-1]['time'],evaluation=dict(error_cm=row['all']['error_cm'],force_rms_N=row['all']['force_rms'],Goal_aware_J=row['goal_aware']['J'])))
            return dict(path=exported[ix],title=title,role='FROZEN FINAL | ACTUAL EXECUTION')
        videos=[dict(name='01-near-FOLLOW-four-methods',grid4=True,sides=[entry(1,'E 32k | small request | FOLLOW'),entry(25,'I 32k | small request | FOLLOW'),entry(49,'Goal-only | standing/arm15 lookup'),entry(73,'Native standing | frozen original BFM')]),dict(name='02-mid-FOLLOW-tracking-shortfall',grid4=True,sides=[entry(17,'E 32k | mid request | FOLLOW'),entry(41,'I 32k | mid request | FOLLOW'),entry(65,'Goal-only | standing/arm15 lookup'),entry(89,'Native standing | frozen original BFM')]),dict(name='03-common-prefix-STOP-versus-CONTINUE',grid4=True,cross_condition_description='Same source, same Goal stream, same partner prefix then stop/continue; each row is one frozen checkpoint. Not independent parents.',sides=[entry(9,'E 32k | partner STOPS'),entry(10,'E 32k | partner CONTINUES'),entry(33,'I 32k | partner STOPS'),entry(34,'I 32k | partner CONTINUES')])]
        (out/'video-pairs.json').write_text(json.dumps(videos,indent=2)+'\n');(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BFM_ROLLING_VIDEO_RUN=str(out),BFM_RECOVERY_OUTPUT=str(out),BFM_RENDER_ASSET_ROOT=str(R),PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('ei_v2_movie.py'))];job['renderer_command']=cmd;side.write_text(json.dumps(job,indent=2)+'\n')
        subprocess.run(cmd,env=env,cwd=out,check=True,timeout=1200)
        import imageio_ffmpeg
        manifest=json.loads((out/'manifest.json').read_text());assert len(manifest)==3
        for item in manifest:
            path=Path(item['path']);assert path.stat().st_size>10000;assert sha(path)==item['sha256']
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True,timeout=120,stdout=subprocess.DEVNULL)
        job.update(status='completed',videos=3,decode_all_frames=True,manifest=manifest)
        (out/'verified.json').write_text(json.dumps(dict(actual_qpos_only=True,physics_steps=0,full_decode=True,fps=25,playback_speed=1.,recorded_prefix_start_s=3.,unrecorded_initial_settle_0_3s_not_reconstructed=True,shorter_branch_end_frame_held=True),indent=2)+'\n')
    except BaseException as exc:
        job.update(status='failed_preserved',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        job.update(wall_s=time.monotonic()-start,end_epoch=time.time());side.write_text(json.dumps(job,indent=2)+'\n')
        if locks:
            lp=S/'ledger.json';ledger=json.loads(lp.read_text());ledger['charged_s']+=job['wall_s'];ledger['jobs'].append(job);tmp=lp.with_suffix('.pending');tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,lp)
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()
        print(json.dumps({k:v for k,v in job.items()if k!='manifest'}),flush=True)
if __name__=='__main__':main()
