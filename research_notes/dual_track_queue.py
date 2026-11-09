"""Stage56 only: serial approved KL pair and1M probe, stop on any failed job.

No new experiments or dynamic recipe selection. Uses fixed immutable runtime;
each job rechecks authorized GPU occupancy and aggregate allowance. This queue
never treats a smoke update as a formal initial checkpoint.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--fb-verification',type=Path,required=True)
    a=p.parse_args();r=a.runtime/'research_notes';status=a.stage/'queue-status.json'
    assert not status.exists(),'Queue identity cannot be overwritten'
    def write(x):
        t=status.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(status)
    def job(name,category,purpose,seconds,mode,extra):
        write(dict(status='running',current=name,updated=time.time(),runtime=str(a.runtime)))
        cmd=[sys.executable,str(r/'dual_track_job.py'),'--stage',str(a.stage),'--runtime-repo',str(a.runtime),
             '--name',name,'--category',category,'--purpose',purpose,'--timeout',str(seconds),
             '--contract',str(r/'dual_track_formal_contract_v1.json'),'--',sys.executable,str(r/'dual_track_run.py'),
             '--mode',mode,'--output',str(a.stage/'runs'/name),*extra]
        subprocess.run(cmd,check=True)
        result=json.loads((a.stage/'jobs'/f'{name}.json').read_text())
        assert result['status']=='completed',result
    try:
        # Actual successful engineering result, not mere creation of a directory.
        high=json.loads((a.stage/'runs/high-smoke-01/result.json').read_text())
        fb=json.loads(a.fb_verification.read_text())
        assert high['study']['status']=='completed'
        assert fb.get('actual_action_physics_resume_exact') or fb.get('study',{}).get('actual_action_physics_resume_exact')
        for tag,kl in [('kl002','.02'),('kl008','.08')]:
            name='expert-'+tag
            job(name,'frozen','frozen_ppo',80000,'high_train',['--kl',kl])
            selected=json.loads((a.stage/'runs'/name/'selection.json').read_text())
            ck=a.stage/'runs'/name/f"candidate-{selected['iteration']}.pt"
            job(name+'-sealed','frozen','engineering',8000,'high_eval',
                ['--checkpoint',str(ck),'--sha256',selected['sha256'],'--iteration',str(selected['iteration']),'--split','sealed','--record'])
        job('fb-1m','fb','fb_train',260000,'fb_train',[])
        for step in (0,250000,1000000):
            ck=a.stage/'runs/fb-1m'/f'step{step}'
            job(f'fb-dev-{step}','fb','engineering',6000,'fb_eval',['--checkpoint',str(ck),'--split','dev'])
        job('fb-sealed-1m','fb','engineering',8000,'fb_eval',
            ['--checkpoint',str(a.stage/'runs/fb-1m/step1000000'),'--split','sealed','--record'])
        job('fb-reward-dev-1m','fb','engineering',1200,'fb_eval',
            ['--checkpoint',str(a.stage/'runs/fb-1m/step1000000'),'--split','dev','--fb-prompt','reward'])
        write(dict(status='completed',updated=time.time(),automatic_next_stage=False))
    except Exception as exc:
        write(dict(status='paused_on_failure',error=repr(exc),updated=time.time(),automatic_retry=False))
        raise


if __name__=='__main__':main()
