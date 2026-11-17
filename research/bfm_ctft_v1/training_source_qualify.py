"""Frozen sampled source/partner qualification; no optimizer or formal data."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def study(e,original,contract,out):
    import torch
    from agent import build,CRITIC_KEY
    from learning_checkpoint import cpu,rng,set_rng
    from training_live import TrainingLive
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    scale=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())['value']
    with torch.enable_grad():agent,identity=build(aux_scale=scale,compile_execution=False)
    write(out/'identity.json',identity)
    audit=json.loads((STAGE/'runs/s0-source-admission-01/result.json').read_text())
    live=TrainingLive(e,STAGE,identity['original_training_config']['aux_rewards_scaling'],audit['train'])
    plan=json.loads((STAGE/'locks/training-source-qualification-v1.json').read_text())
    results=[];paired_rng=None
    for i,spec in enumerate(plan['cases']):
        if i%2==0:paired_rng=rng()
        else:set_rng(paired_rng)
        live.start(spec);rows=[];mean_max=0.;leak_max=0.;boundary=None
        try:
            while not live.done:
                assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
                before=live.observation();z=live.command()
                with torch.no_grad():
                    baseline=original.act(e.obs,z,mean=True);extended=agent.act(before,z,mean=True)
                    mean_max=max(mean_max,float((baseline-extended).abs().max()))
                    assert torch.equal(baseline,extended),'Original zero-column mean changed'
                    changed=dict(before,**{CRITIC_KEY:before[CRITIC_KEY]+1234.})
                    hidden=agent.act(changed,z,mean=True);assert torch.equal(hidden,extended),'Partner field leaked to Actor'
                    altered=dict(before,time=torch.full_like(before['time'],123456)) if 'time' in before else before
                    assert torch.equal(agent.act(altered,z,mean=True),extended),'Task clock reached Actor'
                    action=agent.act(before,z,mean=False)
                rows.append(live.step(action,before,z))
        except UnsafePhysics as exc:boundary=copy.deepcopy(exc.row)
        finally:e.qualification_active=False
        path=out/f'case-{i:02d}.pkl.gz'
        with gzip.open(path,'wb',compresslevel=1) as f:pickle.dump(dict(spec=spec,actual_rows=rows,
            actual_physics=e.trace,boundary=boundary,final_Task=live.snapshot()),f,protocol=5)
        result=dict(case=i,source=spec['source'],mode=spec['mode'],force=spec['force'],
            complete=live.done and boundary is None,controls=len(rows),physics_ticks=len(e.trace),
            original_mean_max_abs=mean_max,hidden_partner_and_clock_invariance_exact=True,
            boundary=boundary,evidence_file=path.name,updates=0)
        results.append(result);write(out/'progress.json',results)
        if boundary is not None:
            write(out/'qualification.json',dict(status='FROZEN_SAMPLED_SOURCE_HARD_BOUNDARY',cases=results,formal_training=False))
            raise RuntimeError('Original sampled source execution hit real hard boundary; no recipe change or source skipping')
    write(out/'qualification.json',dict(status='FROZEN_SAMPLED_SOURCE_PARTNER_QUALIFIED',cases=results,formal_training=False))
    return dict(status='FROZEN_SAMPLED_SOURCE_PARTNER_QUALIFIED',cases=len(results),updates=0,formal_training=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    check(STAGE)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kwargs):
        n=int(kwargs.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=24000
        result=raw(*args,**kwargs);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output;begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_updates=0))


if __name__=='__main__':main()
