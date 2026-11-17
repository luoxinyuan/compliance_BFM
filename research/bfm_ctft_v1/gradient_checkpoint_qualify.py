"""Qualify smaller actual checkpoints without omitting effective learner state.

Starting from a real complete packed state, clear consumed .grad buffers in the
actual held agent, save that exact state and compare the next physical window
and full native update against the unchanged-gradient reference. No formal run.
"""
import argparse
import copy
import json
import os
from pathlib import Path
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
    from packed_checkpoint import PackedTensorStore
    from cas_checkpoint import restore_learner,learner_state
    from gradient_housekeeping import clear_consumed
    from learning_checkpoint import fingerprint,digest,one_update
    from contact_live import ContactLive
    from contact_replay import cat
    from f1_mjdata_contract import metadata,canonical
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_GRAD_ARGS']);source=PackedTensorStore(Path(args['source']))
    pack=source.load('round-3');state=pack['learner']
    with torch.enable_grad():agent,buffers=restore_learner(state)
    coefficients=json.loads((STAGE/'runs/s0-online-probe-01/agent-identity.json').read_text())['original_training_config']['aux_rewards_scaling']
    live=ContactLive(e,STAGE,coefficients);live.restore(pack['task'])
    if args['mode']=='reference':
        change=clear_consumed(agent);write(out/'housekeeping.json',change)
        cleaned=learner_state(agent,step=state['step'],replay_segments=state['replay_segments'],
            replay_capacity=state['replay_capacity'],replay_idx=state['replay_idx'],reference=state['reference'])
        store=PackedTensorStore(out/'store');base=copy.deepcopy(source.metadata('round-3'))
        def rebase(v):
            if isinstance(v,dict):
                if 'file' in v and 'raw_sha256' in v:v['file']=os.path.relpath(source.root/v['file'],store.root)
                for x in v.values():rebase(x)
            elif isinstance(v,(list,tuple)):
                for x in v:rebase(x)
        rebase(base)
        _,stats=store.save('round-3',dict(learner=cleaned,task=live.snapshot(),scheduler=pack['scheduler'],housekeeping=change),base=base,anchor=True)
        write(out/'save.json',stats)
    else:assert all(p.grad is None for p in agent._model.parameters()),'Actual restored gradients are notNone'
    e.trace=[];rows=[]
    for i in range(12):
        before=live.observation();z=live.command()
        with torch.no_grad():action=agent.act(before,z,mean=False)
        rows.append(live.step(action,before,z))
    torch.save(rows,out/'actual-causal-rows.pt');buffers['train'].extend(cat(rows))
    with torch.enable_grad():updated=one_update(agent,buffers,state['step'])
    answer=dict(actual_rows_sha256=digest(rows),physics_sha256=digest(e.trace),
        full_task_sha256=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
        learner={k:updated[k] for k in ['state','sample_trace','metrics']},committed_replay_rows=buffers['train']._idx)
    write(out/'continuation.json',answer)
    expected=json.loads((Path(args['reference'])/'continuation.json').read_text())
    different=[k for k in answer if expected[k]!=answer[k]]
    write(out/'comparison.json',dict(status='PASS' if not different else 'FAIL',different=different,
        scope='Clear only already consumed gradients; actual12 controls plus next complete native update versus unchanged-gradient reference',
        actual_gradNone_checkpoint_restore=args['mode']=='resume',formal_training=False))
    assert not different,('Consumed-gradient housekeeping changed actual execution/update',different)
    return dict(status='CONSUMED_GRADIENT_CLEAR_EQUIVALENT' if args['mode']=='reference' else 'GRAD_NONE_CHECKPOINT_RESTORE_EXACT',
        formal_training=False,actual_rows=12,complete_updates=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--mode',choices=['reference','resume'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from resource_guard import check
    check(STAGE)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    os.environ.update(CTFT_GRAD_ARGS=json.dumps(dict(mode=a.mode,source=a.source,reference=a.reference)),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kwargs):
        n=int(kwargs.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=5000
        result=raw(*args,**kwargs);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output;begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_updates=0))


if __name__=='__main__':main()
