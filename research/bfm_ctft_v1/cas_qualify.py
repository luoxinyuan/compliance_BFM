"""Full native-reference learner + real Task CAS/delta fresh-process regression."""
import argparse
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
    import torch,numpy as np
    from agent import build
    from contact_replay import build_buffers,cat
    from native_reference import build_reference
    from cas_checkpoint import TensorStore,learner_state,restore_learner,durable
    from learning_checkpoint import fingerprint,file_sha,digest,one_update
    from contact_live import ContactLive
    from f1_mjdata_contract import metadata,canonical
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    args=json.loads(os.environ['CTFT_CAS_ARGS']);mode=args['mode']
    if args.get('store_kind','cas')=='packed':
        from packed_checkpoint import PackedTensorStore as TensorStore
    location=out/'store' if mode=='reference' else Path(args['store'])
    store=TensorStore(location)
    if mode=='reference':
        scale=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())['value']
        with torch.enable_grad():agent,identity=build(aux_scale=scale,compile_execution=False)
        write(out/'identity.json',identity)
        reference,ref_identity=build_reference(e.base,STAGE,out/'reference')
        write(out/'reference-contract.json',ref_identity)
        buffers,_,provenance=build_buffers(ROOT,identity['original_training_config']['aux_rewards_scaling'])
        buffers['expert_slicer']=reference
        write(out/'actual-replay-provenance.json',provenance)
        initial=buffers['train'].get_full_buffer();replay_path=out/'actual-initial-replay.pt';torch.save(initial,replay_path)
        segments=[dict(path=str(replay_path),sha256=file_sha(replay_path))]
        agent.actor_enabled=False
        with torch.enable_grad():one_update(agent,buffers,0)
        agent.actor_enabled=True
        with torch.enable_grad():one_update(agent,buffers,1)
        live=ContactLive(e,STAGE,identity['original_training_config']['aux_rewards_scaling']);live.start(dict(mode=1,force=True))
        def package(step):
            state=learner_state(agent,step=step,replay_segments=segments,replay_capacity=buffers['train'].capacity,
                replay_idx=buffers['train']._idx,reference=out/'reference')
            return dict(learner=state,task=live.snapshot(),scheduler=dict(actual_rows=len(buffers['train']),update_step=step))
        before=fingerprint(agent)
        base,stats=store.save('round-2',package(2),anchor=True);assert before==fingerprint(agent)
        write(out/'save-2.json',stats)
        rows=[]
        for i in range(12):
            obs=live.observation();z=live.command()
            with torch.no_grad():action=agent.act(obs,z,mean=False)
            rows.append(live.step(action,obs,z))
        new=cat(rows);buffers['train'].extend(new)
        replay_path=out/'actual-next-replay.pt';torch.save(new,replay_path);segments.append(dict(path=str(replay_path),sha256=file_sha(replay_path)))
        with torch.enable_grad():one_update(agent,buffers,2)
        pack=package(3);expected=fingerprint(agent)
        _,stats=store.save('round-3',pack,base=base);assert expected==fingerprint(agent)
        write(out/'save-3.json',stats)
        assert store.load('round-2')['scheduler']['update_step']==2
        # Transaction interruption leaves the current complete generation exact.
        prior=(location/'latest.json').read_bytes()
        for kind in ['after_blobs','before_publish']:
            try:store.save('injected-'+kind,pack,base=base,inject=kind)
            except RuntimeError as exc:assert 'INJECTED_' in str(exc)
            else:raise AssertionError('Failure injection did not interrupt')
            assert (location/'latest.json').read_bytes()==prior
            assert digest(store.load('round-3'))==digest(pack)
        write(out/'failure-injection.json',dict(two_transaction_boundaries=True,latest_unchanged=True,previous_complete_exact=True))
    else:
        pack=store.load('round-3')
        with torch.enable_grad():agent,buffers=restore_learner(pack['learner'])
        identity=json.loads((Path(args['reference'])/'identity.json').read_text())
        live=ContactLive(e,STAGE,identity['original_training_config']['aux_rewards_scaling']);live.restore(pack['task'])
        assert fingerprint(agent)==pack['learner']['initial_fingerprint']
        actual=canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))
        assert digest(actual)==digest(canonical(pack['task'],metadata(e.sim.model,e.sim.data))),'Initial full Task mismatch'
    e.trace=[];rows=[]
    for i in range(12):
        before=live.observation();z=live.command()
        with torch.no_grad():action=agent.act(before,z,mean=False)
        rows.append(live.step(action,before,z))
    buffers['train'].extend(cat(rows))
    with torch.enable_grad():updated=one_update(agent,buffers,3)
    answer=dict(actual_rows_sha256=digest(rows),physics_sha256=digest(e.trace),
        full_task_sha256=digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data))),
        learner={k:updated[k] for k in ['state','sample_trace','metrics']})
    write(out/'continuation.json',answer)
    if mode=='resume':
        reference=json.loads((Path(args['reference'])/'continuation.json').read_text());differences=[k for k in answer if answer[k]!=reference[k]]
        write(out/'comparison.json',dict(status='PASS' if not differences else 'FAIL',different=differences,
            initial_full_task_exact=True,actual_actions_physics_and_next_update_exact=not differences))
        assert not differences,('CAS new process continuation differs',differences)
    return dict(status='CAS_REFERENCE_AND_FAILURE_INJECTION_COMPLETE' if mode=='reference' else 'CAS_FRESH_PROCESS_RESTORE_EXACT',
        formal_training=False,real_rows=24 if mode=='reference' else 12,complete_updates=4 if mode=='reference' else 1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['reference','resume'],required=True)
    p.add_argument('--store');p.add_argument('--reference');p.add_argument('--store-kind',choices=['cas','packed'],default='cas')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CTFT_CAS_ARGS=json.dumps(dict(mode=a.mode,store=a.store,reference=a.reference,store_kind=a.store_kind)),
        BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    from resource_guard import check
    check(STAGE)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time()))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kwargs):
        n=int(kwargs.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=6000
        result=raw(*args,**kwargs);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output
    begun=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,formal_updates=0))


if __name__=='__main__':main()
