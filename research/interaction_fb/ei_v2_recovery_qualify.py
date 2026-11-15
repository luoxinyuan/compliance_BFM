"""P0 joint sampler and complete new-process recovery on frozen fresh policy.
Every physical row retained; no optimizer step or scientific checkpoint.
"""
import os,sys,json,argparse,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import random,torch,numpy as np
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from dual_track_task import global_rng
    from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.ei_v2_ppo import segments,current,advantages,kl
    from research.interaction_fb.ei_v2_checkpoint import Journal,restore
    from research.interaction_fb.journal import dump,read
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
    parents=json.loads((STAGE/'locks/expanded-corrected-train.json').read_text())['parents']
    parent=next(p for p in parents if p['family']=='near' and p['goal_stream']=='FOLLOW_TO_HOLD')
    env=GoalEnvironment(e,base,c,[parent]);env.reset(0);ob=env.proxy.enrich(env.task.extended_observation());a=ContactGoalAgent(base,ob,'E',e.standing.device);env.agent=a;env.proxy.agent=a;env.phase='P3'
    b=ContactGoalAgent(base,ob,'I',e.standing.device)
    for name in ('tail','film','selector','goal_encoder','value'):
        assert tree_digest(getattr(a,name).state_dict())==tree_digest(getattr(b,name).state_dict())
    hidden={k:v.clone() for k,v in ob.items()};hidden['_target_z']+=10;hidden['_slot']+=4;hidden['_option_memory']+=3
    assert torch.equal(a.value_input(ob),a.value_input(hidden)), 'state-value leaked the newly chosen action'
    before=tree_digest(a.state())
    if MODE=='write':
        # Frozen stochastic continuation crosses a real public Goal change; no future partner data in policy.
        while float(e.tether.time)<env.handoff+1.5-1e-7:env.collect(5)
        env.collect(2);assert env.offset==2 and env.goal_changed
        groups=segments(env.rows)
        with torch.no_grad():
            new,old=current(a,env.rows,groups,list(range(len(groups))));err=float((new-old).abs().max());assert err<=3e-4,err
            ad,re,gae=advantages(a,env.rows,groups);assert torch.isfinite(ad).all() and torch.isfinite(re).all();k=kl(a,env.rows,groups);assert k['joint']==0.,k
        journal=Journal(out/'checkpoints');manifest=journal.save(0,a,env,RUNTIME,'frozen_sampler_mid_option_mid_command')
        first=len(env.rows);fp=len(env.physical_stream)
        for n in (3,5,2):env.collect(n)
        after=canonical(portable(env.task),metadata(e.sim.model,e.sim.data))
        archive=dump(out/'expected-continuation.ptz',dict(rows=env.rows[first:],physics=env.physical_stream[fp:],after=after,rng=global_rng(),agent=a.state(),phase='P3'))
        assert tree_digest(a.state())==before
        result=dict(status='FROZEN_JOINT_SAMPLER_AND_CHECKPOINT_WRITTEN',Gaussian_and_real_option_logprob_error=err,strict_atol=3e-4,analytic_joint_KL=k,categorical_draws=sum(r['cat_boundary'] for r in env.rows),native_sampler_rows=len(env.rows),GAE=gae,goal_change_crossed=True,mid_command_offset=2,actual_updates=0,checkpoint=manifest,expected=archive,full_new_process_restore='PENDING')
    else:
        source=STAGE/'runs/P0-joint-recovery-write-01';old=json.loads((source/'recovery.json').read_text());state=restore(source/'checkpoints/block-0',a,env,out)
        first=len(env.rows);fp=len(env.physical_stream)
        for n in (3,5,2):env.collect(n)
        expected=read(source/'expected-continuation.ptz',old['expected'])
        actual=dict(rows=env.rows[first:],physics=env.physical_stream[fp:],after=canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),rng=global_rng(),agent=a.state(),phase='P3')
        checks={k:exact(actual[k],expected[k]) for k in actual};dump(out/'actual-continuation.ptz',actual)
        (out/'exact-fields.json').write_text(json.dumps(checks,indent=2)+'\n');assert all(checks.values()),checks
        result=dict(status='NEW_PROCESS_FULL_FROZEN_POLICY_RECOVERY_EXACT',checks=checks,actual_updates=0,optimizer_states='fresh empty Adam preserved; trained Adam qualification remains P1',Gaussian_and_option_rng_exact=True)
    (out/'recovery.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    global MODE,RUNTIME
    import subprocess
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['write','restore'],required=True);a=p.parse_args();MODE=a.mode;RUNTIME=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
