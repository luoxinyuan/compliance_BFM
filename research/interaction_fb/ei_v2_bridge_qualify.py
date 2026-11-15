"""P0 real command chunking/partner/public-Goal qualification; no optimizer update."""
import os,sys,argparse,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import torch,numpy as np
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,exact
    from dual_track_task import global_rng,restore_global_rng
    from diagnostic_fb_resume import tree_digest
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.journal import dump
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    parents=json.loads((STAGE/'locks/expanded-corrected-train.json').read_text())['parents']
    parent=next(p for p in parents if p['family']=='near' and p['source']=='goal_arm5')
    env=GoalEnvironment(e,base,c,[parent]);env.reset(0)
    ob=env.proxy.enrich(env.task.extended_observation());ag=ContactGoalAgent(base,ob,'E',e.standing.device);env.agent=ag;env.proxy.agent=ag;env.proxy.stochastic=True
    before=tree_digest(ag.state());rng=global_rng();start=len(env.physical_stream);r,_=env.collect(5)
    full=dict(rows=copy.deepcopy(r),physics=copy.deepcopy(env.physical_stream[start:]),control=copy.deepcopy(env.task.controls),rng=global_rng(),anchor=env.task.anchor.copy())
    env.reset(0);restore_global_rng(rng);start=len(env.physical_stream);r1,_=env.collect(2);r2,_=env.collect(3)
    split=dict(rows=copy.deepcopy(r1+r2),physics=copy.deepcopy(env.physical_stream[start:]),control=copy.deepcopy(env.task.controls),rng=global_rng(),anchor=env.task.anchor.copy())
    assert exact(full['physics'],split['physics']),'1024-boundary chunking changed actual 5ms physics'
    assert exact(full['control'],split['control']),'chunking changed native action/qPD/control'
    assert exact(full['rng'],split['rng']),'chunking changed true sampling RNG'
    assert exact(full['anchor'],split['anchor'])
    for a,b in zip(full['rows'],split['rows']):
        for k in ('raw_u','action','z','old_mu','old_logp','observation','duration','aux_rewards'):
            assert exact(a[k],b[k]),('chunking changed executed on-policy record',k)
    # Repeat the source and same partner under public HOLD; anchor must remain exactly fixed.
    env.reset(0);env.issue('HOLD');hold_anchor=env.task.anchor.copy();assert exact(hold_anchor,full['anchor'])
    rows,st=env.collect(5);assert len(rows)==5 and env.goal_mode==1
    assert all('hold' in co['reward_terms'] for co in env.task.controls)
    assert tree_digest(ag.state())==before,'qualification changed weights/Adam'
    pack=dump(out/'bridge-actual.ptz',dict(full=full,split=split,hold=rows,physical=env.task.physics,reference=env.reference_trace,reference_anchor=hold_anchor))
    result=dict(status='COMMAND_CHUNKING_AND_COMMON_PARTNER_QUALIFIED',actual_full_vs_2plus3_control_physics_exact=True,actual_gaussian_rng_exact=True,raw_and_executed_records_exact=True,reference_full_handoff_restore_exact=True,follow_hold_same_partner_anchor_exact=True,hold_reward_branch_executed=True,formal_updates=0,optimizer_steps=0,archive=pack,remaining_gates=['full sampled physical preservation','categorical joint and GAE update','new-process optimizer-and-physics recovery','reward-ranking and full end-to-end cost'])
    (out/'bridge.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
