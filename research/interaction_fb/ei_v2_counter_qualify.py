"""Counter IO qualification against an archived actual-action physical branch."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,m,c,out):
    import torch
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,exact
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.journal import read,dump
    from research.interaction_fb.ei_v2_counter_io import self_test
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=STAGE/'runs/P0-command-bridge-01';meta=json.loads((old/'bridge.json').read_text());packet=read(old/'bridge-actual.ptz',meta['archive'])
    ps=json.loads((STAGE/'locks/expanded-corrected-train.json').read_text())['parents'];parent=next(p for p in ps if p['family']=='near' and p['source']=='goal_arm5')
    env=GoalEnvironment(e,m,c,[parent]);env.reset(0);agent=ContactGoalAgent(m,env.proxy.enrich(env.task.extended_observation()),'E',e.standing.device);env.agent=agent;env.proxy.agent=agent;env.proxy.replay=iter(packet['full']['rows']);env.collect(5);env.proxy.replay=None
    oldp=packet['full']['physics'];newp=env.physical_stream
    checks=dict(physics_exact=exact(oldp,newp),native_control_exact=exact(packet['full']['control'],env.task.controls),same_anchor_exact=exact(packet['full']['anchor'],env.task.anchor))
    assert all(checks.values()),checks
    result=dict(status='COUNTER_IO_BATCHING_REAL_PHYSICAL_PARITY_PASS',checks=checks,counter_only_fixture=self_test(),source_archive=meta['archive'],formal_updates=0,actual_physics_resolution_ms=5,merged_counter_writes_only=True,raw_action_reward_PD_contract_unchanged=True)
    (out/'counter-IO.json').write_text(json.dumps(result,indent=2)+'\n');dump(out/'counter-IO-actual.ptz',dict(physics=newp,controls=env.task.controls));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    from research.interaction_fb.ei_v2_counter_io import install
    install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
