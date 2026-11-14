"""Frozen initial-function qualification on preselected training parents only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=ROOT.parents[2]

def study(e,m,c,out):
    import torch,numpy as np
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.journal import read
    from research.interaction_fb.execution_agent import ExecutionAgent
    from research.interaction_fb.execution_preflight import LOCK
    from research.interaction_fb.evaluation import run
    from diagnostic_fb_resume import tree_digest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    path=ROOT/'research_notes/r2_splits/train.json';data=json.loads(path.read_text());selected=[];seen=set()
    for g in data['parents']:
        if g['family'] in seen:continue
        seen.add(g['family'])
        for source in g['specs']:
            s=dict(source,split='train')
            if g['family']=='stop':s['pair_branch_at']=g['specs'][0]['stop_time']
            selected.append(s)
    assert seen=={'near','mid','stop','zero','none'} and len(selected)==6
    env=Environment(e,m,c);env.reset(0);obs=env.proxy.enrich(env.task.extended_observation())
    path=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-A-v3/checkpoints/round-128'
    meta=json.loads((path/'manifest.json').read_text());s=read(path/'core.ptz',meta['files']['core.ptz'])['agent']
    u={k.removeprefix('u.'):v for k,v in s['nets'].items() if k.startswith('u.')}
    torch.manual_seed(6301);np.random.seed(6301)
    agent=ExecutionAgent(m,obs,s['goal'].cuda(),torch.tensor(c['q_PD_scale'],device='cuda'),'E',u,LOCK)
    env.proxy.agent=agent;before=tree_digest(agent.state())
    results=[];prefixes={}
    for i,spec in enumerate(selected):
        dest=out/f'case-{i}';dest.mkdir()
        one=run(env,agent,dest,[spec]);row=one['results'][0];results.append(row)
        if 'pair_branch_at' in spec:
            controls=[{k:c[k] for k in ('bfm_obs','executed_z','action','q_PD','pre_qpos','pre_qvel')} for c in env.task.controls if c['time']<spec['pair_branch_at']-1e-8]
            digest=tree_digest(controls);key=spec['parent_id']
            if key in prefixes:assert prefixes[key]==digest,'Common prefix changed'
            else:prefixes[key]=digest
        (out/'completed-cases.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        if row['boundary'] is not None:break
    result=dict(status='qualification_completed' if len(results)==6 else 'qualification_stopped',results=results,optimizer_updates=0)
    assert tree_digest(agent.state())==before
    result['qualification_only']=True;result['engineering_models_not_loaded']=True
    result['hard_boundary_pass']=all(r['boundary'] is None for r in result['results'])
    (out/'qualification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);ARGS=p.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(ARGS.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
