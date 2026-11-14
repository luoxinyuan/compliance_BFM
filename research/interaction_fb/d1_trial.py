"""Issue63 D1: one actual 20ms intervention, common frozen V1 continuation."""
import argparse, copy, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
from research.interaction_fb import return_trial as rt
P=ROOT.parents[2]
ARCH=P/'experiments/adapter-return-pilot-20261113'


def evaluate(env,agent,out,case,reset,prefix,incoming):
    import torch
    from research.interaction_fb.agent import Agent
    from research.interaction_fb.envelope import Envelope
    from research.interaction_fb.schema import robot,to_device
    from diagnostic_fb_resume import tree_digest
    from f1_mjdata_contract import exact
    operator=json.loads((ROOT/'research_notes/adapter_envelope_lock_v1.json').read_text())['operator']
    idx=rt.ARGS.case_index
    assert idx in (2,3) and case['role']=='recovery_dev'
    original_path=ARCH/f'runs/evaluate-case-{idx:02d}-01/V1-mean/trace.pt'
    archived=torch.load(original_path,map_location='cpu',weights_only=False)
    identities=json.loads((ARCH/'runs/fit-01/checkpoint-identities.json').read_text())
    candidate=Agent(agent.base,env.proxy.enrich(env.task.extended_observation()),agent.goal,agent.scale,'B')
    results=[]
    for name in ('V1','control','return','base-only'):
        reset();agent.load(incoming);agent.envelope=Envelope(operator,agent.goal.device)
        env.proxy.agent=agent
        model=incoming
        if name in ('control','return'):
            path=ARCH/f'runs/fit-01/{name}/adapter-64.pt'
            assert rt.sha(path)==identities[name+'64']['sha256']
            model=torch.load(path,map_location='cpu',weights_only=False)['agent']
        candidate.load(model);candidate.envelope=Envelope(operator,agent.goal.device)
        original=agent.action;count=[0]
        def first_only(obs,target=False,sample=False,z_override=None):
            assert not target and not sample
            common,z,zi=original(obs,False,False,z_override)
            if count[0]==0:
                if name=='base-only':
                    base=agent.base._actor(agent.base._normalize(robot(obs)),z,agent.base.cfg.actor_std).mean
                    a,audit=agent.envelope(base,base.clamp(-1,1))
                    agent.last_action_audit=dict(audit,adapter=torch.zeros_like(a),noise=torch.zeros_like(a))
                else:
                    a,cz,ci=candidate.action(obs,False,False,z_override=z)
                    assert torch.equal(cz,z) and torch.equal(ci,zi)
                    agent.last_action_audit=copy.deepcopy(candidate.last_action_audit)
                if name=='V1':assert torch.equal(a,common)
            else:a=common
            count[0]+=1
            return a,z,zi
        agent.action=first_only
        dest=out/name;dest.mkdir()
        try:r=rt.branch(env,agent,dest,False,'envelope')
        finally:agent.action=original
        trace=torch.load(dest/'trace.pt',map_location='cpu',weights_only=False)
        assert count[0]==len(trace['rows'])
        r.update(candidate=name,first_action_controls=1,common_followup='V1 mean',
                 model_sha=rt.sha(ARCH/f'runs/fit-01/{name}/adapter-64.pt') if name in ('control','return') else rt.sha(rt.OLD/'pilot-B-v3/checkpoints/round-19/core.ptz'))
        if name=='V1':
            assert exact(trace['physics'],archived['physics']),'V1 archived physics reproduction failed'
            assert len(trace['controls'])==len(archived['controls'])
            for i,(a,b) in enumerate(zip(trace['controls'],archived['controls'])):
                assert exact(a,b),('V1 archived control reproduction failed',i)
            for a,b in zip(trace['rows'],archived['rows']):
                assert torch.equal(a['action'],b['action']) and torch.equal(a['z'],b['z'])
            r['archived_physics_controls_actions_exact']=True
        row=trace['rows'][0];o=to_device(row['observation']);no=to_device(row['next']['observation'])
        a=row['action'].cuda();z=row['z'].cuda();zi=row['zi'].cuda()
        with torch.no_grad():
            q=agent.nets['q'](agent.inp(o,a,z,zi))
            # Original B19 target contract has no Envelope. Forward only; never executed.
            envlp=agent.envelope;agent.envelope=None
            try:na,nz,ni=agent.action(no,target=True,sample=False)
            finally:agent.envelope=envlp
            tail=agent.targets['q'](agent.inp(no,na,nz,ni))
            reward=row['aux_rewards']['contact_physical'].cuda()
            d=.99**(row['duration'].cuda()/.1)*(~row['next']['terminated'].cuda())
            bell=reward+d*tail
        rewards=[float(x['aux_rewards']['contact_physical']) for x in trace['rows']]
        gamma=.99**.2
        segments={}
        for label,lo,hi in [('first20ms',0,1),('20to100ms',1,5),('100msTo1s',5,50),('1to3s',50,150)]:
            segments[label]=sum(gamma**j*rewards[j] for j in range(lo,min(hi,len(rewards))))
        r['one_step']=dict(q=float(q),reward=float(reward),target_q=float(tail),bellman=float(bell),td=float(q-bell),target_action_executed=False)
        r['discounted_segments']=segments;r['discounted_return']=sum(segments.values())
        r['remaining_discount']=gamma**len(rewards)
        r['first_action']=a.cpu().tolist();r['initial_obs_digest']=tree_digest(o)
        results.append(r);rt.write(out/'results.json',results)
        print('D1_BRANCH',idx,name,r['status'],r['metrics'],r['discounted_return'],flush=True)
        if r['status']!='completed_3s':break
    assert len({x['initial_obs_digest'] for x in results})==1
    result=dict(status='complete' if len(results)==4 and all(x['status']=='completed_3s' for x in results) else 'stopped',
                case=case,prefix=prefix,branches=results,archived_trace_sha256=rt.sha(original_path),training_updates=0)
    rt.write(out/'evaluation.json',result)
    return result


def main():
    import mujoco
    ledger=json.loads((ROOT.parent/'ledger.json').read_text())
    remaining=18000-sum(x.get('physics_ticks',0) for x in ledger['job_entries'])
    original_step=mujoco.mj_step
    used=[0]
    def bounded_step(*a,**kw):
        n=int(kw.get('nstep',1))
        if used[0]+n>remaining:raise RuntimeError('D1_PHYSICS_BUDGET')
        used[0]+=n
        return original_step(*a,**kw)
    mujoco.mj_step=bounded_step
    rt.evaluate_case=evaluate
    rt.main()
if __name__=='__main__':main()
