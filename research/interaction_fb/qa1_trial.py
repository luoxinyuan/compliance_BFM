"""Approved QA1: fixed E replay; only original Q update with locked batch indices."""
import argparse,collections,copy,hashlib,itertools,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
P=ROOT.parents[2]
OLD=P/'experiments/r2-v11-qentry-20261114'
STAGE=P/'experiments/r2-g17-qa1-20261115'

def write(p,x):
    tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)

def study(e,m,c,out):
    import numpy as np,torch
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from research.interaction_fb.env import Environment
    from research.interaction_fb.approved_learning import ApprovedLearner
    from research.interaction_fb.approved_journal import load_state
    from research.interaction_fb.journal import dump,read
    from research.interaction_fb.schema import to_device
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng,restore_global_rng
    from dual_track_fb_online import stack
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    lock=json.loads((STAGE/'prep/fixed-schedule-01/lock.json').read_text())
    schedule=read(STAGE/'prep/fixed-schedule-01/schedule.ptz',lock['schedule'])
    source=Path(lock['source']);assert hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()==lock['source_manifest_sha256']
    s=load_state(source);rows=s['rows'];assert tree_digest(rows)==lock['source_replay_digest']
    assert tree_digest(s['agent'])==lock['source_agent_digest'];assert len(rows)==3500
    install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix')
    env=Environment(e,m,c);env.reset(0);obs=env.proxy.enrich(env.task.extended_observation())
    init=json.loads((ROOT/'research_notes/r2_ei_v11_entry_lock.json').read_text())
    class Fixed(ApprovedLearner):
        def batch(self,rr,kind):
            assert rr is rows and kind=='q' and 0<=self.qa_cursor<456
            ix=self.qa_schedule[self.qa_cursor];self.qa_cursor+=1
            return ix,stack([rr[int(i)] for i in ix])
    agent=Fixed(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),'E',init)
    agent.load(s['agent'])
    def frozen(a):
        ss=a.state();ss=copy.copy(ss)
        for k in ('audit','updates','q_steps'):ss.pop(k)
        for k in ('nets','targets'):ss[k]={n:v for n,v in ss[k].items() if not n.startswith(('q1.','q2.'))}
        ss['optimizers']={n:v for n,v in ss['optimizers'].items() if n not in ('q1','q2')}
        return tree_digest(ss)
    frozen0=frozen(agent);rng0=global_rng()
    archived=json.loads((OLD/'runs/paired-Q-ranking-12/qualification.json').read_text())
    traces=[]
    for state in archived['results']:
        for br in state['branches']:
            tr=read(OLD/'runs/paired-Q-ranking-12'/f"round-{state['round']}"/(br['id']+'.ptz'),br['archive'])
            traces.append((state,br,tr))
    # Before optimization: same actual input, one-step targets, full 1s target semantics.
    semantic=[];twenty=[];maxa=0.;maxz=0.;max_100ms=0.;exact=0;total=0
    with torch.no_grad():
        for st,br,tr in traces:
            branch_a=0.;branch_z=0.;series=[]
            for j,row in enumerate(tr['rows']):
                oo=to_device(row['observation']);no=to_device(row['next']['observation'])
                aa,zz,ii=agent.action(oo,target=True)
                if j or br['id']=='mean':
                    da=float((aa.cpu()-row['action']).abs().max());dz=float((zz.cpu()-row['z']).abs().max())
                    branch_a=max(branch_a,da);branch_z=max(branch_z,dz);maxa=max(maxa,da);maxz=max(maxz,dz)
                    if int(oo['_slot'].item())==0:max_100ms=max(max_100ms,da)
                    exact+=bool(torch.equal(aa.cpu(),row['action']) and torch.equal(zz.cpu(),row['z']));total+=1
                na,nz,ni=agent.action(no,target=True)
                r=float(row['aux_rewards']['contact_physical']);d=.99**(float(row['duration'])/.1)*(not bool(row['next']['terminated']))
                tq=min(float(agent.targets[k](agent.inp(no,na,nz,ni))) for k in ('q1','q2'))
                q=min(float(agent.nets[k](agent.inp(oo,row['action'].cuda(),row['z'].cuda(),row['zi'].cuda()))) for k in ('q1','q2'))
                series.append(dict(j=j,slot=int(oo['_slot'].item()),reward=r,discount=d,target_q=tq,current_q=q,backup=r+d*tq,residual=q-r-d*tq))
            twenty.append(dict(round=st['round'],candidate=br['id'],series=series))
            semantic.append(dict(round=st['round'],candidate=br['id'],target_vs_physical_mean_action_linf=branch_a,z_linf=branch_z,
                realized_G1s=br['discounted_return'],Q56=br['scores']['E']['q_min'],
                nstep_residual=br['scores']['E']['q_min']-br['discounted_return']-(.99**10)*series[-1]['target_q']))
    write(out/'preupdate-target-audit.json',dict(controls_compared=total,exact_controls=exact,max_action_difference=maxa,max_z_difference=maxz,
        high_level_boundary_max_action_difference=max_100ms,branches=semantic,
        hidden_future_test='UNMEASURED: archived40 does not include same complete physical/history state under two alternative future partner specifications. No proof of opposite optimal actions.',
        value_semantics='Frozen deterministic target mean is compared on actual visited states. Realized negative 1s cost is a pathwise upper bound only conditional on that exact deterministic future, not automatically the conditional expected Q without future spec. Large positive residual remains a calibration risk.'))
    write(out/'preupdate-20ms-backups.json',twenty)
    assert frozen(agent)==frozen0 and tree_digest(global_rng())==tree_digest(rng0)
    def evaluate():
        scores=[];allpairs=[]
        with torch.no_grad():
            for st,br,tr in traces:
                row=tr['rows'][0];oo=to_device(row['observation']);aa=row['action'].cuda();zz=row['z'].cuda();ii=agent.zi(1)
                qq=[float(agent.nets[k](agent.inp(oo,aa,zz,ii))) for k in ('q1','q2')]
                no=to_device(tr['rows'][-1]['next']['observation']);na,nz,ni=agent.action(no,target=True)
                tail=min(float(agent.targets[k](agent.inp(no,na,nz,ni))) for k in ('q1','q2'))
                oo2=to_device(row['next']['observation']);n2,z2,i2=agent.action(oo2,target=True)
                target0=min(float(agent.targets[k](agent.inp(oo2,n2,z2,i2))) for k in ('q1','q2'))
                pred=min(qq);g=br['discounted_return'];backup0=float(row['aux_rewards']['contact_physical'])+.99**.2*target0
                scores.append(dict(round=st['round'],parent=st['parent_id'],family=st['family'],candidate=br['id'],q=pred,q1=qq[0],q2=qq[1],target_tail=tail,
                    return_1s=g,nstep_residual=pred-g-.99**10*tail,TD_residual=pred-backup0,optimistic_prefix_gap=pred-g))
        for rnd in sorted({x['round'] for x in scores}):
            ss=[x for x in scores if x['round']==rnd]
            for x,y in itertools.combinations(ss,2):
                d=x['return_1s']-y['return_1s'];eligible=abs(d)>=1e-4
                allpairs.append(dict(round=rnd,parent=x['parent'],a=x['candidate'],b=y['candidate'],eligible=eligible,correct=d*(x['q']-y['q'])>0))
        pp=[x for x in allpairs if x['eligible']];assert len(pp)==58 and len({x['parent'] for x in pp})==4
        byp={p:sum(x['correct'] for x in pp if x['parent']==p)/sum(x['parent']==p for x in pp) for p in {x['parent'] for x in pp}}
        byround={r:dict(correct=sum(x['correct'] for x in pp if x['round']==r),n=sum(x['round']==r for x in pp)) for r in sorted({x['round'] for x in scores})}
        return dict(correct=sum(x['correct'] for x in pp),n=58,accuracy=sum(x['correct'] for x in pp)/58,
            parent_balanced_accuracy=float(np.mean(list(byp.values()))),by_parent=byp,by_round=byround,scores=scores,comparisons=allpairs)
    base=evaluate();assert base['correct']==26
    write(out/'node-56.json',base)
    def packet(branch,node):
        return dict(version='QA1-full-Q-state-v1',source_manifest_sha256=lock['source_manifest_sha256'],source=str(source),
            agent=agent.state(),branch=branch,node=node,cursor=agent.qa_cursor,
            schedule_sha256=lock['schedule']['sha256'],rng=global_rng(),frozen_digest=frozen0,
            unchanged_physical_state_reference=str(source),no_new_main_rollout=True)
    def save(branch,node):
        path=out/f'{branch}-Q{node}.ptz';pa=packet(branch,node);digest=tree_digest(pa);info=dump(path,pa)
        rr=read(path,info);assert tree_digest(rr)==digest
        write(path.with_suffix('.manifest.json'),dict(**info,logical_digest=digest,source_manifest_sha256=lock['source_manifest_sha256'],node=node,branch=branch,readback_exact=True))
        return info
    results={};timings=[]
    for branch in ('U','C'):
        agent.load(s['agent']);restore_global_rng(rng0);agent.qa_cursor=0;agent.qa_schedule=schedule[branch]
        assert frozen(agent)==frozen0 and tree_digest(agent.state())==lock['source_agent_digest']
        counts=schedule['historical_counts'].numpy().copy();records=[];nodes={56:base};snapshots={56:save(branch,56)}
        for group in range(57,513):
            before=time.monotonic()
            with torch.enable_grad():rec=agent.q_update(rows)
            torch.cuda.synchronize()
            assert rec['indices']==schedule[branch][group-57].tolist()
            for j in rec['indices']:counts[j]+=1
            rec.update(total_group=group,new_unique=int((counts>0).sum()),update_s=time.monotonic()-before)
            # Each parent/phase exposure includes historical56, all repetitions, and unique rows.
            rec['by_stratum']={ss:dict(draws=int(counts[[m['index'] for m in schedule['metadata'] if m['stratum']==ss]].sum()),
                unique=int((counts[[m['index'] for m in schedule['metadata'] if m['stratum']==ss]]>0).sum())) for ss in lock['strata']}
            records.append(rec)
            assert np.isfinite(rec['losses']).all() and agent.actor_steps==0
            if group in (128,256,512):
                assert frozen(agent)==frozen0,'Non-Q mutable state changed'
                assert tree_digest(global_rng())==tree_digest(rng0),'Unscheduled global RNG draw'
                nodes[group]=evaluate();snapshots[group]=save(branch,group)
                nodes[group]['unique']=int((counts>0).sum());nodes[group]['old_target_fraction']=.99**(group-56)
                write(out/f'{branch}-node-{group}.json',nodes[group]);write(out/f'{branch}-updates.json',records)
                print('QA1_NODE',branch,group,nodes[group]['correct'],58,'unique',nodes[group]['unique'],flush=True)
        results[branch]=dict(nodes=nodes,snapshots=snapshots,new_Q_groups=456,actor_groups=0,nonQ_exact=True,global_rng_exact=True)
        write(out/'results-partial.json',results)
    qualified=[]
    for branch,result in results.items():
        end=result['nodes'][512];high=[x for x in end['scores'] if x['candidate']=='mean' and x['round'] in (5,9)]
        # Conservative exact pathwise calibration check, not a replacement for the unchanged ranking gate.
        calibration=all(x['q']<=x['return_1s'] for x in high)
        result['endpoint_calibration_clear']=calibration
        if end['accuracy']>=.7 and calibration:qualified.append(branch)
    answer=dict(status='QA1_FIXED_REPLAY_COMPLETE',retrospective=results,prospective_eligible=qualified,
        prospective_status='PENDING_PRELOCKED_PHYSICS' if qualified else 'NOT_TRIGGERED_ENDPOINT_FAIL',
        actor_updates=0,I_updates=0,new_real_online_rows=0,old_G17='FAIL_UNCHANGED',
        sealed_read=False,source_manifest_sha256=lock['source_manifest_sha256'])
    write(out/'qa1-result.json',answer);return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
