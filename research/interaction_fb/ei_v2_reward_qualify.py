"""P1 fixed four-state/four-command causal reward qualification, no learning/search labels."""
import os,sys,argparse,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def measure(ph,controls,goal_world,goal_mode):
    import numpy as np
    from .ei_v2_quality import quality
    active=[p for p in ph if p['active']];hold=int(goal_mode)==1
    errors=[np.asarray(p['wrist'])-goal_world if hold else np.asarray(p['contacts'][0]['anchor'])-p['contacts'][0]['point'] for p in active]
    forces=[p['force'] for p in active]
    gamma=.99**.2;discount=1.;ret=0.
    for c in controls:ret+=discount*c['reward_integral'];discount*=gamma**(c['actual_duration_s']/.02)
    return dict(error_m=float(np.sqrt(np.mean(np.sum(np.asarray(errors)**2,axis=1)))) if active else None,force_N=float(np.sqrt(np.mean(np.sum(np.asarray(forces)**2,axis=1)))) if active else None,active_ticks=len(active),discounted_return=ret,integrated_return=sum(c['reward_integral'] for c in controls),quality=quality(ph))

def study(e,base,c,out):
    import torch,numpy as np,random,subprocess
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,canonical,metadata
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.ei_v2_checkpoint import restore
    from research.interaction_fb.ei_v2_quality import EnvBranch
    from research.interaction_fb.journal import dump
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import native_terms
    from research.interaction_fb import benchmark
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
    parents=json.loads((STAGE/'locks/qualification-parents.json').read_text())['parents'];env=GoalEnvironment(e,base,c,parents);env.reset(0)
    install_contact_details(e);install_post_velocity(e)
    agent=ContactGoalAgent(base,env.proxy.enrich(env.task.extended_observation()),'E',e.standing.device);env.agent=agent;env.proxy.agent=agent
    src=STAGE/'runs/P1-frozen-sample-01';qual=json.loads((src/'qualification.json').read_text());assert qual['status']=='FROZEN_PHYSICAL_QUALIFICATION_PASS'
    indices=[r for r in json.loads((src/'source-index.json').read_text()) if r['label'].startswith('reward-')];assert len(indices)==4
    rows=[];pairs=[];identities=[];reward_checks=[]
    for st in indices:
        path=Path(st['file']['path']);restore(path,agent,env,out);baseline=tree_digest(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)));branch=EnvBranch(env);parent=env.task.spec['parent_id'];mode=env.goal_mode;goal=copy.deepcopy(env.goal_world)
        for style,options in [('constant_command',range(4)),('first_100ms_then_shared_goal_policy',(0,2))]:
            values=[]
            for option in options:
                if benchmark.STOP_REQUESTED:raise RuntimeError('BUDGET_STOP_BEFORE_REWARD_QUALIFICATION_COMPLETE')
                branch.restore();assert tree_digest(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)))==baseline
                env.collection_role='frozen_reward_counterfactual';env.proxy.stochastic=False;env.phase='P2';env.reference_constant_option=option
                fp=len(env.physical_stream);cp=len(env.task.controls);rp=len(env.rows);end=float(e.tether.time)+2.
                boundary=None
                try:
                    env.collect(5)
                    if style=='first_100ms_then_shared_goal_policy':del env.reference_constant_option
                    while float(e.tether.time)<end-1e-7 and not env.task.done:env.collect(5)
                except RuntimeError as exc:
                    if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                    boundary=str(exc)
                ph=copy.deepcopy(env.physical_stream[fp:]);ctrl=copy.deepcopy(env.task.controls[cp:]);actual=copy.deepcopy(env.rows[rp:]);m=measure(ph,ctrl,goal,mode)
                checks=[]
                for j,cc in enumerate(ctrl):
                    rr=[p for p in ph if cc['time']-1e-8<=p['force_time']<cc['time']+cc['actual_duration_s']-1e-8]
                    assert rr
                    original=cc.get('v2_old_native_reward',cc['reward_integral'])
                    oldterms=native_terms(rr,cc['q_PD'],ctrl[j-1]['q_PD'] if j else env.task.controls[cp-1]['q_PD'],env.task.spec)
                    independent=-cc['actual_duration_s']*sum(oldterms.values())-cc.get('absorbing_failure_penalty',0.)
                    assert independent==original,('original native reward mismatch',independent,original)
                    if 'v2_goal_cost' in cc:
                        # Exact same listed three replacements; public Goal is fixed across these windows.
                        removed=sum(oldterms[k] for k in ('tracking','relative_velocity','heading'))
                        expected=original+cc['actual_duration_s']*(removed-cc['v2_goal_cost'])
                        # G02 exact expression mirrors actual grouped-native implementation arithmetic.
                        delta=abs(cc['reward_integral']-expected)
                        assert delta<=1e-12,(delta,cc)
                        if not any(p['active'] for p in rr):assert cc['v2_goal_cost']==0
                        checks.append(dict(mode='HOLD',replacement_error=delta,released_task_cost_zero=not any(p['active'] for p in rr)))
                    else:
                        assert cc['reward_integral']==original
                        checks.append(dict(mode='FOLLOW',old_native_reward_exact=True))
                archive=dump(out/f"case-{st['case']}-{style}-{option}.ptz",dict(physics=ph,controls=ctrl,actual_commands=actual,source=st,identity='counterfactual_physical_reward_probe_NOT_PPO_NOT_TEACHER_LABEL'))
                r=dict(case=st['case'],parent_id=parent,label=st['label'],goal='HOLD' if mode else 'FOLLOW',style=style,option=option,source_full_state_exact=True,metrics=m,boundary=boundary,archive=archive)
                rows.append(r);values.append(r);reward_checks.append(dict(case=st['case'],style=style,option=option,checks=checks));(out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            # Conservative physical dominance: no force/error/slip/yaw/support/joint degradation.
            for i,a in enumerate(values):
                for b in values[i+1:]:
                    for better,worse in [(a,b),(b,a)]:
                        x,y=better['metrics'],worse['metrics'];qx,qy=x['quality'],y['quality']
                        if x['error_m'] is None or y['error_m'] is None:continue
                        meaningful=y['error_m']-x['error_m']>=.01 or y['force_N']-x['force_N']>=.5
                        dominates=x['error_m']<=y['error_m'] and x['force_N']<=y['force_N'] and all(qx[k]<=qy[k] for k in ('slip_peak','yaw_excursion_deg','unsupported_ticks','max_joint_excess_rad')) and not better['boundary']
                        if meaningful and dominates:pairs.append(dict(parent_id=parent,case=st['case'],style=style,better=better['option'],worse=worse['option'],error_improvement_m=y['error_m']-x['error_m'],force_improvement_N=y['force_N']-x['force_N'],reward_order_correct=x['discounted_return']>y['discounted_return']))
        branch.restore();identities.append(dict(case=st['case'],parent_id=parent,source_digest=baseline,goal='HOLD' if mode else 'FOLLOW'))
    by={}
    for p in pairs:by.setdefault(p['parent_id'],[]).append(p['reward_order_correct'])
    correct=float(np.mean([np.mean(v) for v in by.values()])) if by else None
    # Full per-candidate traces remain reviewable. Absence of a structural counterexample is not inferred from lack of pairs.
    enough=len(pairs)>=8 and len(by)>=4
    result=dict(status='REWARD_QUALIFICATION_PASS' if enough and correct>=.8 else 'REWARD_QUALIFICATION_UNDERPOWERED' if not enough else 'REWARD_QUALIFICATION_FAIL',states=identities,candidates=rows,meaningful_strict_quality_pairs=pairs,pair_count=len(pairs),independent_parent_count=len(by),parent_equal_correct_fraction=correct,strict_dominance_definition='force/error weak Pareto, >=1cm or .5N useful difference; slip/yaw/support/joint each no worse',structural_counterexample_assessment='full candidate physical summaries required; PASS also requires explicit review before any optimizer',reward_checks=reward_checks,optimizer_updates=0,sealed_read=False)
    (out/'reward.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    from research.interaction_fb.ei_v2_counter_io import install
    install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
