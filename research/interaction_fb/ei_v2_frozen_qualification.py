"""P1 full frozen mean/sample qualification including actual stop/release and same-state native recovery.
No optimizer, no formal training, no Dev task or sealed.
"""
import os,sys,argparse,json,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import random,torch,numpy as np
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard
    from diagnostic_fb_resume import tree_digest
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.ei_v2_quality import quality,protection,EnvBranch
    from research.interaction_fb.journal import dump
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
    ps=json.loads((STAGE/'locks/qualification-parents.json').read_text())['parents'];env=GoalEnvironment(e,base,c,ps);env.reset(0)
    # Read-only contact diagnostics installed after original archived prefix verification.
    install_contact_details(e);install_post_velocity(e)
    a=ContactGoalAgent(base,env.proxy.enrich(env.task.extended_observation()),'E',e.standing.device);b=ContactGoalAgent(base,env.proxy.enrich(env.task.extended_observation()),'I',e.standing.device)
    env.agent=a;env.proxy.agent=a;env.proxy.stochastic=MODE=='sample';env.phase='P2';before=tree_digest(a.state());before_b=tree_digest(b.state());results=[];keeps={};native_stats=[];total_start=time.monotonic()
    # Same initial output verified at every actual legal source/Goal/feedback state.
    execute=a.sample
    def checked(o,z,option,stochastic=True):
        with torch.no_grad():
            am=a.mean(o,z,option);bm=b.mean(o,z,option);assert torch.equal(am,bm)
        return execute(o,z,option,stochastic)
    a.sample=checked
    for idx,case in enumerate(env.cases):
        case_start=time.monotonic();env.reset_cursor=idx;env.reset(idx);start=len(env.rows);pstart=len(env.physical_stream);native_begin=len(env.native)-len(env.proxy.source_pairs);reference=None;reference_meta=None;reference_bad=[];reference_ph=None
        # Keep actual contiguous source qPD changes for parent-level retention tolerance.
        ns=env.proxy.source_pairs;qs=np.asarray([env.task.qof(r['action'].to(e.standing.device))[0].detach().cpu().numpy() for r in ns]);dq=np.diff(qs,axis=0)
        native_stats.append(dict(parent_id=case['spec']['parent_id'],p90=float(np.quantile(np.sqrt(np.mean(dq*dq,axis=1)),.9)),p95=float(np.quantile(abs(dq).reshape(-1),.95))))
        failed=[];release=case['spec']['release']
        try:
            while not env.task.done:
                if reference is None and case['spec']['enabled'] and case['spec']['distance']>0 and float(e.tether.time)>=release-1e-7:
                    assert abs(float(e.tether.time)-release)<1e-6,'release state must be actual native boundary'
                    branch=EnvBranch(env);env.reference_constant_option=0;env.proxy.stochastic=False;rp=len(env.physical_stream)
                    try:
                        while not env.task.done:env.collect(5)
                    except RuntimeError as exc:
                        if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                        reference_bad.append('hard_physics_boundary')
                    finally:
                        reference_ph=copy.deepcopy(env.physical_stream[rp:]);reference=quality(reference_ph) if reference_ph else None
                        if reference:reference_bad+=protection(reference,'recovery',reference)
                        else:reference_bad.append('no_physical_reference')
                        reference_meta=dump(out/f'case-{idx}-same-state-native-recovery.ptz',dict(physics=reference_ph,quality=reference,protection_failures=reference_bad,identity='actual_native_reference_from_actual_release_state_not_RL'))
                        branch.restore()
                    if reference_bad:failed.extend('native_reference_'+x for x in reference_bad);break
                env.collect(5)
        except RuntimeError as exc:
            if not str(exc).startswith('HARD_PHYSICS_BOUNDARY:'):raise
            failed.append('actual_hard_physics_boundary')
        ph=env.physical_stream[pstart:];whole=quality(ph) if ph else None;recph=[p for p in ph if p['force_time']>=release-1e-8];rec=quality(recph) if recph else None;family=case['spec']['demand_family']
        if env.task.boundary:failed.append('task_boundary')
        if whole and family in ('none','zero'):failed+=protection(whole,family)
        if case['spec']['enabled'] and case['spec']['distance']>0:
            if rec and reference:failed+=protection(rec,'recovery',reference)
            elif not failed:failed.append('UNMEASURED_release_or_reference')
        if not env.task.done and not failed:failed.append('incomplete_episode')
        actual=env.rows[start:];pre=[r for r in actual if float(r['provenance']['time'])<case['spec']['onset']-1e-8]
        eligible=actual if family in ('none','zero') else pre+([r for r in actual if float(r['provenance']['time'])>=release-1e-8] if rec and reference and not failed else [])
        # Finite predeclared eight-frame qualification set per parent; equal parent weights.
        selected=[]
        if eligible:
            for j in np.linspace(0,len(eligible)-1,min(8,len(eligible)),dtype=int):selected.append(copy.deepcopy(eligible[int(j)]))
        keeps.setdefault(case['spec']['parent_id'],[]).extend(selected)
        archive=dump(out/f'case-{idx}-actual.ptz',dict(spec=env.task.spec,rows=actual,physics=ph,controls=env.task.controls,instructions=[ins for ins in env.instructions if ins['episode']==env.reset_cursor],source_prefix=env.task.prefix_data,reference=reference_meta,identity='frozen_initial_mean_or_sampled_policy_qualification_not_PPO_training'))
        r=dict(index=idx,parent_id=case['spec']['parent_id'],family=family,goal_stream=case['goal_stream'],source=case['spec']['source'],K=case['spec']['h0_stiffness'],complete=env.task.done and env.task.boundary is None,whole=whole,recovery=rec,same_state_recovery_reference=reference,protection_failures=failed,actual_policy_rows=len(actual),keep_selected_rows=len(selected),archive=archive,wall_s=time.monotonic()-case_start,mode=MODE)
        results.append(r);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('EI_V2_FROZEN_CASE',idx,family,MODE,'failures',failed,flush=True)
        if failed:break
    assert tree_digest(a.state())==before and tree_digest(b.state())==before_b
    unique={r['parent_id']:r for r in native_stats};eps=dict(eps_rms=float(np.clip(.25*np.median([r['p90'] for r in unique.values()]),.015,.05)),eps_p95=float(np.clip(.5*np.median([r['p95'] for r in unique.values()]),.03,.10)),qpd_scale=c['q_PD_scale'],original_action_clip=c['clip'],parent_native_stats=list(unique.values()))
    keeps_meta=dump(out/'qualified-retention-data.ptz',dict(keep_by_parent=keeps,eps=eps,selection='stable8 endpoint-spaced eligible actual frames per parent per sibling; siblings subsequently share parent weight, not independent parents',qualification_complete=len(results)==len(env.cases) and all(not r['protection_failures'] for r in results)))
    result=dict(status='FROZEN_PHYSICAL_QUALIFICATION_PASS' if len(results)==len(env.cases) and all(not r['protection_failures'] for r in results) else 'FROZEN_PHYSICAL_QUALIFICATION_FAIL',mode=MODE,expected=len(env.cases),completed=len(results),results=results,retention=keeps_meta,eps=eps,elapsed_hot_s=time.monotonic()-total_start,formal_updates=0,formal_RL_rows=0,dev_task_episodes=0,sealed_read=False)
    (out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    global MODE
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['mean','sample'],required=True);a=p.parse_args();MODE=a.mode
    from research.interaction_fb import benchmark
    benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
