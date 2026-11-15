"""Train-only signed OR1 physical gates, exact complete state isolation, no PPO labels."""
import copy,json,random
from pathlib import Path
import torch,numpy as np
from .ei_v2_agent import ContactGoalAgent
from .ei_v2_quality import EnvBranch,quality,protection
from .ei_v2_ppo import device_obs
from .journal import read,dump
from .checkpoint import sha
from diagnostic_frozen import portable
from diagnostic_fb_resume import tree_digest
from f1_mjdata_contract import canonical,metadata,exact
from dual_track_task import global_rng

ROOT=Path(__file__).resolve().parents[2];ARCHIVE=ROOT.parent.parent/'contact-goal-ei-v2-20261115'

def run(env,agent,out,mode,phase):
    from . import ei_v2_checkpoint as checkpoint
    out=Path(out);out.mkdir(parents=True,exist_ok=False);original_cases=env.cases;start_agent=tree_digest(agent.state());snap=EnvBranch(env);start=canonical(portable(env.task),metadata(env.e.sim.model,env.e.sim.data));start_rng=global_rng();oldload=checkpoint.load
    qualification=json.loads((ARCHIVE/'runs/P1-frozen-sample-01/qualification.json').read_text());parents=json.loads((ARCHIVE/'locks/qualification-parents.json').read_text())['parents'];cases=[dict(spec=copy.deepcopy(sp),goal_stream=p['goal_stream'],execution_source_parent_id=p['execution_source_parent_id'])for p in parents for sp in p['specs']]
    sources=qualification['source_index'];reader=ContactGoalAgent(agent.base,env.proxy.enrich(env.task.extended_observation()),'E',agent.device)
    def trusted(path):
        path=Path(path)
        if not path.is_file():return oldload(path)
        path.resolve().relative_to((ARCHIVE/'runs/P1-frozen-sample-01/sources').resolve());m=json.loads(path.with_suffix('.manifest.json').read_text());s=read(path,m);ref=s['agent_reference'];rp=Path(ref['path']);rp.resolve().relative_to((ARCHIVE/'runs/P1-frozen-sample-01/sources').resolve());s['agent']=read(rp,ref)['agent'];return s
    checkpoint.load=trusted;records=[];keep_ref=qualification['results']
    # Stable original qualified parents: none0,zero2,near4,mid6,stop8,continue9.
    indices=[0,2,4] if mode=='sentinel' else [0,2,4,6,8,9]
    try:
        for idx in indices:
            env.cases=cases
            if mode=='sentinel' and idx==4:
                source=next(s for s in sources if s['case']==4 and s['label']=='actual-release')
                checkpoint.restore(Path(source['file']['path']),reader,env,out);label='actual-qualified-release';ref=keep_ref[idx]['same_state_recovery_reference']
            else:
                source=next((s for s in sources if s['case']==idx and s['label']=='sentinel-handoff'),None)
                if source is not None:checkpoint.restore(Path(source['file']['path']),reader,env,out)
                else:
                    env.reset_cursor=idx;env.reset(idx)
                label='handoff';ref=None
            # Only current actor controls qualification. Archived reader is never a learned source.
            env.agent=agent;env.proxy.agent=agent;env.phase=phase;env.proxy.stochastic=(mode=='P3_sample');env.collection_role='qualification_not_PPO';first=len(env.rows);ph0=len(env.physical_stream);failed=[]
            if hasattr(env,'reference_constant_option'):del env.reference_constant_option
            # Each independent qualification branch has its own fixed stream; caller RNG restored.
            random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
            release=env.task.spec['release'];release_reference=ref
            while not env.task.done:
                from . import benchmark
                if benchmark.STOP_REQUESTED:raise RuntimeError('qualification_budget_stop')
                if mode=='P3_sample' and env.task.spec['enabled']and env.task.spec['distance']>0 and release_reference is None and float(env.e.tether.time)>=release-1e-7:
                    # Current policy's actual release state: native frozen mean from this exact state.
                    branch=EnvBranch(env);original_agent=env.agent;env.agent=reader;env.proxy.agent=reader;env.phase='P2';env.proxy.stochastic=False;env.reference_constant_option=0;env.collection_role='native_recovery_reference';rp=len(env.physical_stream)
                    try:
                        while not env.task.done:env.collect(5)
                        release_reference=quality(env.physical_stream[rp:]);reference_failed=protection(release_reference,'recovery',release_reference)
                        dump(out/f'case-{idx}-native-reference.ptz',dict(physics=env.physical_stream[rp:],quality=release_reference,failed=reference_failed,role='native_actual_same_complete_release_reference_not_PPO'))
                        if reference_failed:failed+=['native_reference_'+v for v in reference_failed]
                    finally:branch.restore();env.agent=original_agent;env.proxy.agent=original_agent
                    if failed:break
                env.collect(5)
            physical=copy.deepcopy(env.physical_stream[ph0:]);actual=copy.deepcopy(env.rows[first:]);q=quality(physical);family=env.task.spec['demand_family'];recph=[p for p in physical if p['force_time']>=release-1e-8]
            if family in ('none','zero'):failed+=protection(q,family)
            if mode=='sentinel' and idx==4:failed+=protection(q,'recovery',ref)
            elif family not in ('none','zero') and recph:
                rec=quality(recph);failed+=protection(rec,'recovery',release_reference)
            if not env.task.done or env.task.boundary:failed.append('incomplete_or_hard_boundary')
            archive=dump(out/f'case-{idx}-actual.ptz',dict(rows=actual,physics=physical,controls=env.task.controls,spec=env.task.spec,role=mode+'_current_actor_qualification_not_training',start_kind=label))
            r=dict(case=idx,family=family,parent_id=env.task.spec['parent_id'],mode=mode,phase=phase,complete=env.task.done,whole=q,failed=failed,archive=archive,real_policy_rows=len(actual),reference=release_reference);records.append(r);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print('OR1_QUAL',mode,idx,'failed',failed,flush=True)
            if failed:break
    finally:
        checkpoint.load=oldload;env.cases=original_cases;snap.restore();env.agent=agent;env.proxy.agent=agent
    assert exact(canonical(portable(env.task),metadata(env.e.sim.model,env.e.sim.data)),start),'qualification changed training Task/history/solver'
    assert exact(global_rng(),start_rng),'qualification changed training RNG'
    assert tree_digest(agent.state())==start_agent,'qualification changed learned actor/Adam'
    expected=3 if mode=='sentinel'else 6;ok=len(records)==expected and all(not r['failed']for r in records)
    result=dict(status='PASS'if ok else 'FAILED_PROTECTION',mode=mode,expected=expected,completed=len(records),records=records,training_Task_rng_actor_restored_exact=True,qualification_rows_not_PPO=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
