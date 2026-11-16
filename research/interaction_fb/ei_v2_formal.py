"""Formal E/I v2 driver: unchanged qualified OR1 update, fresh initialization."""
import os,sys,argparse,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;OLD=STAGE.parent/'contact-goal-ei-v2-20261115';sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def result_payload(path):
    """Benchmark stores the study under an envelope; direct fixtures are bare."""
    saved=json.loads(Path(path).read_text())
    if 'study' in saved:
        assert isinstance(saved['study'],dict) and saved['status']==saved['study']['status']
        return saved['study']
    return saved

def study(e,base,c,out):
    import torch,numpy as np,random,subprocess
    from native_env import install as contacts
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from dual_track_task import global_rng
    from .ei_v2_sources import install
    from .ei_v2_env import GoalEnvironment
    from .ei_v2_formal_update import OR1Agent,update
    from .ei_v2_or1_qualification import run as qualify
    from .ei_v2_checkpoint import Journal,restore
    from .ei_v2_auxiliary import ActualSuccessors
    from .journal import read,dump
    from . import benchmark
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contacts();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
    ps=json.loads((STAGE/'locks/execution-train.json').read_text())['parents'];by={p['parent_id']:p for p in ps};ps=[by[k]for k in json.loads((STAGE/'locks/training-order.json').read_text())['parent_ids']]
    env=GoalEnvironment(e,base,c,ps);env.reset(0);install_contact_details(e);install_post_velocity(e)
    agent=OR1Agent(base,env.proxy.enrich(env.task.extended_observation()),ARM,e.standing.device);env.agent=agent;env.proxy.agent=agent;env.collection_role='formal_on_policy';env.proxy.stochastic=True
    qm=json.loads((OLD/'runs/P1-frozen-sample-01/qualification.json').read_text());kd=read(OLD/'runs/P1-frozen-sample-01/qualified-retention-data.ptz',qm['retention']);keep,eps=kd['keep_by_parent'],kd['eps']
    prior=STAGE.parent/'contact-goal-ei-v2-or1-20261115'
    assert json.loads((prior/'ledger.json').read_text())['status']=='OR1_D0_D1_D2_COMPLETE_OR6_NOT_CLEARED_NO_FORMAL_RUN'
    # No engineering model/optimizer/rollout is loaded into formal training.
    assert not agent.opt.state and not agent.vopt.state and agent.actor_accepted==0
    journal=Journal(out/'checkpoints');records=[];nodes=[];first_pass=set();cap_hits=0
    started=time.monotonic();resume_state=None
    if RESUME_PLAN is not None:
        import hashlib
        plan=json.loads(RESUME_PLAN.read_text())
        assert plan['arm']==ARM and plan['phase']=='P2' and plan['rows']==1095
        for name,digest in plan['evidence_sha256'].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
        oldstop=json.loads(Path(plan['stop_json']).read_text())
        assert oldstop['error']=="RuntimeError('SUPERVISOR_REQUESTED_COMPLETE_STOP')"
        assert oldstop['rows']==plan['rows'] and oldstop['records'][-1]['end_row']==1024
        from .ei_v2_checkpoint import load
        resume_state=restore(Path(plan['checkpoint']),agent,env,out)
        assert env.next_cursor==1095 and len(env.rows)==1095 and agent.actor_accepted==2
        assert env.collection_role=='formal_on_policy'
        committed=load(Path(plan['committed_checkpoint']))
        assert tree_digest(committed['agent'])==tree_digest(resume_state['agent']), 'partial block updated unexpectedly'
        del committed
        # The preexisting sentinels belong to this exact unchanged agent.
        q=json.loads(Path(plan['sentinel_result']).read_text())
        assert q['status']=='PASS' and q['training_Task_rng_actor_restored_exact']
        assert [v['case'] for v in q['records']]==[0,2,4] and all(not v['failed'] for v in q['records'])
        records=oldstop['records'];nodes=oldstop['nodes'];first_pass=set(plan['first_accepted_pass']);cap_hits=0
        assert not nodes and first_pass=={'P2'}
        # Two reproducible 12-row continuation branches, discarded after comparison.
        # These are restore checks, not additional PPO samples.
        def check_branch():
            env.collection_role='restore_check_not_PPO'
            new=[]
            while len(new)<12:
                with torch.no_grad():part,_=env.collect(min(5,12-len(new)))
                new.extend(part)
            assert len(new)==12
            return dict(agent=tree_digest(agent.state()),rng=tree_digest(global_rng()),
                        physics=tree_digest(canonical(portable(env.task),metadata(e.sim.model,e.sim.data))),
                        rows=tree_digest(new))
        a=check_branch();restore(Path(plan['checkpoint']),agent,env,out)
        b=check_branch();assert a==b, 'formal resumed continuation mismatch'
        restore(Path(plan['checkpoint']),agent,env,out)
        from .checkpoint import jsonwrite
        jsonwrite(out/'resume-qualification.json',dict(status='PASS',rows=1095,pending_rows=71,
                  block_start=1024,block_target=2048,new_formal_rows=0,verification_rows=24,
                  exact_continuation=a,source_plan_sha256=hashlib.sha256(RESUME_PLAN.read_bytes()).hexdigest()))
        print('FORMAL_RESUME_EXACT',ARM,1095,'pending',71,flush=True)
    initial=journal.save('initial',agent,env,runtime,'formal_restored_complete_state' if resume_state is not None else 'formal_fresh_initial_not_OR1_scratch')
    from .checkpoint import jsonwrite
    def progress():
        jsonwrite(out/'progress.json',dict(status='RUNNING',arm=ARM,formal_rows=env.next_cursor,records=records,nodes=nodes,first_accepted_pass=sorted(first_pass),elapsed_hot_s=time.monotonic()-started))
        jsonwrite(out/'native-count.json',dict(actual_formal_rows=env.next_cursor,reserved_formal_rows=env.next_cursor,exact_final=False,engineering_rows=0))
    def guard():
        import shutil
        if benchmark.STOP_REQUESTED:raise RuntimeError('SUPERVISOR_REQUESTED_COMPLETE_STOP')
        policy=json.loads((STAGE/'locks/resource-policy-formal-v1.json').read_text())
        assert shutil.disk_usage(STAGE).free>=(policy['disk_free_floor_GiB']+policy['pending_atomic_peak_reservation_GiB'])*2**30,'DISK_STOP_BEFORE_ATOMIC_RESERVE'
    def sentinel(label,phase,fixed_node=False):
        nonlocal cap_hits
        result=qualify(env,agent,out/('sentinel-'+label),'sentinel',phase)
        assert result['status']=='PASS','FORMAL_TRAIN_SENTINEL_FAILED'
        from .ei_v2_ppo import retention
        with torch.no_grad():_,violation=retention(agent,keep,eps)
        if fixed_node:
            cap_hits=cap_hits+1 if agent.retention_lambda>=3 and float(violation)>1 else 0
            assert cap_hits<2,'RETAIN_MAX_TWO_CHECKPOINTS'
        return result
    def node(n,phase):
        manifest=journal.save('node-'+str(n),agent,env,runtime,'fixed_node_real_state_no_extra_optimizer_update')
        q=sentinel('node-'+str(n),phase,fixed_node=True)
        nodes.append(dict(rows=n,phase=phase,checkpoint=manifest,sentinel=q['status'],actor_accepted=agent.actor_accepted));progress()
    try:
        for phase,begin,end in [('P2',0,12000),('P3',12000,32000)]:
            assert env.next_cursor==begin or (resume_state is not None and phase=='P2' and env.next_cursor==1095)
            env.phase=phase;agent.selector.requires_grad_(phase=='P3')
            if phase=='P3':
                episodes=env.episodes
                if env.task.done:episodes=episodes+[dict(spec=env.task.spec,boundary=env.task.boundary)]
                parents={x['spec']['parent_id']for x in episodes};families={x['spec']['demand_family']for x in episodes}
                release_times={j:x['spec']['release'] for j,x in enumerate(episodes)}
                released_rows=sum((not env.physical_stream[r['physics_begin']]['active']) and env.physical_stream[r['physics_begin']]['force_time']>=release_times[r['episode']] for r in env.rows if r['episode'] in release_times)
                assert released_rows>0,'G08_NO_ACTUAL_RELEASED_COVERAGE'
                coverage=dict(released_rows=released_rows,parents=len(parents),families=sorted(families),complete_episodes=len(episodes),actor_accepted=agent.actor_accepted)
                jsonwrite(out/'G08-coverage.json',coverage)
                assert len(parents)>=8 and {'none','zero','near','mid','stop','continue'}<=families and agent.actor_accepted>0,'G08_COVERAGE_OR_EFFECTIVE_UPDATE'
                q=qualify(env,agent,out/'P3-sample-entry','P3_sample','P3');assert q['status']=='PASS','P3_ENTRY_SAMPLING_FAILED'
            block=max((r['block'] for r in records if r['phase']==phase),default=0)
            while env.next_cursor<end:
                guard();block+=1
                # Resume retains the original pending block boundary, not cursor+1024.
                b0=max((r['end_row'] for r in records if r['phase']==phase),default=begin)
                assert b0<=env.next_cursor<min(end,b0+1024)
                target=min(end,b0+1024);t0=time.monotonic();a0=agent.actor_accepted
                # Intermediate safety/checkpoint nodes do not split the PPO block.
                while env.next_cursor<target:
                    guard()
                    milestone=next((n for n in (4000,8000,12000,20000,28000,32000) if env.next_cursor<n<=target),target)
                    with torch.no_grad():new,_=env.collect(min(5,milestone-env.next_cursor))
                    for row in new:row['actual_actor_policy_version']=agent.actor_accepted
                    if env.next_cursor%100==0:progress()
                    if env.next_cursor==milestone and milestone<target:node(milestone,phase)
                collection_s=time.monotonic()-t0
                pending=journal.save(phase+'-pending-'+str(block),agent,env,runtime,'complete_formal_rollout_before_update')
                tu=time.monotonic();metrics=update(agent,env.rows[b0:],keep,eps,ActualSuccessors(env,phase),phase);update_s=time.monotonic()-tu
                ti=time.monotonic();manifest=journal.save(phase+'-'+str(block),agent,env,runtime,'complete_formal_update');io=time.monotonic()-ti
                records.append(dict(phase=phase,block=block,start_row=b0,end_row=env.next_cursor,rows=env.next_cursor-b0,collection_s=collection_s,update_s=update_s,commit_IO_s=io,actor_steps=agent.actor_accepted-a0,actor_accepted_total=agent.actor_accepted,metrics=metrics,checkpoint=manifest))
                progress();print('FORMAL_BLOCK',ARM,phase,block,env.next_cursor,agent.actor_accepted,flush=True)
                if phase not in first_pass and agent.actor_accepted>a0:
                    q=sentinel('first-accepted-'+phase,phase);first_pass.add(phase);progress()
                if env.next_cursor in (4000,8000,12000,20000,28000,32000):node(env.next_cursor,phase)
            assert phase in first_pass,'NO_ACCEPTED_ACTOR_IN_PHASE'
        jsonwrite(out/'final-coverage.json',dict(complete_episodes=env.episodes,formal_rows=env.next_cursor,actor_accepted=agent.actor_accepted))
        result=dict(status='FORMAL_32K_COMPLETE_EVALUATION_PENDING',arm=ARM,formal_rows=env.next_cursor,records=records,nodes=nodes,dev_episodes=0,sealed_read=False)
        jsonwrite(out/'native-count.json',dict(actual_formal_rows=env.next_cursor,reserved_formal_rows=env.next_cursor,exact_final=True,engineering_rows=0));jsonwrite(out/'result.json',result);return result
    except BaseException as exc:
        # A partially updated block is never silently rerun as a fresh full update.
        failure=Journal(out/'stop-packages');m=failure.save('stop-'+str(env.next_cursor),agent,env,runtime,'incomplete_formal_stage_preserve_pending_update_and_failure')
        jsonwrite(out/'formal-stop.json',dict(status='STOPPED',error=repr(exc),rows=env.next_cursor,checkpoint=m,records=records,nodes=nodes,automatic_resume=False))
        raise

def main():
    global ARM,RESUME_PLAN
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['E','I'],required=True);p.add_argument('--resume-plan',type=Path);a=p.parse_args();ARM=a.mode;RESUME_PLAN=a.resume_plan
    from . import benchmark
    from .ei_v2_counter_io import install
    original_write=benchmark.write
    def atomic_counter(path,data):
        if Path(path).name!='physics-counter.json':return original_write(path,data)
        if data['reserved']>int(os.environ['EI_FORMAL_PHYSICS_CAP']):
            raise RuntimeError('FORMAL_PHYSICS_RESERVATION_CAP')
        from .ei_v2_counter_lock import atomic_write
        atomic_write(path,data)
    benchmark.write=atomic_counter;install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
