"""Signed OR1 bounded new on-policy engineering, never formal32k or old-rollout reuse."""
import os,sys,argparse,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;OLD=STAGE.parent/'contact-goal-ei-v2-20261115';sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

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
    from .ei_v2_or1_ppo import OR1Agent,update
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
    agent=OR1Agent(base,env.proxy.enrich(env.task.extended_observation()),ARM,e.standing.device);env.agent=agent;env.proxy.agent=agent;env.collection_role='engineering_on_policy';env.proxy.stochastic=True
    qm=json.loads((OLD/'runs/P1-frozen-sample-01/qualification.json').read_text());kd=read(OLD/'runs/P1-frozen-sample-01/qualified-retention-data.ptz',qm['retention']);keep,eps=kd['keep_by_parent'],kd['eps']
    assert json.loads((STAGE/'runs/OR1-D0-01/result.json').read_text())['status']=='OR1_D0_PASS_FOR_NEW_ENGINEERING_ONLY'
    old_native=0;carried_rows=0
    if FAILURE_RESUME:
        failed=RESUME_PACKAGE or STAGE/'runs/OR1-D1-E-01/failed-sampling-packages/block-failed-15'
        assert failed.resolve().is_relative_to((STAGE/'runs').resolve())
        fm=json.loads((failed/'manifest.json').read_text());assert fm['complete'] and fm['reason']=='failed_sampling_not_complete_training_block'
        expected_rows=int(fm['actual_50Hz']);assert 0<expected_rows<1024
        restored=restore(failed,agent,env,out)
        assert ARM=='E' and PART=='D1' and agent.actor_accepted==0 and agent.aux_accepted==0 and not agent.opt.state and not agent.vopt.state
        assert len(env.rows)==env.next_cursor==expected_rows and all(r['eligible_for_joint_PPO'] for r in env.rows)
        carried_rows=expected_rows
        (out/'verified-carried-on-policy-prefix.json').write_text(json.dumps(dict(rows=carried_rows,source_manifest_sha256=__import__('hashlib').sha256((failed/'manifest.json').read_bytes()).hexdigest(),original_policy_unchanged=True,model_Adam_RNG_full_physical_restore_exact=True,already_charged_in_failed_job=True,continuation_of_same_first_rollout_not_old_policy_replay=True),indent=2)+'\n')
    if PART=='D2' and not RESTORE:
        src=SOURCE_RUN or STAGE/f'runs/OR1-D1-{ARM}-01'
        assert src.resolve().is_relative_to((STAGE/'runs').resolve())
        assert json.loads((src/'result.json').read_text())['status']=='OR1_D1_TWO_NEW_P2_BLOCKS_PASS';restore(src/'checkpoints/block-2',agent,env,out);old_native=env.next_cursor;env.phase='P3'
        sample_qualification=qualify(env,agent,out/'P3-sample-entry','P3_sample','P3');assert sample_qualification['status']=='PASS','P3_SAMPLE_QUALIFICATION_FAILED'
    initial_cursor=env.next_cursor;prior=json.loads((STAGE/'ledger.json').read_text())['new_scratch_50Hz'];records=[];journal=Journal(out/'checkpoints');journal.save(0,agent,env,runtime,'OR1_initial_or_phase_resume_qualification_only')
    def count(final=False,reserved=None):
        actual=0 if RESTORE else env.next_cursor-initial_cursor
        v=dict(actual_scratch_rows=actual,reserved_scratch_rows=actual if reserved is None else reserved,exact_final=final,restore_validation=RESTORE)
        p=out/'native-count.pending';p.write_text(json.dumps(v,indent=2)+'\n');os.replace(p,out/'native-count.json')
    def fill(n):
        count(False,env.next_cursor-initial_cursor+n);begin=len(env.rows)
        assert RESTORE or prior+(env.next_cursor-initial_cursor)+n<=8192,'new OR1 sampling hard cap'
        try:
            while len(env.rows)-begin<n:
                if benchmark.STOP_REQUESTED:raise RuntimeError('OR1_STOP_OR_BUDGET_REQUEST')
                new,_=env.collect(min(5,n-len(env.rows)+begin))
                for r in new:r['actual_actor_policy_version']=agent.actor_accepted
        except BaseException:
            count(True);failure=Journal(out/'failed-sampling-packages');failure.save('failed-'+str(env.next_cursor),agent,env,runtime,'failed_sampling_not_complete_training_block');raise
        count(True);return env.rows[begin:]
    def signature(metrics):
        return dict(agent=copy.deepcopy(agent.state()),physics=canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),rng=global_rng(),metrics=metrics,last_rows=copy.deepcopy(env.rows[-12:]))
    if RESTORE:
        src=SOURCE_RUN or STAGE/f'runs/OR1-D2-{ARM}-01'
        assert src.resolve().is_relative_to((STAGE/'runs').resolve())
        sm=json.loads((src/'result.json').read_text());restore(src/'checkpoints/block-mid-last',agent,env,out);initial_cursor=env.next_cursor
        fill(12);aux=ActualSuccessors(env,'P3');mt=update(agent,env.rows[-1024:],keep,eps,aux,'P3');actual=signature(mt);expected=read(src/'expected-last-update.ptz',sm['expected']);checks={k:exact(actual[k],expected[k])for k in actual};dump(out/'actual-restored-update.ptz',actual);assert all(checks.values()),checks
        result=dict(status='OR1_ACCEPTED_ACTOR_ADAM_PHYSICS_RESTORE_EXACT',arm=ARM,checks=checks,new_on_policy_rows=0,actual_validation_native_continuation=12,replayed_physics_not_training=True)
    else:
        phase='P2' if PART=='D1'else 'P3';env.phase=phase;agent.selector.requires_grad_(phase=='P3')
        for block in range(1,3):
            t=time.monotonic();b0=(0 if FAILURE_RESUME and block==1 else len(env.rows));accepted0=agent.actor_accepted;tu=time.monotonic()
            if block==2:fill(1012);journal.save('mid-last',agent,env,runtime,'real_accepted_Actor_Adam_mid_second_block');fill(12)
            else:fill(1024-carried_rows if FAILURE_RESUME and block==1 else 1024)
            collected=time.monotonic()-tu;rows=env.rows[b0:];assert len(rows)==1024
            journal.save('pending-'+str(block),agent,env,runtime,'complete_sampling_before_pending_update_not_completed_training_step')
            aux=ActualSuccessors(env,phase);tu=time.monotonic()
            try:mt=update(agent,rows,keep,eps,aux,phase)
            except BaseException:
                failure=Journal(out/'failed-update-packages');failure.save('failed-'+str(block),agent,env,runtime,'failed_not_committed_training_block');dump(out/f'failed-{block}-grads.ptz',{k:p.grad for k,p in agent.modules.named_parameters()});raise
            update_s=time.monotonic()-tu;actor_steps=agent.actor_accepted-accepted0
            assert actor_steps>0,'OR1_BLOCK_NO_ACCEPTED_ACTOR_STOP'
            # Same actual states: only describe changes; not new physics evidence.
            from .ei_v2_ppo import device_obs,kl
            deltas=[];selector_deltas=[]
            with torch.no_grad():
                for r in rows:
                    mu=agent.mean(device_obs(r['observation'],agent.device),r['z'].to(agent.device),torch.tensor([r['option']],device=agent.device));deltas.append((mu-r['old_mu'].to(agent.device)).cpu().numpy())
                    if r['cat_boundary']:
                        old=r['old_logits'].to(agent.device).softmax(-1);new=agent.logits(device_obs(r['selector_obs'],agent.device),r['candidate_first_z'].to(agent.device)).softmax(-1);selector_deltas.append(float(abs(new-old).max()))
            if block==2:expected=dump(out/'expected-last-update.ptz',signature(mt))
            ti=time.monotonic();manifest=journal.save(block,agent,env,runtime,'complete_OR1_engineering_block_not_formal');io=time.monotonic()-ti
            m=dict(block=block,phase=phase,new_rows=1024-(carried_rows if block==1 else 0),block_rows=1024,collection_s=collected,update_s=update_s,checkpoint_s=io,hot_s=time.monotonic()-t,actor_steps_this_block=actor_steps,actor_accepted_total=agent.actor_accepted,aux_accepted_total=agent.aux_accepted,selector_probability_max_delta=max(selector_deltas or[0.]),actual_selector_draws=len(selector_deltas),mean_delta_RMS=float(np.sqrt(np.mean(np.concatenate(deltas)**2))),metrics=mt,checkpoint=manifest)
            records.append(m);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print('OR1_BLOCK',ARM,PART,block,'actor',actor_steps,'hot',m['hot_s'],flush=True)
            if block==1:
                q=qualify(env,agent,out/'first-accepted-sentinel','sentinel',phase);assert q['status']=='PASS','CURRENT_ACTOR_SENTINEL_PROTECTION_FAILED'
        if PART=='D2':
            assert len(env.episodes)>=6,'UNDERPOWERED_COMPLETE_EPISODES_STOP'
            assert max(m['selector_probability_max_delta']for m in records)>=1e-6,'SELECTOR_CHANGE_BELOW_LOCKED_OBSERVABILITY_STOP'
        count(True)
        result=dict(status='OR1_'+PART+'_TWO_NEW_BLOCKS_PASS_RESTORE_PENDING'if PART=='D2'else 'OR1_D1_TWO_NEW_P2_BLOCKS_PASS',arm=ARM,phase=phase,records=records,new_rows=env.next_cursor-initial_cursor,carried_same_policy_rows=carried_rows,completed_block_rows=2048,training_total_rows=env.next_cursor,complete_episodes=len(env.episodes),new_parents_visited=len(set(bytes(r['provenance']['parent_sha256'].reshape(-1).tolist()).hex()for r in env.rows)),expected=expected,formal_rows=0,dev_episodes=0,sealed_read=False,discard_before_any_formal_training=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    global ARM,PART,RESTORE,FAILURE_RESUME,RESUME_PACKAGE,SOURCE_RUN
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',required=True,choices=['D1E','D1I','D2E','D2I','restoreE','restoreI','D1E_resume']);p.add_argument('--resume-package',type=Path);p.add_argument('--source-run',type=Path);a=p.parse_args();RESUME_PACKAGE=a.resume_package;SOURCE_RUN=a.source_run;FAILURE_RESUME=a.mode.endswith('_resume');ARM=('E' if FAILURE_RESUME else a.mode[-1]);RESTORE=a.mode.startswith('restore');PART='D2'if RESTORE else a.mode[:2]
    assert RESUME_PACKAGE is None or FAILURE_RESUME
    assert SOURCE_RUN is None or RESTORE or PART=='D2'
    from . import benchmark
    from .ei_v2_counter_io import install
    original_write=benchmark.write
    def atomic_counter(path,data):
        if Path(path).name!='physics-counter.json':return original_write(path,data)
        tmp=Path(path).with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
    benchmark.write=atomic_counter;install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
