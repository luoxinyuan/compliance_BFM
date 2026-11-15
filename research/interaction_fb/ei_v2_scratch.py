"""P1 disposable actual PPO/auxiliary/Adam/IO throughput qualification, not scientific training."""
import os,sys,json,argparse,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import torch,numpy as np,random,subprocess
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.ei_v2_checkpoint import Journal,restore
    from research.interaction_fb.ei_v2_ppo import update
    from research.interaction_fb.ei_v2_auxiliary import ActualSuccessors
    from research.interaction_fb.journal import dump,read
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    from diagnostic_frozen import portable
    from dual_track_task import global_rng
    from research.interaction_fb import benchmark
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    random.seed(660301);np.random.seed(660301);torch.manual_seed(660301);torch.cuda.manual_seed_all(660301)
    reward=json.loads((STAGE/'runs/P1-reward-01/reward.json').read_text());assert reward['status']=='REWARD_QUALIFICATION_PASS'
    gate=json.loads((STAGE/'locks/P1-reward-review.json').read_text());assert gate['status']=='PASS' and gate['reward_file_sha256']
    parents=json.loads((STAGE/'locks/execution-train.json').read_text())['parents'];order=json.loads((STAGE/'locks/training-order.json').read_text())
    ids=order['parent_ids'];by={p['parent_id']:p for p in parents};parents=[by[k] for k in ids]
    env=GoalEnvironment(e,base,c,parents);env.reset(0);install_contact_details(e);install_post_velocity(e)
    agent=ContactGoalAgent(base,env.proxy.enrich(env.task.extended_observation()),ARM,e.standing.device);env.agent=agent;env.proxy.agent=agent;env.collection_role='engineering_on_policy';env.proxy.stochastic=True
    q=STAGE/'runs/P1-frozen-sample-01';qm=json.loads((q/'qualification.json').read_text());keeps=read(q/'qualified-retention-data.ptz',qm['retention']);keep,eps=keeps['keep_by_parent'],keeps['eps'];records=[];runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    def fill(n):
        start=len(env.rows)
        while len(env.rows)-start<n:
            if benchmark.STOP_REQUESTED:raise RuntimeError('BUDGET_STOP_DURING_DISPOSABLE_SCRATCH')
            env.collect(min(5,n-(len(env.rows)-start)))
        return env.rows[start:]
    def signature(metrics):
        return dict(agent=copy.deepcopy(agent.state()),physics=canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),rng=global_rng(),metrics=metrics,last_rows=copy.deepcopy(env.rows[-12:]))
    if RESTORE:
        source=STAGE/f'runs/P1-scratch-{ARM}-01';s=restore(source/'checkpoints/block-mid4',agent,env,out);phase='P3';env.collection_role='engineering_on_policy';agent.selector.requires_grad_(True)
        fill(12);aux=ActualSuccessors(env,phase);block=env.rows[-1024:];t=time.monotonic();metrics=update(agent,block,keep,eps,aux,phase)
        expected_meta=json.loads((source/'scratch.json').read_text())['expected'];expected=read(source/'expected-fourth-update.ptz',expected_meta);actual=signature(metrics)
        checks={k:exact(actual[k],expected[k]) for k in actual};dump(out/'actual-fourth-update.ptz',actual);(out/'restore-fields.json').write_text(json.dumps(checks,indent=2)+'\n');assert all(checks.values()),checks
        result=dict(status='TRAINED_ADAM_NEW_PROCESS_CONTINUATION_AND_UPDATE_EXACT',arm=ARM,checks=checks,update_s=time.monotonic()-t,unique_new_scratch_rows=0,replayed_unique_archive_rows=4096,formal_rows=0)
    else:
        journal=Journal(out/'checkpoints');expected=None
        for block in range(4):
            phase='P2' if block<2 else 'P3';env.phase=phase;agent.selector.requires_grad_(phase=='P3');aux=ActualSuccessors(env,phase)
            start=len(env.rows);tb=time.monotonic();collect=time.monotonic()
            if block==3:
                fill(1012);journal.save('mid4',agent,env,runtime,'scratch_only_mid_fourth_block_with_nonempty_Adam')
                fill(12)
            else:fill(1024)
            collect=time.monotonic()-collect;rows=env.rows[start:];assert len(rows)==1024
            tu=time.monotonic();metrics=update(agent,rows,keep,eps,aux,phase);update_s=time.monotonic()-tu
            if block==3:expected=dump(out/'expected-fourth-update.ptz',signature(metrics))
            tio=time.monotonic();manifest=journal.save(block+1,agent,env,runtime,'scratch_only_discard_before_formal');io_s=time.monotonic()-tio
            record=dict(block=block+1,phase=phase,unique_actual_rows=1024,collection_s=collect,update_s=update_s,checkpoint_s=io_s,wall_s=time.monotonic()-tb,accepted=agent.accepted,rejected=agent.rejected,checkpoint=manifest,metrics=metrics,gpu_peak_bytes=torch.cuda.max_memory_allocated(),optimizer_nonempty=bool(agent.opt.state) and bool(agent.vopt.state));records.append(record)
            (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n');print('SCRATCH_BLOCK',ARM,block+1,phase,record['wall_s'],flush=True)
        result=dict(status='DISPOSABLE_FOUR_FULL_BLOCKS_WRITTEN_RESTORE_PENDING',arm=ARM,records=records,expected=expected,unique_scratch_rows=4096,formal_rows=0,scratch_discard_required=True)
    (out/'scratch.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    global ARM,RESTORE
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['E','I','restoreE','restoreI'],required=True);a=p.parse_args();RESTORE=a.mode.startswith('restore');ARM=a.mode[-1]
    from research.interaction_fb import benchmark
    from research.interaction_fb.ei_v2_counter_io import install
    install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
