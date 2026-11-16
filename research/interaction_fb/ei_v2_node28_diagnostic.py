"""Bounded frozen node28 restore diagnosis: preserve strict original assertions."""
import os,sys,json,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;OLD=STAGE.parent/'contact-goal-ei-v2-20261115';sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc
ARM='E'
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
    from . import ei_v2_checkpoint as cp
    import inspect
    original_exact=cp.exact;failures=[]
    def diagnosed_exact(a,b):
        ok=original_exact(a,b)
        if not ok and isinstance(a,dict) and isinstance(b,dict) and 'qpos' in a and 'qpos' in b:
            frame=inspect.currentframe().f_back
            path=str(frame.f_locals.get('path','unknown'))
            fields=[];raw={}
            for k in sorted(set(a)|set(b)):
                x=a.get(k);y=b.get(k)
                if original_exact(x,y):continue
                rr=dict(name=k,shape_actual=list(getattr(x,'shape',())),shape_expected=list(getattr(y,'shape',())))
                if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.shape==y.shape:
                    xb=x.view(np.uint8).reshape(-1);yb=y.view(np.uint8).reshape(-1)
                    rr['different_bytes']=int(np.count_nonzero(xb!=yb));rr['nbytes']=int(x.nbytes)
                    if x.dtype.kind in 'fiu' and x.size:
                        delta=np.abs(x.astype(np.float64)-y.astype(np.float64));rr['max_abs']=float(np.nanmax(delta));rr['different_elements']=int(np.count_nonzero(x!=y))
                    raw['actual_'+k]=x.copy();raw['expected_'+k]=y.copy()
                fields.append(rr)
            failures.append(dict(checkpoint=path,fields=fields))
            np.savez_compressed(out/('mismatch-'+str(len(failures))+'.npz'),**raw)
            (out/'first-difference.json').write_text(json.dumps(failures,indent=2)+'\n')
        return ok
    cp.exact=diagnosed_exact
    source=STAGE/'runs/formal-E-05/checkpoints/block-node-28000'
    try:
        restore(source,agent,env,out)
        before=tree_digest(agent.state())
        print('DIAGNOSTIC_TRAIN_NODE_RESTORED',env.next_cursor,flush=True)
        result=qualify(env,agent,out/'repeat-node28','sentinel','P3')
        assert tree_digest(agent.state())==before
        return dict(status='RESTORE_AND_QUALIFICATION_PASS',result=result,formal_rows_added=0,optimizer_updates=0)
    except BaseException as exc:
        payload=dict(status='RESTORE_DIAGNOSTIC_FAILED',error=repr(exc),mismatches=failures,formal_rows_added=0,optimizer_updates=0)
        (out/'diagnostic-result.json').write_text(json.dumps(payload,indent=2)+'\n')
        raise
    finally:cp.exact=original_exact

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from . import benchmark
    from .ei_v2_counter_io import install
    from .ei_v2_counter_lock import atomic_write
    original_write=benchmark.write
    def counter(path,value):
        if Path(path).name!='physics-counter.json':return original_write(path,value)
        assert value['reserved']<=20000,'DIAGNOSTIC_TICK_CAP'
        atomic_write(path,value)
    benchmark.write=counter;install(benchmark);benchmark.study=study
    sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
