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
    import inspect,mujoco
    from . import ei_v2_checkpoint as cp
    from .ei_v2_quality import EnvBranch,quality
    from dual_track_task import restore_global_rng
    # Experimental restoration includes the TWO archived arrays, never masks them.
    # Main production restore remains unchanged until this intervention passes.
    original_load=cp.load
    def trusted(path):
        path=Path(path)
        if not path.is_file():return original_load(path)
        path.resolve().relative_to((OLD/'runs/P1-frozen-sample-01/sources').resolve())
        mm=json.loads(path.with_suffix('.manifest.json').read_text());ss=read(path,mm)
        rr=ss['agent_reference'];ss['agent']=read(Path(rr['path']),rr)['agent'];return ss
    ns=dict(cp.__dict__);ns['load']=trusted
    src=inspect.getsource(cp.restore)
    assert src.count("padding=('efc_J_colind','efc_J_rowsuper','efc_JT')")==1
    src=src.replace("padding=('efc_J_colind','efc_J_rowsuper','efc_JT')", "padding=('efc_J_colind','efc_J_rowsuper','efc_JT','efc_J_rowadr','efc_J_rownnz')")
    exec(compile(src,'candidate_restore_full_arrays','exec'),ns);candidate_restore=ns['restore']
    q=json.loads((OLD/'runs/P1-frozen-sample-01/qualification.json').read_text())
    parents=json.loads((OLD/'locks/qualification-parents.json').read_text())['parents']
    env.cases=[dict(spec=copy.deepcopy(sp),goal_stream=p['goal_stream'],execution_source_parent_id=p['execution_source_parent_id']) for p in parents for sp in p['specs']]
    names=('efc_J_rowadr','efc_J_rownnz');results=[]
    cases=[(0,'sentinel-handoff'),(4,'reward-local_follow'),(8,'reward-stop_crossing')]
    for ci,label in cases:
        source=next(x for x in q['source_index'] if x['case']==ci and x['label']==label)
        folder=out/('case-'+str(ci));folder.mkdir()
        candidate_restore(Path(source['file']['path']),agent,env,folder)
        env.agent=agent;env.proxy.agent=agent;env.collection_role='frozen_restore_intervention_not_PPO'
        initial=portable(env.task);mm=metadata(e.sim.model,e.sim.data);assert mm['is_sparse'] is False and mm['solver']==int(mujoco.mjtSolver.mjSOL_NEWTON)
        branch=EnvBranch(env);rng=copy.deepcopy(global_rng());records=[];baseline=None
        for mode in ('original','efc_J_rowadr:zero','efc_J_rowadr:random','efc_J_rownnz:zero','efc_J_rownnz:random'):
            branch.restore();restore_global_rng(rng)
            for name in names:
                a=getattr(e.sim.data,name);ref=initial['mjdata_arrays'][name]
                if mode==name+':zero':a[:]=0
                elif mode==name+':random':
                    lo=min(0,int(ref.min(initial=0)));hi=max(1,int(ref.max(initial=0)))
                    a[:]=np.random.default_rng(660301+ci).integers(lo,hi+1,size=a.shape,dtype=a.dtype)
            now=canonical(portable(env.task),metadata(e.sim.model,e.sim.data));was=canonical(initial,mm)
            for name in names:now['mjdata_arrays'][name]=was['mjdata_arrays'][name]
            assert exact(now,was),'intervention changed another effective field'
            first=len(env.physical_stream);r0=len(env.rows);controls=len(env.task.controls);t0=float(e.tether.time)
            while float(e.tether.time)<t0+3.-1e-7:
                with torch.no_grad():env.collect(min(5,round((t0+3.-float(e.tether.time))/.02)))
                assert env.task.boundary is None
            ph=env.physical_stream[first:];rows=env.rows[r0:]
            result=dict(physics=tree_digest(ph),controls=tree_digest(env.task.controls[controls:]),rows=tree_digest(rows),rng=tree_digest(global_rng()),quality=quality(ph))
            if baseline is None:baseline=result
            assert result==baseline,('array interference changes dynamics',ci,mode)
            rec=dict(mode=mode,exact_to_original=True,seconds=float(e.tether.time)-t0,physics_ticks=len(ph),actual_controls=len(rows),active_ticks=sum(p['active'] for p in ph),max_force_N=max(float(np.linalg.norm(p['force']))for p in ph),initial_nefc=mm['counters']['nefc'],quality=result['quality'])
            records.append(rec);print('ARRAY_INTERVENTION',ci,mode,'PASS',rec['seconds'],flush=True)
        branch.restore();restore_global_rng(rng)
        results.append(dict(case=ci,label=label,arrays=list(names),records=records))
        (out/'intervention-progress.json').write_text(json.dumps(results,indent=2)+'\n')
    assert any(r['max_force_N']>0 for p in results for r in p['records'])
    result=dict(status='PASS',mode='two-array-dense-Newton-physical-intervention',results=results,formal_rows_added=0,optimizer_updates=0,production_restore_unchanged=True)
    (out/'intervention-result.json').write_text(json.dumps(result,indent=2)+'\n');return result

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
