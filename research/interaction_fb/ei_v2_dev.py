"""Fixed-terminal independent Dev execution. No learning or checkpoint selection.
The train runtime is immutable; this entry adds only authorized evaluation.
"""
import os,sys,json,copy,time,random,hashlib,argparse,subprocess,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def load_core(path):
    """Original lossless core decoder and hashes; replay segments are not needed
    to evaluate frozen weights from fresh, actual native source prefixes.
    This does NOT claim to restore a training Task/optimizer continuation.
    """
    from . import delta_tree
    path=Path(path);chain=[];seen=set();cursor=path
    while True:
        assert cursor.name not in seen;seen.add(cursor.name)
        m=json.loads((cursor/'manifest.json').read_text());assert m['complete'] and m['format']=='EI-v2-lossless-v1'
        chain.append((cursor,m))
        if m['previous'] is None:break
        assert Path(m['previous']).name==m['previous'];cursor=path.parent/m['previous']
    state=None
    for p,m in reversed(chain):state=delta_tree.load(p/'core.delta',m['files']['core.delta'],state)
    return state,dict(checkpoint=str(path),manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),core_sha256=chain[0][1]['files']['core.delta']['sha256'],rows=chain[0][1]['actual_50Hz'],training_runtime=chain[0][1]['runtime'],core_only_frozen_weight_read=True)


def reverse_goal(stream):
    return '_TO_'.join('HOLD' if s=='FOLLOW' else 'FOLLOW' for s in stream.split('_TO_'))


def plan(stage):
    """Expand only the already-locked 100-case list, without reading sealed."""
    stage=Path(stage);locked=json.loads((stage/'locks/formal_budget_manifest_enumeration.json').read_text())
    for name,d in locked['source_sha'].items():assert hashlib.sha256((stage/'locks'/name).read_bytes()).hexdigest()==d
    parents=json.loads((stage/'locks/execution-dev.json').read_text())['parents'];by={p['parent_id']:p for p in parents}
    result=[]
    for i,r in enumerate(locked['dev_cases']):
        p=by[r['parent_id']];sp=copy.deepcopy(p['specs'][r['spec_index']]);assert sp['split']=='dev'
        goal=p['goal_stream']
        if r['role']=='alternate_public_goal':goal=reverse_goal(goal)
        if r['method']=='BC500':assert r['role']=='FOLLOW_reference';goal='FOLLOW'
        result.append(dict(index=i,**r,goal_stream=goal,case=dict(spec=sp,goal_stream=goal,execution_source_parent_id=p['execution_source_parent_id'])))
    assert len(result)==100 and len(parents)==16
    return result


def stage_metrics(env,ph,rows):
    import numpy as np
    from .ei_v2_quality import quality
    if not ph:return dict(status='UNMEASURED_EMPTY')
    spec=env.task.spec
    instructions=env.instructions;points=[]
    for p in ph:
        t=p['force_time'];ins=next((r for r in reversed(instructions) if r['time']<=t+1e-8),None)
        if ins is None:continue
        mode='FOLLOW' if ins['goal_mode']==0 else 'HOLD';error=np.asarray(p['anchor'])-p['wrist']
        target=ins['goal_position_world'];hold=None if target is None else float(np.linalg.norm(np.asarray(p['wrist'])-target))
        if not p['active']:phase='precontact' if t<spec['onset'] else 'released'
        elif t<spec.get('stop_time',spec['onset']+spec['ramp'])-1e-8:phase='moving'
        elif t<spec.get('stop_time',spec['onset']+spec['ramp'])+spec.get('deceleration',0)-1e-8:phase='decelerating'
        else:phase='holding'
        points.append((p,mode,phase,error,hold))
    out={}
    for phase in ('precontact','moving','decelerating','holding','released'):
        for mode in ('FOLLOW','HOLD'):
            v=[r for r in points if r[1:3]==(mode,phase)]
            if not v:continue
            pp=[r[0]for r in v];f=np.asarray([p['force']for p in pp]);err=np.asarray([r[3]for r in v]);hh=[r[4]for r in v if r[4]is not None]
            er=float(np.sqrt(np.mean(np.sum(err**2,axis=1))));fr=float(np.sqrt(np.mean(np.sum(f**2,axis=1))))
            k=f'{phase}:{mode}';rr=dict(n_ticks=len(v),duration_s=.005*len(v),error_xyz_rms_m=np.sqrt(np.mean(err**2,axis=0)).tolist(),signed_error_xyz_m=err.mean(0).tolist(),error_3D_rms_m=er,force_xyz_rms_N=np.sqrt(np.mean(f**2,axis=0)).tolist(),force_3D_rms_N=fr,saturation_fraction=float(np.mean(np.linalg.norm(f,axis=1)>=10-1e-6)),quality=quality(pp),J=None)
            if hh:rr.update(public_goal_error_rms_m=float(np.sqrt(np.mean(np.square(hh)))),public_goal_error_p95_m=float(np.quantile(hh,.95)))
            if phase in ('moving','decelerating','holding'):
                if mode=='FOLLOW':
                    fam=spec['demand_family'];ef=.04 if fam=='near' else .08;ff=4 if fam=='near' else (6.5 if fam=='stop' and phase=='holding' else 7)
                    rr['J']=.5*er/ef+.5*fr/ff
                else:rr['J']=.5*rr['public_goal_error_rms_m']/.04+.5*fr/7
            out[k]=rr
    goals=[]
    for mode in ('FOLLOW','HOLD'):
        js=[r['J']for k,r in out.items() if k.endswith(':'+mode) and r['J']is not None]
        if js:goals.append(float(np.mean(js)))
    return dict(windows=out,J=float(np.mean(goals))if goals else None,aggregation='RMS within stage/Goal; nonempty stages equal then Goals equal; parent/sibling aggregation remains offline')


def study(e,base,c,out):
    import torch,numpy as np
    from native_env import install as contacts
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,canonical,metadata,exact
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from .ei_v2_sources import install
    from .ei_v2_env import GoalEnvironment
    from .ei_v2_formal_update import OR1Agent
    from .ei_v2_agent import ContactGoalAgent
    from .ei_v2_quality import EnvBranch,quality,protection
    from .evaluation import summarize
    from .journal import dump
    from .checkpoint import jsonwrite,sha
    from . import benchmark
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    from dual_track_task import global_rng,restore_global_rng
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_grad_enabled(False)
    contacts();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    records=plan(STAGE);selected=[r for r in records if METHOD is None or r['method']==METHOD]
    parent=json.loads((STAGE/'locks/execution-dev.json').read_text())['parents']
    env=GoalEnvironment(e,base,c,parent);env.reset(0);install_contact_details(e);install_post_velocity(e)
    obs=env.proxy.enrich(env.task.extended_observation());agents={};identities={}
    runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for method in dict.fromkeys(r['method']for r in selected):
        arm='I'if method=='I-final'else'E';a=OR1Agent(base,obs,arm,e.standing.device)
        if method in ('E-final','I-final'):
            src=STAGE/f'runs/formal-{arm}-06/checkpoints/block-node-32000';state,identity=load_core(src);assert identity['rows']==32000
            a.load(state['agent']);assert tree_digest(a.state())==tree_digest(state['agent']);identities[method]=dict(**identity,agent_sha256=tree_digest(a.state()),actor_accepted=a.actor_accepted)
            episodes=state['own']['episodes'];identities[method]['completed_training_parent_ids']=sorted({r['spec']['parent_id'] for r in episodes});identities[method]['completed_training_episodes']=len(episodes)
            del state
        else:identities[method]=dict(initial_function='original frozen actor + zero FiLM',agent_sha256=tree_digest(a.state()),command=('constant standing'if method=='native-zero-U'else'Goal lookup HOLD-standing FOLLOW-arm15'))
        a.modules.eval().requires_grad_(False);a.value.eval().requires_grad_(False);agents[method]=a
    reader=ContactGoalAgent(base,obs,'E',e.standing.device);reader.modules.requires_grad_(False)
    bcnet=None
    if any(r['method']=='BC500'for r in selected):
        from clean_policy import Expert
        path=STAGE.parents[1]/'experiments/interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt'
        assert sha(path)=='a6688f292d8a212bebed4384532026ce3f703269165cd74afd0d0d635d4d4e1e'
        ck=torch.load(path,map_location='cuda',weights_only=True);bcnet=Expert(1639,'direct').cuda();bcnet.load_state_dict(ck['state_dict'],strict=True);bcnet.eval().requires_grad_(False)
        identities['BC500']=dict(model_sha256=sha(path),input_dim=1639,goal_not_an_input=True)
    jsonwrite(out/'evaluation-lock.json',dict(runtime=runtime,plan_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),cases=selected,identities=identities,optimizer_updates=0,sealed_read=False,native_zero_U='original actor and constant standing legal z; same source prefix',alternate_goal_rule='invert FOLLOW/HOLD in every segment, preserve issue time',evaluation_seed=660304))
    results=[];prefix={};source={};actual_targets=env.targets;started=time.monotonic()
    for record in selected:
        guard(out,started);i=record['index'];method=record['method'];a=agents[method]
        random.seed(660304);np.random.seed(660304);torch.manual_seed(660304);torch.cuda.manual_seed_all(660304)
        env.rows=[];env.native=[];env.physical_stream=[];env.instructions=[];env.episodes=[];env.next_cursor=0;env.reset_cursor=0;env.targets=actual_targets
        if hasattr(env,'reference_constant_option'):del env.reference_constant_option
        env.cases=[copy.deepcopy(record['case'])];env.agent=a;env.proxy.agent=a;env.reset(0);env.phase='P3'if method in ('E-final','I-final')else'P2';env.proxy.stochastic=False;env.collection_role='independent_dev_not_PPO'
        if method=='native-zero-U':env.reference_constant_option=0
        if method=='BC500':
            from clean_policy import direct_command
            def targets():
                raw=bcnet.distribution(env.task.student_features()).mean;z=direct_command(raw,e.standing)
                return [raw]*4,torch.cat([z]*4),0.
            env.targets=targets
        frozen=tree_digest(a.state());initial=canonical(portable(env.task),metadata(e.sim.model,e.sim.data));source_sha=env.task.prefix_data['audit']['sha256'];key=record['parent_id'];anchor=env.task.anchor.copy()
        if key in source:assert source[key]==(source_sha,anchor.tolist()),'source/common partner differs across methods'
        else:source[key]=(source_sha,anchor.tolist())
        initial_rng=global_rng();initial_file=dump(out/f'case-{i:03d}-handoff.ptz',dict(physics=portable(env.task),rng=initial_rng,source_sha=source_sha,common_anchor=anchor,role='evaluation_handoff_not_PPO'))
        release_ref=None;ref_fail=None;hard=None
        while not env.task.done:
            guard(out,started)
            if release_ref is None and env.task.spec['enabled'] and env.task.spec['distance']>0 and float(e.tether.time)>=env.task.spec['release']-1e-7:
                snap=EnvBranch(env);before=canonical(portable(env.task),metadata(e.sim.model,e.sim.data));rng=global_rng();release_state=portable(env.task);saved_agent=env.agent;saved_phase=env.phase;saved_targets=env.targets
                try:
                    env.agent=reader;env.proxy.agent=reader;env.phase='P2';env.targets=actual_targets;env.reference_constant_option=0;env.collection_role='same_state_native_recovery_not_PPO';rp=len(env.physical_stream)
                    ref_hard=None
                    try:
                        while not env.task.done:
                            guard(out,started);env.collect(5)
                    except RuntimeError as ex:
                        if not str(ex).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                        ref_hard=str(ex)
                    refph=copy.deepcopy(env.physical_stream[rp:]);release_ref=quality(refph);ref_fail=protection(release_ref,'recovery',release_ref)
                    if ref_hard:ref_fail.append('native_reference_hard_boundary')
                    dump(out/f'case-{i:03d}-native-recovery.ptz',dict(initial_complete_state=release_state,physics=refph,quality=release_ref,failed=ref_fail,role='actual_complete_same_state_native_branch_not_PPO'))
                finally:
                    snap.restore();env.agent=saved_agent;env.proxy.agent=saved_agent;env.phase=saved_phase;env.targets=saved_targets;restore_global_rng(rng)
                assert exact(canonical(portable(env.task),metadata(e.sim.model,e.sim.data)),before),'Dev native-reference full state differs'
                assert exact(global_rng(),rng)
            try:env.collect(5)
            except RuntimeError as ex:
                if not str(ex).startswith('HARD_PHYSICS_BOUNDARY:'):raise
                assert env.task.boundary;hard=str(ex);break
        assert tree_digest(a.state())==frozen,'Evaluation changed frozen weights/normalizer/Adam'
        assert all(not r['eligible_for_joint_PPO']for r in env.rows)
        t=env.task;ph=env.physical_stream;r=summarize(t);r.update(case_index=i,method=method,role=record['role'],goal_stream=record['goal_stream'],runtime=runtime,identity=identities[method],source_sha=source_sha,complete=bool(t.done and hard is None),hard_boundary=hard,goal_aware=stage_metrics(env,ph,env.rows),online_quality=quality(ph),optimizer_updates=0)
        family=t.spec['demand_family'];rec=[p for p in ph if p['force_time']>=t.spec['release']-1e-8]
        r['recovery']=dict(status='NOT_APPLICABLE_no_actual_loaded_release')
        if t.spec['enabled'] and t.spec['distance']>0:
            measured=release_ref is not None and not ref_fail
            r['recovery']=dict(status='MEASURED'if measured else'UNMEASURED_native_reference_unqualified',reference=release_ref,reference_failed=ref_fail,actual=quality(rec)if rec else None,failed=protection(quality(rec),'recovery',release_ref if measured else None)if rec else['missing_recovery'])
        if family in ('none','zero'):r['protection_failed']=protection(r['online_quality'],family)
        if family in ('stop','continue_pair'):
            branch_at=next(p['specs'][0]['stop_time']for p in parent if p['parent_id']==key)
            pre=[{k:x[k]for k in ('bfm_obs','executed_z','action','q_PD','pre_qpos','pre_qvel')}for x in t.controls if x['time']<branch_at-1e-8]
            pk=(method,record['role'],key,record['goal_stream']);digest=tree_digest(pre)
            if pk in prefix:assert prefix[pk]==digest,'same model stop/continue common prefix differs'
            else:prefix[pk]=digest
            r['common_prefix_sha256']=digest
        r['archive']=dump(out/f'case-{i:03d}-actual.ptz',dict(spec=t.spec,rows=env.rows,physics=t.physics,online_physics=env.physical_stream,controls=t.controls,source=t.prefix_data,instructions=env.instructions,role='actual_frozen_policy_independent_dev_not_training'))
        r['handoff_archive']=initial_file;results.append(r);jsonwrite(out/'evaluation.json',results)
        print('FROZEN_DEV_CASE',i,method,record['role'],r['complete'],r['all']['error_cm'],r['all']['force_rms'],'boundary',hard,flush=True)
        if hard:return dict(status='DEV_HARD_BOUNDARY_STOP',episodes=len(results),planned=len(selected),failed_case=i,optimizer_updates=0,sealed_read=False,results_file='evaluation.json')
    return dict(status='FIXED_TERMINAL_DEV_COMPLETE',episodes=len(results),planned=len(selected),optimizer_updates=0,sealed_read=False,identities=identities,results_file='evaluation.json')


def guard(out,started):
    import shutil
    from . import benchmark
    if benchmark.STOP_REQUESTED:raise RuntimeError('DEV_SUPERVISOR_STOP')
    if time.monotonic()-started>float(os.environ.get('EI_DEV_HOT_CAP','11000')):raise RuntimeError('DEV_HOT_BUDGET_STOP')
    assert shutil.disk_usage(STAGE).free>6*2**30,'DEV_FREE_SPACE_RESERVE'


def main():
    global METHOD
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--method',choices=['E-final','I-final','Goal-only','native-zero-U','BC500']);a=p.parse_args();METHOD=a.method
    from . import benchmark
    from .ei_v2_counter_io import install
    from .ei_v2_counter_lock import atomic_write
    oldwrite=benchmark.write
    def write(path,data):
        if Path(path).name!='physics-counter.json':return oldwrite(path,data)
        if data['reserved']>int(os.environ['EI_FORMAL_PHYSICS_CAP']):raise RuntimeError('DEV_PHYSICS_CAP')
        atomic_write(path,data)
    benchmark.write=write;install(benchmark);benchmark.study=study
    sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
