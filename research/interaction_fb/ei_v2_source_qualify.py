"""New v2 arm-ready execution-source qualification, all fixed train/dev sources.
No task/Goal evaluation, sealed access, optimization or source selection.
"""
import os,sys,json,copy,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def prepare():
    from research.interaction_fb.ei_v2_sources import SOURCE_VERSION,source_key,arm_fraction
    import hashlib
    out={};arms=[]
    for split in ('train','dev'):
        data=json.loads((STAGE/f'locks/expanded-{split}.json').read_text())
        for p in data['parents']:
            p['original_source_parent_id']=p['source_parent_id']
            key=source_key(p['specs'][0]);p['execution_source_parent_id']=key
            for s in p['specs']:
                assert source_key(s)==key
                s.update(execution_source_version=SOURCE_VERSION if p['source']=='goal_arm5' else 'existing-standing-native-pulse',execution_source_parent_id=key)
            if p['source']=='goal_arm5':arms.append({'split':split,'parent_id':p['parent_id'],'source_key':key,'spec':p['specs'][0]})
        out[split]=data
    assert len(arms)==12
    assert len({a['source_key'] for a in arms})==12
    tr={p['execution_source_parent_id'] for p in out['train']['parents']};de={p['execution_source_parent_id'] for p in out['dev']['parents']};assert not tr&de
    # Input/command-level qualification precedes physics; no ignored phase fields used to create independence.
    import numpy as np
    histories=[np.asarray([arm_fraction(t,a['spec']['source_profile']) for t in np.arange(3.,4.1,.02)]) for a in arms]
    for i in range(len(histories)):
        for j in range(i):assert not np.array_equal(histories[i],histories[j]),(i,j)
    plan=dict(version=SOURCE_VERSION,scope='all 12 already-assigned arm sources; source-only physics, no Goal/task outcomes, no sealed',cases=arms,no_training=True,original_distribution_changed_only_by_authorized_source_implementation_correction=True)
    for split,data in out.items():(STAGE/f'locks/expanded-corrected-{split}.json').write_text(json.dumps(data,indent=2)+'\n')
    path=STAGE/'locks/corrected-source-plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    return dict(cases=len(arms),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_dev_execution_source_disjoint=True,full_command_histories_distinct=True,sealed_cases_read=False)

def study(e,model,contract,out):
    import numpy as np,torch
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from dual_track_task import Task
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from f1_mjdata_contract import canonical,metadata,exact,install_restore_guard
    from research.interaction_fb.ei_v2_sources import install,SOURCE_VERSION
    from research.interaction_fb.journal import dump
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,model,out/'backend-prefix');install_restore_guard()
    plan=json.loads((STAGE/'locks/corrected-source-plan.json').read_text());frozen=tree_digest(model._actor.state_dict());task=Task(e,model,contract,660301)
    records=[];brief=[]
    for i,c in enumerate(plan['cases']):
        task.reset(copy.deepcopy(c['spec']),record=True,label=f'source-{i:02d}')
        assert task.boundary is None and not e.tether.active
        ph=task.prefix_data['physics'];cs=task.prefix_data['control']
        assert all(not p['active'] and np.linalg.norm(p['force'])==0 for p in ph)
        pp=canonical(portable(task),metadata(e.sim.model,e.sim.data))
        b=[{k:r[k] for k in ('time','qpos','qvel','wrist','root','force','support')} for r in ph]
        ca=[{k:r[k] for k in ('time','executed_z','action','q_PD','bfm_obs')} for r in cs]
        records.append(dict(case=c,body=b,actions=ca,state=pp))
        archive=dump(out/f'source-{i:02d}.ptz',dict(case=c,full_prefix=task.prefix_data,full_state=portable(task),canonical_state=pp,identity='source_only_qualification_not_RL_or_Task_evaluation'))
        brief.append(dict(index=i,split=c['split'],parent_id=c['parent_id'],source_key=c['source_key'],time=float(e.tether.time),physics_rows=len(ph),control_rows=len(cs),original_hard_boundary=None,source_actual_force_zero=True,archive=archive))
        (out/'progress.json').write_text(json.dumps(brief,indent=2)+'\n')
    pairs=[]
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            # Compare common executed post-reset prefix: identical reset first3s is not used as evidence of independence.
            n=min(len(a['body']),len(b['body']));nc=min(len(a['actions']),len(b['actions']))
            same_body=exact(a['body'][:n],b['body'][:n]);same_act=exact(a['actions'][:nc],b['actions'][:nc])
            pairs.append(dict(a=i,b=j,common_physics_rows=n,common_controls=nc,body_prefix_equal=same_body,action_prefix_equal=same_act,cross_split=a['case']['split']!=b['case']['split']))
            assert not same_body and not same_act,('source execution histories still duplicate',i,j)
    assert tree_digest(model._actor.state_dict())==frozen
    result=dict(status='CORRECTED_ARM_SOURCE_EXECUTION_QUALIFIED',version=SOURCE_VERSION,sources=brief,paired_comparisons=pairs,actual_physics_prefixes_distinct=True,model_unchanged=True,formal_updates=0,dev_task_episodes=0,dev_source_prefixes=6,sealed_read=False,
        independence_boundary='distinct executed native preparation histories and disjoint effective source keys; no statistical independence or unseen-goal success claim; original shared BFM assets are common')
    (out/'source-qualification.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.prepare_only:print(json.dumps(prepare(),indent=2));return
    assert a.output
    from research.interaction_fb import benchmark
    import mujoco
    original=mujoco.mj_step;count=0
    def step(*args,**kw):
        nonlocal count
        count+=int(kw.get('nstep',1));assert count<=22000
        return original(*args,**kw)
    mujoco.mj_step=step;benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
