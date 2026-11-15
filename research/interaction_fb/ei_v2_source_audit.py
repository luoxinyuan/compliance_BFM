"""P0 train-only full native source-prefix identity intervention; no learning."""
import argparse, copy, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'research_notes')]
STAGE = ROOT.parent
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import m11_ipc

def study(e, model, contract, out):
    import hashlib, mujoco, numpy as np, torch
    from native_env import install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from dual_track_task import Task, global_rng
    from diagnostic_frozen import portable
    from diagnostic_fb_resume import tree_digest
    from f1_mjdata_contract import canonical, metadata, exact, install_restore_guard
    from research.interaction_fb.journal import dump
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if os.environ.get('EI_V2_SOURCE_FIX')=='1':
        from research.interaction_fb.ei_v2_sources import install as source_install
        source_install()
    assert contract['model_sha256'] == '33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
    install(); speed(e); (out/'backend-prefix').mkdir(); backend(e, model, out/'backend-prefix'); install_restore_guard()
    frozen = tree_digest(model._actor.state_dict())
    lock = json.loads((STAGE/'locks/source-recipe-audit.json').read_text())
    train = json.loads((STAGE/'locks/expanded-train.json').read_text())['parents']
    rows = {p['parent_id']:p for p in train}
    check = lock['physical_check']; boundary = check['short_end']
    task = Task(e, model, contract, 660301)
    native = task.native; captured = []; outputs = []
    def state():
        mask=mujoco.mjtState.mjSTATE_INTEGRATION
        integration=np.zeros(mujoco.mj_stateSize(e.sim.model, mask))
        mujoco.mj_getState(e.sim.model,e.sim.data,integration,mask)
        pp=canonical(portable(task),metadata(e.sim.model,e.sim.data))
        return dict(time=float(e.tether.time),integration=integration,
            mjdata_arrays=pp['mjdata_arrays'],mjdata_validity=pp['mjdata_validity'],
            bfm_obs={k:v.detach().cpu().clone() for k,v in e.obs.items()},
            previous_action=e.previous_action.detach().cpu().clone(),
            previous_z=task.last_command.detach().cpu().clone(),
            previous_qpd=task.lastq.detach().cpu().clone(),
            deploy_h=task.student_features().detach().cpu().clone(),
            actual_force=np.asarray(e.tether.force).copy(),
            physics_rng=copy.deepcopy(task.physics_rng if not task.allow_native_reset else global_rng()))
    def observed(name,t):
        z=native(name,t)
        if abs(t-boundary)<1e-7: captured.append(state())
        return z
    task.native=observed
    for label in ['short_parent','long_parent']:
        parent=rows[check[label]]; sp=copy.deepcopy(parent['specs'][0]);captured.clear()
        task.reset(sp,record=True,label='P0-'+label)
        assert task.boundary is None
        assert not e.tether.active and all(not r['active'] for r in task.prefix_data['physics'])
        assert all(np.linalg.norm(r['force'])==0 for r in task.prefix_data['physics'])
        if label=='short_parent': at=state()
        else:
            assert len(captured)==1,('missing exact train prefix boundary',boundary,len(captured))
            at=captured[0]
        physics=copy.deepcopy(task.prefix_data['physics'])
        controls=copy.deepcopy(task.prefix_data['control'])
        if label=='long_parent':
            physics=[r for r in physics if r['time']<=boundary+1e-8]
            controls=[c for c in controls if c['time']<boundary-1e-8]
        body_keys=['time','qpos','qvel','wrist','root','force','support']
        body=[{k:r[k] for k in body_keys} for r in physics]
        act=[{k:c[k] for k in ['time','executed_z','action','q_PD','bfm_obs']} for c in controls]
        packet=dict(parent=parent,source_at_common_boundary=at,body_prefix=body,
            action_prefix=act,full_source=copy.deepcopy(task.prefix_data),raw_physical=portable(task),
            identity='actual_frozen_native_TRAIN_source_qualification_no_optimization')
        archive=dump(out/(label+'.ptz'),packet)
        outputs.append(dict(parent=parent['parent_id'],label=label,state=at,body=body,actions=act,archive=archive,
            source_seconds=float(e.tether.time)-3,common_physics_rows=len(body),common_controls=len(act)))
    a,b=outputs
    field_equal={k:exact(a['state'][k],b['state'][k]) for k in a['state']}
    result=dict(status='P0_SOURCE_IDENTITY_PHYSICAL_AUDIT_COMPLETE',
        source_fix_enabled=os.environ.get('EI_V2_SOURCE_FIX')=='1',
        common_time_s=boundary,short_parent=a['parent'],long_parent=b['parent'],
        state_field_exact=field_equal,body_prefix_exact=exact(a['body'],b['body']),
        actions_qpd_prefix_exact=exact(a['actions'],b['actions']),
        common_physics_rows=[x['common_physics_rows'] for x in outputs],
        common_controls=[x['common_controls'] for x in outputs],
        archives=[x['archive'] for x in outputs],
        source_has_actual_contact=False,formal_rl_transitions=0,updates=0,dev_physics=0,sealed_read=False,
        original_bfm_actor_unchanged=tree_digest(model._actor.state_dict())==frozen,
        next_action='STOP_G01_DATA_INSUFFICIENT if shared actual source confirmed; do not alter recipe or split')
    assert result['original_bfm_actor_unchanged']
    (out/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from research.interaction_fb import benchmark
    import mujoco
    original=mujoco.mj_step; count=0
    def counted(*a,**kw):
        nonlocal count
        n=int(kw.get('nstep',1)); count+=n
        assert count<=10000,'P0 train source audit bounded ticks'
        return original(*a,**kw)
    # benchmark installs its own per-step counter; our cap wraps the same call.
    mujoco.mj_step=counted;benchmark.study=study
    sys.argv=[__file__,'--output',str(args.output),'--arm','A'];benchmark.main()

if __name__=='__main__': main()
