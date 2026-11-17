"""CTFT S0 complete original-native source restore qualification.

No optimizer or training. Same backend/shape, exact original native commands,
full physical/historical state; portable validity rules are the prior proven
MuJoCo3.3.3/no-island/Newton/dense contract, not relaxed tolerances.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def study(e,model,contract,out):
    import numpy as np,torch,mujoco
    from m8_stage import reset_common,advance
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from rolling_snapshot import put,state_of
    from f1_mjdata_contract import metadata,inactive,exact
    from quality import OnlineSupport
    speed(e);(out/'backend-prefix').mkdir();backend(e,model,out/'backend-prefix')
    reset_common(e,model,model.act)
    source_dir=STAGE/'runs/s0-holdout-sources-02'
    selection=json.loads(Path(os.environ.get('CTFT_SELECTED_MANIFEST',str(STAGE/'locks/selected-source-state-manifest-v1.json'))).read_text())
    summary=selection['states'];assert len(summary)==selection.get('expected_count',24)
    # The original assets/reference definitions stay fixed; no recovery source
    # is silently substituted for a failed native stop.
    actual_sources=[];tests=[];weight=float(e.sim.model.body_mass.sum()*np.linalg.norm(e.sim.model.opt.gravity))
    for record in summary:
        path=source_dir/record['evidence_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['evidence_sha256']
        with gzip.open(path,'rb') as f:x=pickle.load(f)
        # Saved tensors retain their real device, unlike the physics-only reader.
        assert torch.is_tensor(x['controls'][0]['z'])
        full_controls=len(x['controls'])-(1 if x['boundary'] is not None else 0)
        indices=[record['selected_control']]
        assert all(i+12<=full_controls for i in indices),('Prelocked state coverage insufficient',record['case'],full_controls)
        for index in indices:
            saved=x['states'][index];d=e.sim.data;m=e.sim.model
            for obj,key in [(e,'env'),(e.base,'base'),(e.sim,'sim'),(e.wrapped,'wrapped'),(e.tether,'tether')]:
                put(obj,saved[key])
            e.base.history_handler.history=copy.deepcopy(saved['bfm_history'])
            e.rng.bit_generator.state=copy.deepcopy(saved['env_generator_state'])
            e.contact_rng.bit_generator.state=copy.deepcopy(saved['contact_generator_state'])
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            mask=mujoco.mjtState.mjSTATE_INTEGRATION
            mujoco.mj_setState(m,d,saved['integration'],mask);mujoco.mj_forward(m,d)
            meta=metadata(m,d);nulls=inactive(meta)
            for key,value in saved['mjdata_arrays'].items():
                if key in nulls:continue
                dest=getattr(d,key)
                if dest.shape==value.shape and dest.flags.writeable:dest[...]=value
            mujoco.mj_setState(m,d,saved['integration'],mask)
            # All valid solver/dynamics arrays exact; no name-based whitelist.
            bad=[k for k,v in saved['mjdata_arrays'].items() if k not in nulls and not exact(v,getattr(d,k))]
            assert not bad,('Restored valid MuJoCo arrays differ',record['case'],index,bad)
            assert exact(e.obs,saved['env']['obs'])
            assert exact(e.previous_action,saved['env']['previous_action'])
            assert exact(e.base.history_handler.history,saved['bfm_history'])
            assert exact(e.rng.bit_generator.state,saved['env_generator_state'])
            assert exact(e.contact_rng.bit_generator.state,saved['contact_generator_state'])
            assert exact(torch.get_rng_state(),saved['torch_rng'])
            assert exact(torch.cuda.get_rng_state_all(),saved['cuda_rng'])
            assert exact(random.getstate(),saved['python_rng']) and exact(np.random.get_state(),saved['numpy_rng'])
            monitor=OnlineSupport(**saved['online_support']);e.qualification_active=True;e.trace=[]
            for j in range(12):
                control=x['controls'][index+j]
                assert exact(e.obs,control['bfm_observation']),('BFM history mismatch',record['case'],index,j)
                action=model.act(e.obs,control['z'].to(e.standing.device),mean=True)
                assert exact(action.detach().cpu(),control['action']),('Native action mismatch',record['case'],index,j)
                e.previous_z=control['z'].to(e.standing.device).clone()
                advance(e,action)
                expected=x['physics'][(index+j)*4:(index+j+1)*4]
                observed=e.trace[-4:]
                assert len(expected)==len(observed)==4
                for a,b in zip(expected,observed):
                    for key in ['qpos','qvel','root','wrist','force','foot_normal_force_n','support']:
                        assert exact(a[key],b[key]),('Physics restore mismatch',record['case'],index,j,key)
                    status=monitor.update(b['foot_normal_force_n'])
                    assert exact(status,a['ctft_online_support']),('Online support history mismatch',record['case'],index,j)
                    assert not status['stop']
            e.qualification_active=False
            tests.append(dict(case=record['case'],ancestor=record['ancestor'],control_index=index,
                restored_time_s=float(x['controls'][index]['time']),
                matched_native_controls=12,matched_physics_ticks=48,
                valid_solver_arrays_exact=True,full_bfm_action_history_exact=True,
                rngs_exact=True,continuation_physics_exact=True,online_support_exact=True))
            write(out/'progress.json',tests)
        actual_sources.append(dict(case=record['case'],ancestor=record['ancestor'],
            qualified_state_controls=indices,complete_native_13s=record['complete'],
            native_stop_failure_retained=not record['complete']))
        del x
    result=dict(status='SELECTED_HOLDOUT_SOURCE_RESTORE_QUALIFIED',tests=tests,
        source_ancestors=len(actual_sources),source_qualification=actual_sources,
        qualification_scope=f'One prelocked actual state per{len(summary)} independent holdout ancestors. Original source hard tails remain negative; no new contact task has been evaluated.',
        scientific_source_coverage='Full state and12 native controls reproduce; this is not a claim of long-horizon new-goal or contact recovery success.',
        model_unchanged=True,training_transitions=0,updates=0)
    write(out/'result-restore.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
    if a.manifest:os.environ['CTFT_SELECTED_MANIFEST']=str(a.manifest)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'launch.json',dict(pid=os.getpid(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),started_unix=time.time(),purpose='full native source restore qualification, no training'))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1800))
    import mujoco
    raw=mujoco.mj_step;counts=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));counts['reserved']+=n;assert counts['reserved']<=5000
        result=raw(*args,**kw);counts['actual']+=n;return result
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=a.output
    begun=time.monotonic()
    try:m10_physics.main()
    finally:
        write(a.output/'cost.json',dict(wall_s=time.monotonic()-begun,physics=counts,pid=os.getpid()))


if __name__=='__main__':main()
