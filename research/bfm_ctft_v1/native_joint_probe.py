"""Disposable S0 joint-update wiring check using this stage's real g0 sources.

This is NOT the required contact/full-chain T0 benchmark, a formal warm-up, or
a calibration dataset. No probe weights/data migrate into the formal U/T run.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from agent import build,ACTOR_KEY,CRITIC_KEY
from features import ForceHistory,actor_fields
from humanoidverse.agents.buffers.transition import DictBuffer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    def write(name,data):(a.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write('launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),scope='g0-only disposable loop wiring, not full-chain contact qualification'))
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.manual_seed(681117)
    counts=dict(real_new_physics_ticks=0,formal_training_transitions=0,agent_updates=0)
    try:
        agent,manifest=build(aux_scale=1.)
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
        data=mujoco.MjData(model)
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
        pelvis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        assert body>0 and pelvis>0 and model.nv==35
        train_rows=[];expert_groups=[];sources=[]
        scales=manifest['original_training_config']['aux_rewards_scaling']
        for case in [1,2]:
            path=STAGE/'runs/s0-native-sources-03'/f'source-{case:02d}.pkl.gz'
            with gzip.open(path,'rb') as f:x=pickle.load(f)
            assert x['boundary'] is None
            assert all(np.linalg.norm(r['force'])==0 for r in x['physics'])
            cache={}
            def obs(i):
                if i in cache:return cache[i]
                state=x['states'][i];d0=state['mjdata_arrays']
                data.qpos[:]=d0['qpos'];data.qvel[:]=d0['qvel'];mujoco.mj_forward(model,data)
                point=data.xpos[body].copy();jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,point,body)
                # Ten actual prior source samples, proven externally unloaded.
                prior=x['physics'][(i-10)*4:i*4:4]
                assert len(prior)==10
                history=ForceHistory(np.array([np.array(r['pelvis_matrix']).T@np.array(r['force']) for r in prior]))
                extra=actor_fields(force_history=history,known_mount=[0.,0.,0.],
                    wrist_world=point,root_world=data.qpos[:3],wrist_velocity_world=jac@data.qvel,
                    root_velocity_world=data.qvel[:3],world_from_body=data.xmat[pelvis].reshape(3,3),public_mode=0)
                result={k:v.detach().cpu().clone() for k,v in x['controls'][i]['bfm_observation'].items()}
                result[ACTOR_KEY]=torch.tensor(extra[None]);result[CRITIC_KEY]=torch.zeros(1,9)
                cache[i]=result;return result
            group=[]
            for i in range(25,625):
                c=x['controls'][i];n=x['controls'][i+1]
                assert abs(n['time']-c['time']-.02)<1e-6
                raw=x['states'][i+1]['base']['extras']['aux_rewards']
                reward=sum(float(raw[k].detach().cpu().reshape(-1)[0])*coef for k,coef in scales.items())
                row=dict(observation=obs(i),action=c['action'].detach().cpu().clone(),z=c['z'].detach().cpu().clone(),
                    next=dict(observation=obs(i+1),terminated=torch.zeros(1,1,dtype=torch.bool),truncated=torch.zeros(1,1,dtype=torch.bool)),
                    aux_rewards=dict(ctft_total_native=torch.tensor([[reward]],dtype=torch.float32)))
                train_rows.append(row);group.append(row)
            expert_groups.append(group)
            sources.append(dict(case=case,ancestor=x['case']['ancestor'],actual_rows=len(group),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            del x,cache
        def cat(rows):
            def combine(values):
                if isinstance(values[0],dict):return {k:combine([v[k] for v in values]) for k in values[0]}
                return torch.cat(values)
            return combine(rows)
        buffer=DictBuffer(capacity=4096,device='cpu');buffer.extend(cat(train_rows))
        class Sequences:
            def sample(self,batch_size):
                assert batch_size%8==0
                rows=[]
                for _ in range(batch_size//8):
                    g=expert_groups[int(torch.randint(len(expert_groups),()).item())]
                    start=int(torch.randint(len(g)-7,()).item());rows.extend(g[start:start+8])
                c=cat(rows);return dict(observation=c['observation'],next=dict(observation=c['next']['observation']))
        buffers=dict(train=buffer,expert_slicer=Sequences())
        def frozen_sha():
            h=hashlib.sha256()
            for module in ['_backward_map','_target_backward_map','_obs_normalizer']:
                for k,v in getattr(agent._model,module).state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        before=frozen_sha();results=[]
        for step in range(4):
            agent.actor_enabled=step>=2
            torch.cuda.synchronize();t=time.monotonic()
            metrics=agent.update(buffers,step)
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
            counts['agent_updates']+=1
            free,total=torch.cuda.mem_get_info()
            assert torch.cuda.max_memory_reserved()<=.88*total and free>=max(6*2**30,.08*total)
            assert frozen_sha()==before,'Frozen B or normalizer changed during update'
            result=dict(step=step,actor_enabled=agent.actor_enabled,wall_s=elapsed,
                metrics={k:float(v.detach().cpu()) for k,v in metrics.items() if torch.is_tensor(v) and v.numel()==1},
                optimizer_state_entries={k:len(v['state']) for k,v in agent.optimizer_dict.items()})
            results.append(result);write('progress.json',results);print('CTFT_DISPOSABLE_UPDATE',step,elapsed,flush=True)
        assert all(len(v['state'])>0 for v in agent.optimizer_dict.values())
        write('result.json',dict(status='DISPOSABLE_NATIVE_G0_JOINT_LOOP_PASSED',sources=sources,
            unique_real_source_rows=len(train_rows),counts=counts,updates=results,
            original_B_targets_normalizers_unchanged=True,
            compile=agent.cfg.compile,logical_batch=1024,placeholder_aux_scale=1.,
            formal_contact_calibration=False,complete_training_checkpoint_restore=False,
            actual_contact_transition_qualification=False,
            peak_reserved=torch.cuda.max_memory_reserved(),wall_s=time.monotonic()-begun))
    except BaseException as exc:
        write('failure.json',dict(error_type=type(exc).__name__,error=str(exc),counts=counts,wall_s=time.monotonic()-begun));raise
    finally:write('cost.json',dict(**counts,wall_s=time.monotonic()-begun))


if __name__=='__main__':main()
