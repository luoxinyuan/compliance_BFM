"""Pre-formal hardware-only synchronous CPU-task / single-GPU inference probe.

Separate processes own Task, solver, force/history and RNG. Exactly one parent
owns CTFT inference. Workers exchange executed rows via local IPC; no learner,
loss, source sampler or scientific training configuration is changed here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def worker_study(e,original,contract,out):
    import numpy as np
    import torch
    from multiprocessing.connection import Client
    from contact_live import ContactLive
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from locomotion_rollout import UnsafePhysics
    plan=json.loads((out/'worker-protocol.json').read_text())
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix')
    reset_common(e,original,original.act)
    coeff=json.loads(Path('/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference/config.json').read_text())['agent']['aux_rewards_scaling']
    live=ContactLive(e,STAGE,coeff)
    spec=([dict(mode=1,force=True),dict(mode=1,force=False),dict(mode=0,case=1),dict(mode=0,case=2)])[plan['worker_id']%4]
    live.start(spec)
    def numpy(x):
        if torch.is_tensor(x):return x.detach().cpu().numpy().copy()
        if isinstance(x,dict):return {k:numpy(v) for k,v in x.items()}
        return x
    def current():return dict(obs=numpy(live.observation()),z=numpy(live.command()))
    conn=Client(bytes.fromhex(plan['address_hex']),family='AF_UNIX')
    conn.send(dict(ready=True,pid=os.getpid(),worker_id=plan['worker_id'],contract=contract,**current()))
    rows=0
    try:
        while True:
            command=conn.recv()
            if command['kind']=='stop':break
            assert command['kind']=='step'
            before=live.observation();z=live.command()
            action=torch.tensor(command['action'],device=e.standing.device,dtype=torch.float32)
            row=live.step(action,before,z);rows+=1
            if live.done:live.start(spec)
            conn.send(dict(row=numpy(row),**current()))
    except UnsafePhysics as exc:
        write(out/'hard-boundary.json',dict(row=exc.row,actual_rows=rows))
        conn.send(dict(failure='real_hard_physics',row=exc.row,worker_id=plan['worker_id']))
        raise
    finally:
        conn.close();e.qualification_active=False
    return dict(status='HARDWARE_WORKER_DONE',actual_rows=rows,no_updates=True)


def worker_main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'worker-protocol.json',dict(worker_id=a.worker_id,address_hex=a.address))
    os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),CCL_DEADLINE_MONOTONIC=str(time.monotonic()+1500))
    import mujoco
    raw=mujoco.mj_step;count=dict(actual=0,reserved=0)
    def counted(*args,**kw):
        n=int(kw.get('nstep',1));count['reserved']+=n;assert count['reserved']<=10000
        r=raw(*args,**kw);count['actual']+=n;return r
    mujoco.mj_step=counted
    import m10_stage,m10_physics
    m10_stage.run_study=worker_study;m10_physics.OUT=a.output
    start=time.monotonic()
    try:m10_physics.main()
    finally:write(a.output/'worker-cost.json',dict(wall_s=time.monotonic()-start,physics=count,updates=0))


def central(a):
    import numpy as np
    import torch
    import signal
    from multiprocessing.connection import Listener
    from agent import build
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'protocol.json',dict(scope='S0 hardware qualification; frozen mean function, original stochastic sample; no training',
        candidate_workers=[1,4,8,16,32],actual_rows_each=1024,stop_gain_less_than=.10,
        action_cross_batch_tolerance=dict(atol=1e-6,rtol=1e-5),
        joint_update_or_checkpoint_rate='separate real online probe, not falsely counted here',
        selection='stop at first <10% marginal collection gain or resource/physics boundary; full chosen-config restore remains separate',
        wall_cap_s=1800,physics_cap=120000))
    write(a.output/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    lock=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())
    agent,_=build(aux_scale=lock['value'],compile_execution=False)
    all_results=[];start_all=time.monotonic()
    for n in [1,4,8,16,32]:
        assert time.monotonic()-start_all<1500
        folder=a.output/f'workers-{n}';folder.mkdir()
        address=b'\0ctft-hardware-'+os.urandom(12).hex().encode()
        listener=Listener(address,family='AF_UNIX')
        listener._listener._socket.settimeout(240)
        procs=[];connections={};streams=[];begun=time.monotonic()
        try:
            for i in range(n):
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker-id',str(i),'--address',address.hex(),'--output',str(folder/f'worker-{i}')]
                log=(folder/f'worker-{i}.log').open('w');streams.append(log)
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
                procs.append(child)
                write(folder/'owners.json',[dict(pid=p.pid,worker=j) for j,p in enumerate(procs)])
            messages={}
            for _ in range(n):
                conn=listener.accept()
                assert conn.poll(240),'Worker did not send bounded ready message'
                ready=conn.recv();i=ready['worker_id']
                assert ready['pid']==procs[i].pid and i not in connections
                connections[i]=conn;messages[i]=ready
            startup_s=time.monotonic()-begun
            def tensors():
                return ({k:torch.tensor(np.concatenate([messages[i]['obs'][k] for i in range(n)]),device=agent.device) for k in messages[0]['obs']},
                    torch.tensor(np.concatenate([messages[i]['z'] for i in range(n)]),device=agent.device))
            obs,z=tensors()
            with torch.no_grad():
                batched=agent.act(obs,z,mean=True)
                singles=torch.cat([agent.act({k:v[i:i+1] for k,v in obs.items()},z[i:i+1],mean=True) for i in range(n)])
            assert torch.allclose(batched,singles,atol=1e-6,rtol=1e-5),'Cross-shape mean action numerical contract failed'
            difference=float((batched-singles).abs().max())
            action_contract=messages[0]['contract']
            qscale=torch.tensor(action_contract['q_PD_scale'],device=agent.device)
            qoff=torch.tensor(action_contract['q_PD_offset'],device=agent.device)
            clip=action_contract['clip']
            q1=qoff+(batched*5).clamp(-clip,clip)*qscale/5
            q2=qoff+(singles*5).clamp(-clip,clip)*qscale/5
            qpd_difference=float((q1-q2).abs().max())
            torch.manual_seed(681117)
            begun=time.monotonic();rounds=1024//n
            for round_index in range(rounds):
                obs,z=tensors()
                with torch.no_grad():action=agent.act(obs,z,mean=False).cpu().numpy()
                for i in range(n):connections[i].send(dict(kind='step',action=action[i:i+1]))
                for i in range(n):
                    assert connections[i].poll(60),'Hardware worker step deadline'
                    answer=connections[i].recv();assert 'failure' not in answer,answer
                    messages[i]=answer
            torch.cuda.synchronize();elapsed=time.monotonic()-begun
            free,total=torch.cuda.mem_get_info()
            memory=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
            assert free>=max(6*2**30,.08*total) and int(memory)*2**20<=.88*total
            result=dict(workers=n,actual_new_rows=1024,collection_s=elapsed,rows_per_s=1024/elapsed,
                startup_s=startup_s,cross_batch_max_action_abs_error=difference,total_gpu_used_mib=int(memory),
                cross_batch_max_q_PD_abs_error_rad=qpd_difference,
                central_only_updates=0,all_Task_callbacks_RNG_separate=True,physics_executed_in_separate_processes=True,
                chosen_config_full_restore='PENDING')
            all_results.append(result);write(a.output/'progress.json',all_results)
            print('CTFT_HARDWARE_WORKERS',json.dumps(result),flush=True)
        finally:
            for c in connections.values():
                try:c.send(dict(kind='stop'))
                except (EOFError,OSError):pass
            for p in procs:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if p.poll() is None:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            for c in connections.values():c.close()
            listener.close()
            for f in streams:f.close()
        if len(all_results)>1 and all_results[-1]['rows_per_s']/all_results[-2]['rows_per_s']-1<.10:break
    chosen=max(all_results,key=lambda r:r['rows_per_s'])['workers']
    write(a.output/'result.json',dict(status='HARDWARE_COLLECTION_MEASURED',results=all_results,
        selected_for_next_full_chain_qualification=chosen,formal_N_locked=False,
        selection_is_not_final_execution_lock=True,wall_s=time.monotonic()-start_all))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker-id',type=int);p.add_argument('--address');a=p.parse_args()
    if a.worker_id is None:central(a)
    else:worker_main(a)


if __name__=='__main__':main()
