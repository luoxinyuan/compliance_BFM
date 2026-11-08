"""Bounded real-physics parity/throughput probe; no RL or change to control math."""
import argparse, cProfile, hashlib, json, os, pstats, time
from pathlib import Path
import m11_ipc

ROOT = Path(__file__).resolve().parents[1]
S = ROOT.parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def study(env, model, contract, output):
    import mujoco
    import numpy as np
    import torch
    from clean_policy import Expert
    from p4000_env import Task, install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    install()
    write(output/'runtime-speed.json', speed(env))
    folder = S.parent/'native-transfer-20261107/compatibility-prefix'
    write(output/'backend.json', backend(env, model, folder))
    plan = json.loads((ROOT/'research_notes/p4000_protocol.json').read_text())
    ckpath = Path(plan['initial']['path'])
    assert hashlib.sha256(ckpath.read_bytes()).hexdigest() == plan['initial']['sha256']
    checkpoint = torch.load(ckpath, map_location='cuda', weights_only=True)
    net = Expert(1088, 'direct').cuda()
    net.load_state_dict(checkpoint['state_dict'])
    net.eval().requires_grad_(False)
    task = Task(env, model, contract, 5290)
    cases = [('normal72',0.), ('short08',.008), ('stop66',0.), ('none',.008)]
    rows, reports, profiles = [], [], []
    total_physics = 0
    act_count = 0
    original_act = model.act

    def counted(*args, **kwargs):
        nonlocal act_count
        act_count += 1
        return original_act(*args, **kwargs)

    model.act = counted

    def run(case, noise, fast):
        nonlocal total_physics
        task.record_nominal_diagnostic = not fast
        label = case+('__fast' if fast else '__full')
        calls_before = act_count
        started = time.perf_counter()
        obs = task.reset(plan['dev'][case], record=True, label=label)
        reset_seconds = time.perf_counter()-started
        generator = torch.Generator(device=obs.device).manual_seed(5290)
        profiler = cProfile.Profile()
        snapshots, rewards, control_arrays, timings = [], [], [], []
        count = 0
        while not task.done:
            raw = net.distribution(obs).mean
            if noise:
                raw = raw + noise*torch.randn(raw.shape, device=raw.device, generator=generator)
            profiled = count < 16
            if profiled:
                profiler.enable()
            torch.cuda.synchronize()
            started = time.perf_counter()
            obs, reward, done, stats = task.step(raw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if profiled:
                profiler.disable()
            else:
                timings.append(elapsed)
            state = np.zeros(mujoco.mj_stateSize(env.sim.model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(env.sim.model, env.sim.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            # Full solver/integration state + actual BFM history and commands;
            # optional diagnostic logs are deliberately not treated as control.
            snap = [state, obs[0].cpu().numpy(), task.last_command[0].cpu().numpy(),
                    task.lastq[0].cpu().numpy(), env.previous_action[0].cpu().numpy()]
            snap += [env.obs[k].cpu().numpy().reshape(-1) for k in env.keys]
            snapshots.append(np.concatenate(snap))
            rewards.append(reward)
            for c in task.controls[-stats['control_steps']:]:
                control_arrays.append(np.r_[c['time'], np.asarray(c['executed_z']).ravel(),
                                           c['action'], c['q_PD'], c['pre_qpos'], c['pre_qvel']])
            total_physics += stats['physics_steps']
            count += 1
        # All original 200Hz physical fields are preserved and compared using
        # their canonical JSON encoding: includes actual forces, feet/support,
        # anchors and state, rather than just final task scores.
        physical_digest = hashlib.sha256(json.dumps(task.physics,sort_keys=True,allow_nan=False).encode()).hexdigest()
        profile = pstats.Stats(profiler)
        entries = []
        for (file,line,func),(cc,nc,tt,ct,callers) in profile.stats.items():
            entries.append(dict(file=file,line=line,function=func,primitive_calls=cc,calls=nc,
                                exclusive_seconds=tt,inclusive_seconds=ct))
        entries.sort(key=lambda x:x['inclusive_seconds'], reverse=True)
        profiles.append(dict(label=label,profiled_decisions=min(16,count),entries=entries[:25],
                             warning='CPU inclusive timings overlap; not GPU compute times and not additive'))
        timed_seconds=sum(timings)
        result = dict(label=label, fast=fast, control_decisions=count,
                      unprofiled_step_decisions=len(timings),unprofiled_step_seconds=timed_seconds,
                      unprofiled_step_decisions_per_second=len(timings)/timed_seconds,
                      reset_seconds=reset_seconds, frozen_BFM_act_calls=act_count-calls_before,
                      physical_sha256=physical_digest, episode_return=task.episode_return,
                      boundary=task.boundary, diagnostic_omissions=['nominal_q_PD','qPD_vs_prior_rms_deg'] if fast else [],
                      physics_steps=len(task.physics))
        rows.append(result)
        write(output/'timings.json',rows)
        return dict(snapshots=np.stack(snapshots), rewards=np.array(rewards),
                    control=np.stack(control_arrays),physical_digest=physical_digest,row=result)

    try:
        for index,(case,noise) in enumerate(cases):
            # Alternate order to limit cache/order bias. Early profiled steps
            # are excluded from throughput; numerical parity covers all steps.
            order = [False,True] if index%2==0 else [True,False]
            executions={fast:run(case,noise,fast) for fast in order}
            full,fast=executions[False],executions[True]
            differences={k:float(np.max(np.abs(full[k]-fast[k]))) for k in ['snapshots','rewards','control']}
            parity=all(v==0. for v in differences.values()) and full['physical_digest']==fast['physical_digest']
            reports.append(dict(case=case,parity=parity,max_difference=differences,
                                all_200Hz_physics_rows_bitwise_equal=full['physical_digest']==fast['physical_digest'],
                                full_steps_per_second=full['row']['unprofiled_step_decisions_per_second'],
                                fast_steps_per_second=fast['row']['unprofiled_step_decisions_per_second']))
            write(output/'parity.json',reports)
            print('SPEED_PARITY',reports[-1],flush=True)
        full=[r for r in rows if not r['fast']]
        fast=[r for r in rows if r['fast']]
        gain=sum(r['unprofiled_step_seconds'] for r in full)/sum(r['unprofiled_step_seconds'] for r in fast)
        eligible=all(x['parity'] for x in reports) and gain>=1.05
        result=dict(status='completed',adopt_fast_diagnostics=eligible,aggregate_unprofiled_step_speedup=gain,
                    physics_parity_all_cases=all(x['parity'] for x in reports),cases=reports,
                    new_physics_steps=total_physics,RL_decisions=0,new_optimizer_updates=0,
                    selection='zero physical/control/reward/observation difference in all four cases and aggregate gain>=1.05',
                    scope='Single sampler measured, not an established three-worker end-to-end training speedup',
                    limits='Does not change worker count, batch, horizon, BFM math, reward or Gaussian probabilities')
        write(output/'cprofile.json',profiles)
        write(output/'probe-result.json',result)
        print('SPEED_RESULT',result,flush=True)
    finally:
        model.act=original_act
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',BFM_RECOVERY_OUTPUT=str(args.output),
                      MUJOCO_GL='disable',CLEAN_KIND='direct',P4000_SEED='5201',
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    os.chdir(args.output)
    import m10_stage,m10_physics
    m10_physics.OUT=args.output
    m10_stage.run_study=study
    m10_physics.main()


if __name__=='__main__':
    main()
