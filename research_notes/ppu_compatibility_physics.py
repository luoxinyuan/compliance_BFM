"""Bounded PPU compatibility check; no policy/BFM training and no CPU inference fallback."""
import os
os.environ.update(MUJOCO_GL='disable', CUDA_VISIBLE_DEVICES='0', CLEAN_KIND='direct',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, json, time, hashlib
from pathlib import Path
BASE=Path('/root/compliance-ppu-20261106')
ROOT=BASE/'repo'
OUT=BASE/'output'/os.environ.get('PPU_CHECK_ATTEMPT','physical-01')
os.environ['BFM_RECOVERY_OUTPUT']=str(OUT)
sys.path[:0]=[str(ROOT/'research_notes'),str(ROOT)]

def write(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))

def study(env,model,contract,out):
    import numpy as np, torch
    from clean_policy import Expert, direct_command
    start=time.monotonic()
    from m11_runtime_speed import install
    write('runtime-speed.json',install(env))
    saved=json.loads((BASE/'output/nvidia-saved-reference.json').read_text())
    comparisons=[]
    for identity,group in saved.items():
        for r in group['controls']:
            obs={k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in r['bfm_obs'].items()}
            z=torch.tensor(r['executed_z'],dtype=torch.float32,device='cuda')
            a=model.act(obs,z,mean=True)
            expected=torch.tensor(r['action'],dtype=torch.float32,device='cuda')[None]
            q=torch.tensor(contract['q_PD_offset'],device='cuda')+(a*5).clamp(-contract['clip'],contract['clip'])*torch.tensor(contract['q_PD_scale'],device='cuda')/5
            eq=torch.tensor(r['q_PD'],device='cuda')
            comparisons.append(dict(model=identity,time=r['time'],action_max_abs=float((a-expected).abs().max()),
                                    qpd_max_abs_deg=float(torch.rad2deg(q-eq).abs().max())))
    write('saved-nvidia-action-comparison.json',comparisons)
    r=saved['C0']['controls'][3]
    obs={k:torch.tensor(v,dtype=torch.float32,device='cuda') for k,v in r['bfm_obs'].items()}
    with torch.enable_grad():
        raw=torch.tensor(r['executed_z'],device='cuda',requires_grad=True)
        a=model._actor(model._normalize(obs),model.project_z(raw),model.cfg.actor_std).mean.float()
        loss=a.square().mean()
        grad=torch.autograd.grad(loss,raw)[0]
        assert torch.isfinite(grad).all() and grad.norm()>0
        write('bfm-latent-gradient.json',dict(loss=float(loss.detach()),gradient_norm=float(grad.norm()),
              gradient_max=float(grad.abs().max()),device=str(grad.device),bfm_updated=False))
    p=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
    identities={'I':(p/'m39a-demand-20261104/B-ppo-03/job/best.pt','72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a'),
                'C0':(p/'prior-recovery-20261105/correct-C0/C0.pt','ffe0eb0666da12c99788e039061b27e8738051c276a3ca41bebf656d9a96352c')}
    teacher_checks=[]
    for name,(path,sha) in identities.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        ck=torch.load(path,map_location='cuda',weights_only=True)
        net=Expert(ck['input_dim'],'direct').to('cuda')
        net.load_state_dict(ck['state_dict']);net.eval().requires_grad_(False)
        differences=[]
        for d in saved[name]['decisions']:
            x=torch.tensor([d['obs']],device='cuda',dtype=torch.float32)
            raw=net.distribution(x).mean
            ref=torch.tensor([d['raw']],device='cuda',dtype=torch.float32)
            differences.append(float((raw-ref).abs().max()))
            assert torch.isfinite(direct_command(raw,env.standing)).all()
        teacher_checks.append(dict(name=name,sha256=sha,input_dim=ck['input_dim'],max_saved_raw_difference=max(differences)))
    write('teacher-forward-comparison.json',teacher_checks)
    # New compatibility-only physical episode, NOT an Issue47 takeover or a replay of an old benchmark.
    # Same original BFM, PD, native 200Hz MuJoCo, 50Hz actor, 10Hz high-level stepping.
    # First settle with force disabled; align the new anchor ONCE at contact onset.
    physics_start=time.monotonic()
    env.interruption_window=None;env.m1_anchor_hook=None;env.qualification_active=False
    for i in range(30):
        _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True)
        assert not done,'Standing prefill terminated'
    env.qualification_active=True
    point=env.tether.point_state()[0]
    env.tether.origin=point.copy();env.tether.delta=np.array([.035,0,0])
    for c in (env.cfg,env.tether.cfg):
        c.enabled=True;c.activation_time=3.;c.anchor_delay=0.;c.ramp_duration=2.;c.release_time=6.;c.max_force=5.
    for i in range(50):
        _,_,done,_=env.step(torch.zeros_like(env.standing),baseline=True)
        assert not done,'Compatibility contact episode terminated'
    rows=env.trace
    assert rows and all(np.isfinite(r['qpos']).all() for r in rows)
    active=[r for r in rows if r['active']]
    stats=dict(physics_steps=len(rows),simulated_seconds=float(env.tether.time),physics_wall_seconds=time.monotonic()-physics_start,
               active_steps=len(active),force_rms=float(np.sqrt(np.mean([np.dot(r['force'],r['force']) for r in active]))),
               force_peak=max(float(np.linalg.norm(r['force'])) for r in active),
               min_root_height=min(r['height'] for r in rows),max_tilt_deg=max(r['tilt_deg'] for r in rows),
               max_joint_limit_excess_rad=max(r['joint_limit_excess_max_rad'] for r in rows),
               any_failed=any(r['failed'] for r in rows),
               max_torque_saturation=max(r['torque_saturation'] for r in rows),
               active_tracking_rms_cm=100*float(np.sqrt(np.mean([r['error']**2 for r in active]))),
               episode='New standing +X3.5cm compatibility probe, 3s onset/2s ramp/6s release/8s end; not a policy performance benchmark')
    write('physical-summary.json',stats)
    import gzip
    with gzip.open(out/'physics.json.gz','wt') as f:json.dump(rows,f)
    torch.cuda.synchronize()
    return dict(status='completed',student_optimizer_updates=0,rl_decisions=0,bfm_updated=False,
                maximum_saved_action_difference=max(r['action_max_abs'] for r in comparisons),
                maximum_saved_qpd_difference_deg=max(r['qpd_max_abs_deg'] for r in comparisons),
                physical=stats,elapsed_s=time.monotonic()-start,
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2)

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    os.chdir(OUT)
    import m10_stage,m10_physics
    m10_stage.run_study=study
    try:m10_physics.main()
    except BaseException:
        import traceback
        (OUT/'failure.txt').write_text(traceback.format_exc())
        raise
