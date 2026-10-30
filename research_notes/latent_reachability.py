"""Temporary bounded latent fits on saved C states. No policy fitting or new rollout."""
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from contact_projector import ContactGate,constrained


def gates_from_trace(episode):
    gate=ContactGate();values=[];forces=[]
    for high,d in enumerate(episode['decisions']):
        force=episode['initial']['force'] if high==0 else episode['physics'][20*high-1]['force']
        gate.update(d['time'],force);forces.append(force)
        values.extend(gate.value(c['tether_time']) for c in episode['control'][5*high:5*high+5])
    return np.array(values),forces


def diagnose(env,model,contract,out):
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/'research_notes/causal_protocol.json').read_text())
    config=protocol['reachability'];source=Path(protocol['C_source'])
    assert hashlib.sha256((source/'accumulated.json').read_bytes()).hexdigest()==protocol['C_episode_sha256']
    assert hashlib.sha256((source/'accumulated-initial-state.json').read_bytes()).hexdigest()==protocol['C_state_sha256']
    ep=json.loads((source/'accumulated.json').read_text());cs=ep['control'];dev=env.standing.device;standing=env.standing
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    gates,forces=gates_from_trace(ep)
    ids=[round((t-3)*10) for t in protocol['selected_times_s']]
    assert len(set(ids))==len(ids)
    # Every independent fit holds one raw vector over the original five successive observations.
    cases=[]
    for i in ids:
        assert abs(cs[5*i]['tether_time']-(3+i/10))<1e-8
        modes=['deployed']+(['gate_one_diagnostic'] if min(gates[5*i:5*i+5])<1.-1e-8 else [])
        for mode in modes:
            for weight in [.1,1.]:
                for init in ['zero','random']:
                    cases.append(dict(decision=i,time=cs[5*i]['tether_time'],mode=mode,nonarm_weight=weight,init=init))
    index=torch.tensor([[5*c['decision']+j for j in range(5)] for c in cases],device=dev)
    flat=index.flatten();keys=env.keys
    allobs={k:torch.tensor([c['bfm_obs'][k][0] for c in cs],device=dev,dtype=torch.float32) for k in keys}
    obs={k:v[flat] for k,v in allobs.items()};norm={k:v.detach() for k,v in model._normalize(obs).items()}
    target=torch.tensor([c['q_PD'] for c in cs],device=dev,dtype=torch.float32)[index]
    target_a=torch.tensor([c['action'] for c in cs],device=dev,dtype=torch.float32)[index]
    original_z=torch.tensor([c['executed_z'][0] for c in cs],device=dev,dtype=torch.float32)[flat]
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    q_of=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    gate_values=torch.tensor([gates[5*c['decision']:5*c['decision']+5] if c['mode']=='deployed' else np.ones(5) for c in cases],device=dev,dtype=torch.float32)
    weights=torch.tensor([c['nonarm_weight'] for c in cases],device=dev)
    with torch.no_grad():
        public=model.act(obs,standing.expand(len(flat),-1),mean=True)
        direct=model._actor(norm,standing.expand(len(flat),-1),model.cfg.actor_std).mean.float()
        parity=float(abs(public-direct).max());assert parity<1e-6
        original=model.act(obs,original_z,mean=True)
        old_q=torch.tensor([c['goal_q_PD'] for c in cs],device=dev,dtype=torch.float32)[flat]
        teacher_parity=float(abs(q_of(original)-old_q).max());assert teacher_parity<2e-6
    torch.manual_seed(0)
    raw=torch.stack([torch.zeros(256,device=dev) if c['init']=='zero' else .25*torch.randn(256,device=dev) for c in cases]).requires_grad_(True)

    def evaluate():
        z,t=constrained(model,standing,raw.repeat_interleave(5,0),gate_values.flatten())
        a=model._actor(norm,z,model.cfg.actor_std).mean.float();q=q_of(a)
        error=q.reshape(-1,5,29)-target
        per=error[:,:,22:].square().mean((1,2))+weights*error[:,:,:22].square().mean((1,2))
        return per,a.reshape(-1,5,29),error,z.reshape(-1,5,256)

    def metrics(a,error,z):
        rows=[]
        for i,case in enumerate(cases):
            e=error[i];angles=torch.rad2deg(torch.acos(((z[i]*standing).sum(-1)/256).clamp(-1,1)))
            rows.append(dict(**case,gate_five=gate_values[i].tolist(),arm_q_RMS_deg=float(torch.rad2deg(e[:,22:].square().mean().sqrt())),
                nonarm_q_RMS_deg=float(torch.rad2deg(e[:,:22].square().mean().sqrt())),
                legs_q_RMS_deg=float(torch.rad2deg(e[:,:12].square().mean().sqrt())),waist_q_RMS_deg=float(torch.rad2deg(e[:,12:15].square().mean().sqrt())),
                max_joint_deg=float(torch.rad2deg(e.abs().max())),action_RMS=float((a[i]-target_a[i]).square().mean().sqrt()),
                arm_action_RMS=float((a[i,:,22:]-target_a[i,:,22:]).square().mean().sqrt()),
                nonarm_action_RMS=float((a[i,:,:22]-target_a[i,:,:22]).square().mean().sqrt()),
                per_frame_arm_q_RMS_deg=torch.rad2deg(e[:,22:].square().mean(-1).sqrt()).tolist(),
                angle_max_deg=float(angles.max()),z_norm_error=float(abs(z[i].norm(dim=-1)-16).max()),
                action_clip_fraction=float((abs(a[i]*5)>=contract['clip']).float().mean()),raw_tanh_max=float(raw[i].tanh().abs().max())))
        return rows

    with torch.enable_grad():
        opt=torch.optim.Adam([raw],lr=config['lr'])
        with torch.no_grad():before_loss,a,e,z=evaluate();before=metrics(a,e,z)
        traces=[];best=before_loss.detach().clone();first_gradient=None
        for step in range(config['updates']):
            assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC'])
            opt.zero_grad();loss,*_=evaluate();loss.sum().backward()
            assert torch.isfinite(raw.grad).all() and all(p.grad is None for p in model.parameters())
            if step==0:first_gradient=raw.grad.norm(dim=-1).tolist()
            opt.step()
            with torch.no_grad():
                losses,_,_,_=evaluate();best=torch.minimum(best,losses)
                if step%50==0 or step==config['updates']-1:
                    traces.append(dict(step=step+1,loss_per_case=losses.tolist(),gradient_norm_per_case=raw.grad.norm(dim=-1).tolist()))
                    (out/'latent-optimization-trace.json').write_text(json.dumps(traces,indent=2))
                    print('LATENT_PROBE_STEP',step+1,'cases',len(cases),'loss_mean',float(losses.mean()),flush=True)
            time.sleep(config['optimization_throttle_sleep_s'])
        with torch.no_grad():final_loss,a,e,z=evaluate();after=metrics(a,e,z)
    assert max(r['angle_max_deg'] for r in after)<=math.degrees(math.atan(.3))+.002
    assert max(r['z_norm_error'] for r in after)<1e-4
    for r in after:
        r['screen_pass']=r['arm_q_RMS_deg']<=1 and r['nonarm_q_RMS_deg']<=1 and r['max_joint_deg']<=3
    result=dict(status='completed',selected_times=protocol['selected_times_s'],cases=len(cases),temporary_latent_optimizer_steps=config['updates'],
      policy_updates=0,public_actor_parity=parity,teacher_goal_forward_parity_rad=teacher_parity,
      baseline=before,final=after,best_observed_loss=best.tolist(),final_loss=final_loss.tolist(),first_gradient=first_gradient,
      gate_trace=gates.tolist(),decision_force=forces,
      limitation='All50Hz observations fixed to teacher trajectory; command matching does not establish closed-loop support or tracking. Finite optimization does not prove global impossibility; no radius expansion or free-z search.')
    (out/'latent-reachability.json').write_text(json.dumps(result,indent=2))
    torch.save(dict(raw=raw.detach().cpu(),z=z.detach().cpu(),target_q=target.cpu(),predicted_q=(e+target).detach().cpu(),predicted_action=a.detach().cpu(),cases=cases),out/'temporary-latent-solutions.pt')
    return dict(status='completed',states=len(ids),cases=len(cases),updates=config['updates'],policy_updates=0,public_actor_parity=parity,
                deployed_passes=sum(r['screen_pass'] for r in after if r['mode']=='deployed'))
