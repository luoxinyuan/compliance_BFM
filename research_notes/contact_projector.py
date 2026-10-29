"""One-scene supervised contact projector. No anchor/time/oracle input to MLP."""
import json, math, os, time
import numpy as np
import torch
from torch import nn

class ContactGate:
    """10Hz force-contact detection, same causal .2s release at 50Hz; no schedule."""
    def __init__(self): self.active=False; self.release=None
    def update(self, now, force):
        sensed=bool(np.linalg.norm(force)>1e-6)
        if sensed:self.release=None
        elif self.active:self.release=now
        self.active=sensed
        return sensed
    def value(self, now):
        if self.active:return 1.
        return 0. if self.release is None else max(0.,1.-(now-self.release)/.2)

def deploy_input(model, obs, history, keys):
    # Same frozen actor normalizer. Only whitelisted proprio/history + force history.
    assert set(keys)<=set(('state','last_action','history_actor'))
    normalized=model._normalize(obs)
    proprio=torch.cat([normalized[k].reshape(len(obs[k]),-1).float() for k in keys],-1)
    force=torch.as_tensor(history,dtype=torch.float32,device=proprio.device).reshape(len(proprio),-1)/20.
    return torch.cat((proprio,force),-1)

def constrained(model, standing, raw, gate):
    r=raw.tanh();t=r-(r*standing).sum(-1,keepdim=True)/standing.square().sum(-1,keepdim=True)*standing
    g=torch.as_tensor(gate,dtype=raw.dtype,device=raw.device).reshape(-1,1)
    z=model.project_z(standing+.3*g*t)
    # At zero gate preserve standing bitwise without severing active gradients.
    z=torch.where(g==0,standing.expand_as(z),z)
    return z,t

class Projector(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer('center',x.mean(0));self.register_buffer('scale',x.std(0,unbiased=False).clamp_min(.1))
        self.net=nn.Sequential(nn.Linear(x.shape[-1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,256))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net((x-self.center)/self.scale)

def check_deadline():
    assert time.monotonic()<float(os.environ['CCL_DEADLINE_MONOTONIC']), 'Budget exhausted'

def save(out,name,obj): (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))

def fit(model,standing,episode,contract,keys,out,public_act):
    """Use all five ACTUAL 50Hz observations/targets for each held 10Hz raw latent."""
    dev=standing.device;controls=episode['control'];ds=episode['decisions'];assert len(ds)==60 and len(controls)==300
    obs={k:torch.tensor([c['bfm_obs'][k][0] for c in controls],device=dev,dtype=torch.float32) for k in keys}
    dobs={k:torch.tensor([d['bfm_obs'][k][0] for d in ds],device=dev,dtype=torch.float32) for k in keys}
    h=deploy_input(model,dobs,[d['force_history'] for d in ds],keys).detach()
    assert h.shape[0]==60
    target=torch.tensor([c['q_PD'] for c in controls],device=dev)
    target_action=torch.tensor([c['action'] for c in controls],device=dev)
    normalized={k:v.detach() for k,v in model._normalize(obs).items()}
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    def q_of(a):return offset+a.clamp(-1,1)*scale
    gate=ContactGate();g=[];force_gate=[]
    for i,d in enumerate(ds):
        gate.update(d['tether_time'],d['force_xyz_n'])
        for c in controls[i*5:i*5+5]:g.append(gate.value(c['tether_time']));force_gate.append(min(1.,np.linalg.norm(d['force_history'][-1])/5.))
    g=torch.tensor(g,device=dev);fg=torch.tensor(force_gate,device=dev)
    phase=np.array(['pre' if c['tether_time']<1.1 else 'ramp' if c['tether_time']<3.1 else 'hold' if c['tether_time']<4.5 else 'release' for c in controls])
    weights=torch.tensor([5. if p in ('ramp','hold') else 2. if p=='release' else 1. for p in phase],device=dev)
    def evaluate(raw,decisions,chosen_gate=g):
        idx=(decisions[:,None]*5+torch.arange(5,device=dev)).reshape(-1)
        z,t=constrained(model,standing,raw.repeat_interleave(5,0),chosen_gate[idx])
        a=model._actor({k:v[idx] for k,v in normalized.items()},z,model.cfg.actor_std).mean.float()
        err=q_of(a)-target[idx];arm=err[:,22:].square().mean(-1);other=err[:,:22].square().mean(-1)
        loss=((arm+.1*other)*weights[idx]).sum()/weights[idx].sum()+1e-6*raw.tanh().square().mean()
        return loss,a,err,z,t
    allids=torch.arange(60,device=dev)
    # Gradient path exactly agrees with public inference on identical o,z.
    with torch.no_grad():
        public=public_act({k:v[:5] for k,v in obs.items()},standing.expand(5,-1),mean=True)
        direct=model._actor({k:v[:5] for k,v in normalized.items()},standing.expand(5,-1),model.cfg.actor_std).mean.float()
    parity=float((public-direct).abs().max());assert parity<1e-6
    save(out,'dataset-manifest.json',dict(decisions=60,control_samples=300,student_dim=h.shape[-1],keys=keys,
        student_inputs='frozen-normalized state/last_action/history_actor + last 3 measured force XYZ /20; no clock, anchor, wrist-world, Jacobian, teacher residual, latent label or previous_z',
        target='actual teacher full29D action/q_PD on its own 50Hz state, 5 frames per single10Hz raw latent',
        gate='measured force norm>1e-6 at10Hz, active1, causal .2s release at50Hz; ideal simulator sensor proxy, real sensor robustness untested',
        original_force_gate='clip(last measured force norm/5,0,1), held10Hz; diagnostic comparator only',
        gates=g.tolist(),original_force_gates=fg.tolist(),phase=phase.tolist(),public_actor_parity=parity,
        optimizer=dict(name='Adam',lr=.001,updates=2000,batch_decisions=12,seed=0,grad_clip=1.),
        loss='q_PD arm MSE +0.1 nonarm MSE +1e-6 tanh(raw)^2, active phase weight5/release2/pre1',
        projection='256D tanh -> standing-tangent -> residual_scale .3 -> original sphere norm16; no execution mask',
        data_scope='one chosen old development trajectory; deliberately overfit; no heldout/generalization claim'))
    torch.save(dict(h=h.cpu(),obs={k:v.cpu() for k,v in obs.items()},target_q=target.cpu(),target_action=target_action.cpu(),gate=g.cpu()),out/'teacher-dataset.pt')
    probes=[]
    with torch.enable_grad():
        for i in [11,12,13,31,35,44,46,47]:
            for name,gate_values in [('original_force',fg),('contact_release',g)]:
                u=torch.zeros(1,256,device=dev,requires_grad=True);opt=torch.optim.Adam([u],lr=.05);ids=torch.tensor([i],device=dev)
                _,_,e0,_,_=evaluate(u,ids,gate_values);initial=float(e0[:,22:].square().mean().sqrt())
                for step in range(200):
                    check_deadline();opt.zero_grad();loss,*_=evaluate(u,ids,gate_values);loss.backward();opt.step()
                _,aa,err,zz,tt=evaluate(u,ids,gate_values)
                probes.append(dict(decision=i,time=ds[i]['time'],gate=name,gate_five=gate_values[i*5:i*5+5].tolist(),initial_arm_q_rms=initial,
                    arm_q_rms=float(err[:,22:].square().mean().sqrt()),nonarm_q_rms=float(err[:,:22].square().mean().sqrt()),
                    per_frame_arm_rms=err[:,22:].square().mean(-1).sqrt().tolist(),raw_tanh_max=float(u.tanh().abs().max()),z_norm=zz.norm(dim=-1).tolist()))
                print('PROBE',json.dumps(probes[-1]),flush=True)
    save(out,'projection-spot-checks.json',probes)
    torch.manual_seed(0);net=Projector(h).to(dev);opt=torch.optim.Adam(net.parameters(),lr=.001);trace=[]
    def stats():
        with torch.no_grad():
            raw=net(h);loss,act,err,z,t=evaluate(raw,allids)
            metrics={}
            for label in ['all','pre','ramp','hold','release']:
                idx=torch.ones(300,dtype=torch.bool,device=dev) if label=='all' else torch.tensor(phase==label,device=dev)
                metrics[label]=dict(arm_q_rms=float(err[idx,22:].square().mean().sqrt()),nonarm_q_rms=float(err[idx,:22].square().mean().sqrt()),
                    full_action_rms=float((act[idx]-target_action[idx]).square().mean().sqrt()),q_peak=float(err[idx].abs().max()))
            return dict(loss=float(loss),phases=metrics,z_norm_max_error=float((z.norm(dim=-1)-16).abs().max()),
                angle_max_deg=float(torch.rad2deg(torch.acos(((z*standing).sum(-1)/256).clamp(-1,1))).max()),
                tangent_dot_standing_max=float((t*standing).sum(-1).abs().max()),raw_tanh_max=float(raw.tanh().abs().max()))
    before=stats();save(out,'fit-before.json',before)
    with torch.enable_grad():
        for step in range(2000):
            check_deadline();ids=torch.randperm(60,device=dev)[:12];raw=net(h[ids]);opt.zero_grad();loss,*_=evaluate(raw,ids)
            loss.backward();gn=nn.utils.clip_grad_norm_(net.parameters(),1.)
            assert torch.isfinite(gn) and all(p.grad is None for p in model.parameters())
            if step==0:assert float(gn)>0,'No gradient through frozen BFM to projector'
            opt.step()
            if step%100==0 or step==1999:
                row=dict(step=step+1,grad_norm=float(gn),**stats());trace.append(row);print('FIT',json.dumps(row),flush=True)
                save(out,'fit-trace.json',trace)
    after=stats();save(out,'fit-after.json',after)
    torch.save(dict(state_dict=net.state_dict(),updates=2000,contract=contract,hidden=128),out/'contact-projector-final.pt')
    # Paired same-h counterfactuals isolate force channels, holding contact gate fixed.
    perturb={}
    with torch.no_grad():
        original=net(h);_,aa,_,_,_=evaluate(original,allids)
        for name,factor in [('force_zero',0.),('force_reverse',-1.)]:
            hx=h.clone();hx[:,-9:]*=factor;rr=net(hx);_,ab,_,_,_=evaluate(rr,allids)
            active=(g>0)
            perturb[name]=dict(raw_rms=float((rr-original).square().mean().sqrt()),active_q_PD_rms=float((q_of(ab)[active]-q_of(aa)[active]).square().mean().sqrt()),
                active_right_arm_q_rms=float((q_of(ab)[active,22:]-q_of(aa)[active,22:]).square().mean().sqrt()))
    save(out,'contact-input-perturbation.json',perturb)
    assert after['z_norm_max_error']<1e-4 and after['angle_max_deg']<=math.degrees(math.atan(.3))+.001
    return net,h,dict(updates=2000,before=before,after=after,perturb=perturb)
