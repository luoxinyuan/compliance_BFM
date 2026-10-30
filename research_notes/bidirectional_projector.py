"""Balanced two-demonstration alignment. Original student/latent/gate unchanged."""
import math
import numpy as np
import torch
from contact_projector import ContactGate, Projector, deploy_input, constrained, check_deadline, save

def balanced_ids(device):
    return torch.cat([torch.randperm(60,device=device)[:6],60+torch.randperm(60,device=device)[:6]])

def fit_balanced(model,standing,episodes,contract,keys,out,public_act):
    assert [e['task']['direction'] for e in episodes]==[1,-1]
    assert all(e['task']['onset_s']==1.1 and e['task']['distance_m']==.035 for e in episodes)
    assert all(len(e['decisions'])==60 and len(e['control'])==300 and not e['failed'] for e in episodes)
    dev=standing.device;controls=sum([e['control'] for e in episodes],[]);ds=sum([e['decisions'] for e in episodes],[])
    obs={k:torch.tensor([c['bfm_obs'][k][0] for c in controls],device=dev,dtype=torch.float32) for k in keys}
    dobs={k:torch.tensor([d['bfm_obs'][k][0] for d in ds],device=dev,dtype=torch.float32) for k in keys}
    h=deploy_input(model,dobs,[d['force_history'] for d in ds],keys).detach()
    target=torch.tensor([c['q_PD'] for c in controls],device=dev);target_action=torch.tensor([c['action'] for c in controls],device=dev)
    normalized={k:v.detach() for k,v in model._normalize(obs).items()}
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    def q_of(a):return offset+a.clamp(-1,1)*scale
    gs=[]
    for e in episodes:
        gate=ContactGate()
        for i,d in enumerate(e['decisions']):
            gate.update(d['tether_time'],d['force_xyz_n'])
            gs.extend(gate.value(c['tether_time']) for c in e['control'][i*5:i*5+5])
    g=torch.tensor(gs,device=dev)
    phase=np.array(['pre' if c['tether_time']<1.1 else 'ramp' if c['tether_time']<3.1 else 'hold' if c['tether_time']<4.5 else 'release' for c in controls])
    weights=torch.tensor([5. if p in ('ramp','hold') else 2. if p=='release' else 1. for p in phase],device=dev)
    def evaluate(raw,ids):
        ix=(ids[:,None]*5+torch.arange(5,device=dev)).reshape(-1)
        z,t=constrained(model,standing,raw.repeat_interleave(5,0),g[ix])
        a=model._actor({k:v[ix] for k,v in normalized.items()},z,model.cfg.actor_std).mean.float()
        err=q_of(a)-target[ix]
        loss=((err[:,22:].square().mean(-1)+.1*err[:,:22].square().mean(-1))*weights[ix]).sum()/weights[ix].sum()+1e-6*raw.tanh().square().mean()
        return loss,a,err,z,t
    with torch.no_grad():
        public=public_act({k:v[:5] for k,v in obs.items()},standing.expand(5,-1),mean=True)
        direct=model._actor({k:v[:5] for k,v in normalized.items()},standing.expand(5,-1),model.cfg.actor_std).mean.float()
    parity=float((public-direct).abs().max());assert parity<1e-6
    save(out,'dataset-manifest.json',dict(decisions=120,control_samples=600,per_direction_decisions=60,keys=keys,student_dim=h.shape[-1],
        inputs='Only frozen-normalized state/last_action/history_actor and last3 measured force XYZ /20; no anchor, direction, clock, future target or teacher correction',
        optimizer=dict(name='Adam',lr=.001,updates=2000,batch_decisions=12,per_direction=6,seed=0,grad_clip=1.),
        initialization='new same small MLP, zero output; balanced normalization statistics; old checkpoint diagnostic only',
        loss='unchanged: arm q_PD MSE +0.1 nonarm MSE +1e-6 tanh(raw)^2; phase weights active5/release2/pre1',
        public_actor_parity=parity,phase=phase.tolist(),gates=gs,heldout_used=False))
    torch.save(dict(h=h.cpu(),obs={k:v.cpu() for k,v in obs.items()},target_q=target.cpu(),target_action=target_action.cpu(),gate=g.cpu()),out/'teacher-dataset.pt')
    torch.manual_seed(0);net=Projector(h).to(dev);opt=torch.optim.Adam(net.parameters(),lr=.001);trace=[];allids=torch.arange(120,device=dev)
    def stats():
        with torch.no_grad():
            raw=net(h);loss,act,err,z,t=evaluate(raw,allids);metrics={}
            for name,start in [('plus',0),('minus',300)]:
                metrics[name]={}
                for label in ['all','pre','ramp','hold','release']:
                    chosen=np.zeros(600,dtype=bool);chosen[start:start+300]=True
                    if label!='all':chosen &= phase==label
                    ix=torch.tensor(chosen,device=dev)
                    metrics[name][label]=dict(arm_q_rms=float(err[ix,22:].square().mean().sqrt()),nonarm_q_rms=float(err[ix,:22].square().mean().sqrt()),
                        full_action_rms=float((act[ix]-target_action[ix]).square().mean().sqrt()),q_peak=float(err[ix].abs().max()))
            return dict(loss=float(loss),directions=metrics,z_norm_max_error=float((z.norm(dim=-1)-16).abs().max()),
                angle_max_deg=float(torch.rad2deg(torch.acos(((z*standing).sum(-1)/256).clamp(-1,1))).max()),
                tangent_dot_standing_max=float((t*standing).sum(-1).abs().max()),raw_tanh_max=float(raw.tanh().abs().max()))
    before=stats();save(out,'fit-before.json',before)
    with torch.enable_grad():
        for step in range(2000):
            check_deadline();ids=balanced_ids(dev);opt.zero_grad();loss,*_=evaluate(net(h[ids]),ids);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            assert torch.isfinite(gn) and all(p.grad is None for p in model.parameters())
            if step==0:assert float(gn)>0
            opt.step()
            if step%100==0 or step==1999:
                row=dict(step=step+1,grad_norm=float(gn),**stats());trace.append(row);save(out,'fit-trace.json',trace)
                print('FIT',step+1,row['loss'],flush=True)
    after=stats();save(out,'fit-after.json',after)
    torch.save(dict(state_dict=net.state_dict(),updates=2000,contract=contract,hidden=128),out/'contact-projector-final.pt')
    perturb={}
    with torch.no_grad():
        raw=net(h);_,aa,_,_,_=evaluate(raw,allids)
        for name,factor in [('force_zero',0.),('force_reverse',-1.)]:
            hx=h.clone();hx[:,-9:]*=factor;rr=net(hx);_,ab,_,_,_=evaluate(rr,allids);perturb[name]={}
            for label,start in [('plus',0),('minus',300)]:
                mask=(torch.arange(600,device=dev)>=start)&(torch.arange(600,device=dev)<start+300)&(g>0)
                perturb[name][label]=dict(active_right_arm_q_rms=float((q_of(ab)[mask,22:]-q_of(aa)[mask,22:]).square().mean().sqrt()),
                    mean_right_arm_delta=(q_of(ab)[mask,22:]-q_of(aa)[mask,22:]).mean(0).tolist())
    save(out,'contact-input-perturbation.json',perturb)
    assert after['z_norm_max_error']<1e-4 and after['angle_max_deg']<=math.degrees(math.atan(.3))+.001
    net.eval().requires_grad_(False)
    for p in net.parameters():p.grad=None
    return net,h,dict(updates=2000,before=before,after=after,perturb=perturb)
