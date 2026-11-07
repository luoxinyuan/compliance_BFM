"""NVIDIA FP32 probability-contract regression on saved real observations.

No physics sampling or retained learning. A legacy actor can supply a nonzero
mean fixture, but its raw-space exploration std is never imported as whitened std.
"""
import argparse, copy, hashlib, json, subprocess, time
from pathlib import Path
import torch
from clean_policy import Expert as Clean, update
from native_policy import Expert as Native
from temporal_policy import Expert, rho_for
from geometry_policy import expand


def tensors_equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(tensors_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(tensors_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--legacy-mean-checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); assert not a.output.exists()
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    start=time.monotonic();x=torch.load(a.observations,map_location='cuda',weights_only=True)
    assert x.ndim==2 and x.shape[1]==1088 and x.dtype==torch.float32
    x=x[torch.arange(768,device='cuda')%len(x)]
    torch.manual_seed(7001)
    legacy=Clean(1088,'direct').cuda()
    if a.legacy_mean_checkpoint:
        expand(legacy,torch.load(a.legacy_mean_checkpoint,map_location='cuda',weights_only=True))
    else:
        legacy.actor[-1].weight.normal_(0,.01);legacy.actor[-1].bias.normal_(0,.05)
    fixture=Native(1088,'direct').cuda()
    fixture.actor.load_state_dict(legacy.actor.state_dict());fixture.critic.load_state_dict(legacy.critic.state_dict())
    fixture.center.copy_(legacy.center);fixture.scale.copy_(legacy.scale)
    assert torch.equal(fixture.log_std,torch.zeros_like(fixture.log_std))
    state=copy.deepcopy(fixture.state_dict())
    probe=Expert(1088).cuda()
    try: probe.initialize_from_raw(legacy.state_dict())
    except ValueError as e: guard=str(e)
    else: raise AssertionError('legacy raw-space state silently accepted')
    report=dict(status='running',source='real saved observations; numerical mean fixture, not native contact learning',
        logprob_threshold=3e-4,joint_KL_limit=.02,FP32=True,TF32=False,PPU_rowwise_path=False,
        loader_guard=guard,cases=[],updates=[],formal_RL_decisions=0,retained_updates=0,
        observations_sha256=hashlib.sha256(a.observations.read_bytes()).hexdigest())
    def save(): a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    for seed in [7003,7007,7011]:
        for delta in [0.,.0001]:
            for arm in ['independent','continuous']:
                n=Expert(1088,rho=rho_for(arm)).cuda();n.initialize_from_raw(state)
                torch.manual_seed(seed)
                D=torch.linalg.qr(torch.randn(256,4,device='cuda'))[0].T;n.set_covariance(D)
                if delta:n.actor[-1].weight.normal_(0,delta);n.actor[-1].bias.normal_(0,delta)
                previous=torch.zeros_like(x[:1]);pu=torch.zeros(1,256,device='cuda');valid=False
                px=[];us=[];logs=[];mus=[]
                for xx in x:
                    xx=xx[None];z=n.pack(xx,previous,pu,valid);d=n.distribution(z);u=d.sample()
                    px.append(z);us.append(u);logs.append(d.log_prob(u).sum(-1));mus.append(d.mean)
                    previous,pu,valid=xx.clone(),u.clone(),True
                px=torch.cat(px);us=torch.cat(us);logs=torch.cat(logs);mus=torch.cat(mus)
                db=n.distribution(px);err=float((db.log_prob(us).sum(-1)-logs).abs().max())
                C=n.raw_L@torch.diag(n.log_std.exp().square())@n.raw_L.T
                covariance_error=float((C-(torch.eye(256,device='cuda')*.008**2+.15**2*(D.T@D))).abs().max())
                with torch.enable_grad():
                    n.zero_grad();q=n.pack(x[:1],x[:1],torch.zeros_like(pu),True)
                    n.distribution(q).mean.sum().backward()
                    gradient_error=float((n.actor[-1].bias.grad-(1-rho_for(arm))/.008).abs().max())
                n.zero_grad()
                item=dict(seed=seed,delta_weight_std=delta,arm=arm,joint_logprob_max_error=err,
                    mean_batch_max_error=float((db.mean-mus).abs().max()),coordinate_std=float(n.log_std.exp().mean()),
                    covariance_max_error=covariance_error,history_bias_gradient_error=gradient_error)
                report['cases'].append(item);save();print(json.dumps(item),flush=True)
                assert err<3e-4 and gradient_error<3e-4
                assert covariance_error<1e-7
                if seed==7003 and delta==0:
                    before_logstd=n.log_std.clone();v=n.value(px)
                    ret=v+torch.randn_like(v);adv=torch.randn_like(v);adv=(adv-adv.mean())/adv.std()
                    batch=dict(obs=px,action=us,logp=logs,advantages=adv,returns=ret,value=v,raw_adv_std=adv.std())
                    initial=copy.deepcopy(n.state_dict())
                    opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=1e-5),dict(params=n.critic.parameters(),lr=3e-4)])
                    result=update(n,opt,batch,epochs=2)
                    assert result['optimizer_updates']>0
                    assert all(e['joint_gaussian_KL']<=.02001 for e in result['epochs'])
                    assert torch.equal(before_logstd,n.log_std)
                    n.load_state_dict(initial)
                    opt=torch.optim.Adam([dict(params=n.actor.parameters(),lr=10000.),dict(params=n.critic.parameters(),lr=3e-4)])
                    adam=copy.deepcopy(opt.state_dict())
                    rejected=update(n,opt,batch,epochs=1)
                    assert rejected['optimizer_updates']==0 and len(rejected['rejected_candidates'])==7
                    assert tensors_equal(initial,n.state_dict())
                    assert tensors_equal(adam['state'],opt.state_dict()['state'])
                    report['updates'].append(dict(arm=arm,discarded_update=result,
                        forced_rejection=rejected,parameter_rollback_exact=True,Adam_moments_rollback_exact=True,
                        note='retry LR intentionally reduced; means and optimizer moments restored'))
                    save()
    report.update(status='passed',seconds=time.monotonic()-start,torch=torch.__version__,device=torch.cuda.get_device_name(0),
        mean_checkpoint_sha256=hashlib.sha256(a.legacy_mean_checkpoint.read_bytes()).hexdigest() if a.legacy_mean_checkpoint else None,
        discarded_optimizer_updates=sum(x['discarded_update']['optimizer_updates'] for x in report['updates']))
    save();print('NVIDIA_PROBABILITY_REGRESSION_PASS',flush=True)

if __name__=='__main__':main()
