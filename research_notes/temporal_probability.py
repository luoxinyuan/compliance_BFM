"""Bounded NVIDIA FP32 likelihood, history-gradient and covariance checks."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
start = time.monotonic()
import numpy as np
import torch
from temporal_policy import Expert, rho_for
from clean_policy import update

torch.set_num_threads(1)
torch.manual_seed(4900)
device = 'cuda'
old = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/native-contact-20261106')
checkpoint = old/'native-initial.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'd399174cc2ddbd3efb858953653c9134495070948341381565f5f918f58da5cd'
ck = torch.load(checkpoint, map_location=device, weights_only=True)
directions = torch.from_numpy(np.load(old/'native-directions.npy')).to(device)
data = np.load(old/'ppo-guided/worker-0/rollout-0000.npz')
x = torch.from_numpy(data['obs'][:128]).to(device)
previous = torch.roll(x, 1, 0)
previous_u = torch.from_numpy(data['action'][:128]).to(device).roll(1, 0)
valid = torch.ones((len(x), 1), device=device)
valid[0] = 0
valid[1:][torch.from_numpy(data['done'][:127]).to(device)] = 0
records = []
# Exact two-step joint Gaussian equals product of recorded conditional densities.
rho = rho_for('continuous')
mu = torch.randn((256,2), device=device, dtype=torch.float64, requires_grad=True)
action = torch.randn_like(mu)
C = torch.tensor([[1.,rho],[rho,1.]],device=device,dtype=torch.float64)
joint = torch.distributions.MultivariateNormal(mu,covariance_matrix=C).log_prob(action).sum()
conditional = (torch.distributions.Normal(mu[:,0],1.).log_prob(action[:,0]) + torch.distributions.Normal(mu[:,1]+rho*(action[:,0]-mu[:,0]),(1-rho*rho)**.5).log_prob(action[:,1])).sum()
assert float((joint-conditional).abs())<1e-8
g1=torch.autograd.grad(joint,mu,retain_graph=True)[0]
g2=torch.autograd.grad(conditional,mu)[0]
assert float((g1-g2).abs().max())<1e-9
joint_check=dict(log_density_error=float((joint-conditional).abs()),gradient_max_error=float((g1-g2).abs().max()),scope='mathematical conditional likelihood identity, not physical success')
for arm in ['independent', 'continuous']:
    net = Expert(x.shape[-1], rho=rho_for(arm)).to(device)
    with torch.no_grad():
        net.initialize_from_raw(ck['state_dict'])
        net.set_covariance(directions)
    packed = torch.cat([x, previous, previous_u, valid], -1)
    with torch.no_grad():
        d = net.distribution(packed)
        target = d.mean+d.stddev*torch.randn_like(d.mean)
        lp = d.log_prob(target).sum(-1)
        row_lp = torch.cat([net.distribution(packed[i:i+1]).log_prob(target[i:i+1]).sum(-1)
                            for i in range(len(x))])
        error = float((lp-row_lp).abs().max())
        assert error < 3e-4, (arm, error)
        mean = net.distribution(x).mean
        prev_mean = net.distribution(previous).mean
        r = rho_for(arm)*valid
        manual = mean+r*(previous_u-prev_mean)
        assert torch.equal(manual, d.mean)
        assert float((d.variance-(1-r.square())).abs().max()) < 3e-7
    # The previous-policy mean MUST contribute to gradient under a new theta.
    one = net.pack(x[:1], x[:1], previous_u[:1], True)
    net.zero_grad()
    net.distribution(one).mean.sum().backward()
    expected = (1-rho_for(arm))/.008
    grad_error = float((net.actor[-1].bias.grad-expected).abs().max())
    assert grad_error < 3e-4, (arm, expected, grad_error)
    # Actual PPO code on discarded clones, without physical interaction.
    clone = copy.deepcopy(net)
    opt = torch.optim.Adam([dict(params=clone.actor.parameters(), lr=1e-5),
                            dict(params=clone.critic.parameters(), lr=3e-4)])
    with torch.no_grad():
        values = clone.value(packed)
        advantages = torch.randn(len(x), device=device)
    batch = dict(obs=packed, action=target.detach(), logp=lp.detach(),
                 advantages=advantages, returns=values+advantages,
                 value=values, raw_adv_std=advantages.std())
    result = update(clone, opt, batch, epochs=1, minibatch=128)
    assert result['pre_logprob_max_error'] < 3e-4
    assert result['epochs'][-1]['joint_gaussian_KL'] <= .02001
    # Make a real parameter perturbation and verify the conditional formula again.
    with torch.no_grad():
        clone.actor[-1].bias.add_(torch.randn_like(clone.actor[-1].bias)*1e-6)
        new = clone.distribution(packed)
        new_mu = clone.distribution(x).mean+r*(previous_u-clone.distribution(previous).mean)
        assert torch.equal(new.mean, new_mu)
    records.append(dict(arm=arm, rho=rho_for(arm),
                        batch_one_vs_batched_joint_logprob_max=error,
                        head_bias_gradient_expected=expected,
                        head_bias_gradient_max_error=grad_error,
                        discarded_clone_update=result))

# Stationary process check: same per-step marginal energy, distinct correlations.
stationary = []
for arm in ['independent', 'continuous']:
    rho = rho_for(arm)
    generator = torch.Generator(device=device).manual_seed(4900)
    e = torch.randn((128, 256), device=device, generator=generator)
    sums = torch.zeros(256, device=device)
    squares = torch.zeros(256, device=device)
    cross = torch.tensor(0., device=device)
    prev_energy = torch.tensor(0., device=device)
    count = 0
    for t in range(512):
        nxt = rho*e+(1-rho*rho)**.5*torch.randn(e.shape, device=device, generator=generator)
        sums += nxt.sum(0)
        squares += nxt.square().sum(0)
        cross += (e*nxt).sum()
        prev_energy += e.square().sum()
        e = nxt
        count += len(e)
    variance = squares/count-(sums/count).square()
    corr = float(cross/prev_energy)
    assert abs(float(variance.mean())-1) < .025
    assert abs(corr-rho) < .015
    stationary.append(dict(arm=arm, mean_variance=float(variance.mean()),
                           min_variance=float(variance.min()), max_variance=float(variance.max()),
                           lag1_correlation=corr,
                           scope='stationary exploration process, not covariance conditional on endogenous robot state'))

result = dict(status='passed', device=torch.cuda.get_device_name(), torch=torch.__version__,
              actor_dtype='float32', joint_density_gradient_check=joint_check, checks=records, stationary_noise_check=stationary,
              original_logprob_threshold=3e-4, original_KL_threshold=.02,
              RL_decisions=0, retained_optimizer_updates=0, seconds=time.monotonic()-start)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
