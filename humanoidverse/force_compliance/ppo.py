"""Small PPO on temporally extended latent commands; original FB trainer untouched."""
import torch


def advantages(rewards, values, next_values, discounts, dones, trace_decay):
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        continuation = (~dones[t]).float()
        delta = rewards[t] + discounts[t] * continuation * next_values[t] - values[t]
        carry = delta + discounts[t] * trace_decay * continuation * carry
        result[t] = carry
    return result, result + values


def train(env, policy, optimizer, iterations=2, rollout=32, epochs=4):
    observation = env.observation()
    logs = []
    for iteration in range(iterations):
        batch = {k: [] for k in ('obs', 'action', 'logp', 'value', 'next_value', 'reward', 'discount', 'done')}
        failures = 0
        for _ in range(rollout):
            with torch.no_grad():
                distribution = policy.distribution(observation)
                raw = distribution.sample()
                value = policy.value(observation)
                logp = distribution.log_prob(raw).sum(-1)
            successor, reward, done, info = env.step(raw)
            with torch.no_grad():
                next_value = policy.value(successor)
            entries = dict(obs=observation, action=raw, logp=logp, value=value, next_value=next_value,
                           reward=torch.tensor([reward], device=observation.device),
                           discount=torch.tensor([info['discount']], device=observation.device),
                           done=torch.tensor([done], device=observation.device))
            for key, entry in entries.items():
                batch[key].append(entry.detach())
            failures += int(info['failed'])
            observation = env.reset() if done else successor
        b = {key: torch.cat(value) for key, value in batch.items()}
        advantage, returns = advantages(b['reward'], b['value'], b['next_value'], b['discount'], b['done'], env.cfg.gae_lambda)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        for _ in range(epochs):
            for indices in torch.randperm(rollout, device=observation.device).split(64):
                dist = policy.distribution(b['obs'][indices])
                ratio = (dist.log_prob(b['action'][indices]).sum(-1) - b['logp'][indices]).exp()
                actor_loss = -torch.minimum(ratio * advantage[indices], ratio.clamp(.8, 1.2) * advantage[indices]).mean()
                critic_loss = .5 * (policy.value(b['obs'][indices]) - returns[indices]).square().mean()
                loss = actor_loss + critic_loss - .001 * dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite PPO loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                optimizer.step()
        log = dict(iteration=iteration, loss=float(loss.detach()), reward=float(b['reward'].mean()), failures=failures)
        logs.append(log)
        print(log, flush=True)
    return logs
