# M11 continuation throughput change (2026-11-02)

User authorized engineering acceleration and checkpoint continuation within the existing resource envelope. This does not change the reward, contact physics/cap, BFM/PD, network, batch size, PPO update or A/B std schedules.

## Measured bottleneck

GPU profile: 64 high-level decisions, 14.676 s with cProfile. OmegaConf lookup consumed 6.56 s inclusive, interpolation resolution 5.69 s, parsing 4.16 s. These are overlapping costs, not additive. PPO update in actual training is about .04 s versus roughly 65 s sampling per 768-decision batch. GPU memory occupancy is not the optimization target.

`m11_runtime_speed.install` materializes available static configuration aliases once at environment initialization, without changing the shared environment source. Unused unavailable custom resolvers remain unresolved. Full real GPU/MuJoCo paired trajectories test qpos/qvel, actual force, anchor, support, BFM action, q_PD, executed latent and total reward before adoption. Full failed probes are retained; the initial full resolve encountered an unused `len` resolver, and the first comparison encountered a boolean-subtraction bug.

Evidence: project `bridge-v2-owned/runs/m11-throughput-20261102/`. Timings under shared host contention are indicative; the resumed sampler's actual throughput must also be checked.

## Resume and A/B interpretation

A is interrupted only while waiting for rollout, after a completed optimizer batch; `interrupted.pt` retains policy, optimizer, RNG and cumulative counters. All old logs/checkpoints stay intact. Resume copies historical selection/evaluation and logs rather than appending through hard links. Original best identity is preserved, and 160 is the total target, not 160 additional batches.

Old workers did not checkpoint their full live physics state. In-flight samples are discarded and new workers start fresh episodes. B restarts its worker processes at the same completed-batch count; this boundary is recorded before B runs. This is a documented protocol modification, not a claim of bitwise uninterrupted training. Schedule changes still occur only before sampling, and old log probabilities use the actual sampling distribution.

The existing cumulative 28,800 s ledger and per-arm allowance include old attempts. No new GPU or broader CPU/memory permission is used. Shared BFM/PD, original weights, prior runtime and other users' jobs remain unchanged.

## Verified adoption

Four complete scenarios (straight/left/local/no-contact): 144.31065 s legacy vs 120.08489 s optimized, 1.2017x aggregate. All recorded physical/control arrays and total reward have zero difference. Ten focused tests passed (7 existing contract tests, 3 configuration/resume tests); pytest is unavailable, so the existing test functions were invoked directly and unittest ran the new tests. No package installation. Runtime `c98abd7912e173f1228afaca996ae43ed6009ac1`. A resumed from batch114; at 2026-11-02 13:32 CST, batch115, sampling 54.484 s /768 decisions. This is engineering throughput evidence, not improved Expert quality.
