# M11 Privileged Contact Expert Learning

Prepared formal acquisition after 1,536 measured GPU RL decisions (18 PPO optimizer updates) in a separate throughput run. Probe models are not final training models. First two parallel-loader failures (relative IPC, then CPFS socket support) are preserved; inherited Linux abstract IPC is now propagated to spawned children and verified.

Parent: M10 49639ed29d4cbf513c285633841426579d80d5bd. New team-main optional accumulated-command API was inspected but not silently merged; BFM/PD/source pins remain unchanged.

Primary method: privileged full-256D Gaussian tangent residual over frozen M10 deployed policy, sphere16 output, norm8 execution trust region. This removes M9C's fixed19D basis; it is not an unrestricted new low-level controller. Current contact error/relative velocity/force/proprio/support and own previous residual are inputs; no future partner path, task ID or teacher phase. M10 prior continues at50Hz, RL decisions10Hz, original physics200Hz.

Real sampling reuses ComplianceEnv and original wrapper; spawn/pipe versioning, per-worker GAE before flattening and raw probability coordinates follow force_compliance.parallel/ppo. Added analytic joint Gaussian KL and old-GAE versus updated-value EV. Reward terms and dev/final conditions are frozen in m11_protocol.json; failed episodes include absorbing tail cost. No root-speed target forces small requests to walk.

Formal target: 122,880 decisions, 160 batches of3x256, approximately2.46million training physics steps plus initialization/recovery-prefix/evaluation steps separately counted. Three workers share one actually free GPU;8CPU,32GiB owned RSS,8GiB owned VRAM,8GiB outputs,14400s cumulative supervised wall budget. Measured old-feature throughput9.24–12.19decisions/s; vectorized physical-history features are tested for float32 equivalence. These are execution budgets, not convergence/success thresholds.

Selection: rank initial and periodic dev checkpoints by the fixed rule in protocol, retain each best version. Final combinations are excluded from training and dev selection. Training distribution spans local and locomotion requests, speed/curvature/yaw, no-contact and actual M10-visited prefix states. No arbitrary joint reset.

Issue32, personal branch gaoe05/m11-privileged-expert. Runs: bridge-v2-owned/runs/m11-privileged-expert-20261101. Actual checkpoints/data/logs stay outside Git. No M11 performance claim yet; no Student training planned before meaningful expert improvement.
