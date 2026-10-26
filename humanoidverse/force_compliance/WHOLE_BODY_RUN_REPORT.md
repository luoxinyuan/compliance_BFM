# Whole-body single-contact implementation and first-round launch

The ten-point catalog and parallel collector are implemented. One contact point is
sampled uniformly per episode; it remains attached throughout the episode. The
points cover both hands, both upper arms, chest, pelvis, both thighs and both
shins. Offsets were checked against the actual mesh vertices and rendered in two
views. Local images/coordinates are in
`artifacts/force_compliance/contact_catalog/`.

## Validation completed

- All **51 regression tests passed**, including ten-point physical wrench
  application, one-hot observations, uniform reproducible contact sampling,
  separate-worker GAE and terminal masks, and PPO updates.
- 6-worker and 12-worker real-G1 smoke runs each completed 20 frozen-policy
  interaction preflight episodes (ten points × two directions), with no falls.
- Each smoke run completed ten paired baseline/policy evaluation episodes using
  identical absolute anchor trajectories. No learned-policy falls occurred.
- Learning rates were adjusted after the first smoke run: the 256-dimensional
  action distribution triggered early KL stopping with the original actor LR.
  Actor now uses 3e-5 and critic 3e-4. The 12-worker test performed 9 and 12 gradient
  steps in its two updates with finite losses and final policy KL about .047/.034.
- These smoke evaluations show no meaningful learned force reduction; they
  validate the implementation and initial stability, not convergence.

## Measured training throughput

| Sampling workers | GPUs | High-level decisions / second | Approx. time for 1M decisions, excluding periodic evaluation |
|---|---|---:|---:|
| 6 (2 per GPU) | 2, 3, 4 | 33.2 | 8.4 hours |
| 12 (4 per GPU) | 2, 3, 4 | 41.8 | 6.6 hours |

Both figures include PPO updates but exclude initialization and preflight. They
come from short runs with 64-decision rollouts per worker; production uses 128.
Hardware is three RTX A6000 48GB cards. The 12-worker test used about 14.5 GB per
GPU. This is synchronous multiprocessing with GPU BFM inference and CPU MuJoCo,
not vectorized GPU physics. The small shared PPO learner runs on CPU.

## First-round process

Started **2026-10-26 17:00 America/New_York**, parent PID **3541125**, in its own
session so it can continue after this chat. Inspect the live files below to
confirm current process state; this report records launch, not completion.

Output directory:
`artifacts/force_compliance/whole_body_round1_20261026/`

- GPUs: physical `nvidia-smi` indices **2, 3, 4**.
- Workers: **12**, four per GPU, one shared trainable high-level policy.
- Budget: **1,000,000 high-level decisions**, approximately 5M low-level controls.
- Rollout: 128 decisions per worker, 1,536 transitions per PPO update.
- Limit: **12 wall-clock training hours**, checked between updates.
- Save: every 5 updates (~7,680 decisions); write atomically to `policy.pt`.
- Evaluate: all ten points against the frozen standing baseline every 50 updates,
  with a held-out seed and matched absolute anchor trajectories.
- Planning estimate: **7–9 hours** for this sample budget, subject to sustained
  throughput and shared-machine load. This is not a time-to-convergence estimate.

`train.log` has concise iteration summaries. `training.jsonl` contains per-contact
force RMS, completed episodes, failures, returns, throughput and ETA. `status.json`
is atomically refreshed each update; `final_status.json` is written at exit.
`launch.json` and `workers.json` record PIDs and GPU assignment. The source snapshot,
source hashes, test log and measured benchmark are saved with the run. Only the
high-level policy/critic are trained; the BFM checkpoint stays frozen.

Monitor:

```bash
tail -f artifacts/force_compliance/whole_body_round1_20261026/train.log
cat artifacts/force_compliance/whole_body_round1_20261026/status.json
nvidia-smi
```

See [PARALLEL_TRAINING.md](PARALLEL_TRAINING.md) for launch/resume and artifact
semantics. SIGTERM to the parent requests a checkpoint and cleanup of its workers;
for this launch, `kill -TERM 3541125` requests that stop. Do not assume the recorded
PID remains valid after the run exits.
