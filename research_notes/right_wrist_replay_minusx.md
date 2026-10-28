# Issue #19 — full no-force action replay and mirrored -X control

## Conclusion

The full 6 s no-Tether replay **exactly reproduces** the no-force baseline's recorded physics, observations, actions and time grids. The subsequently gated -X closed-loop condition completes 6 s without triggering the original failure criteria. No replay implementation correction or numerical retry was needed. The previous +X result and tether-replay failure remain unchanged.

This closes the specific missing replay control: removing online BFM inference does not itself change this unperturbed trajectory. Combined with the previously recorded +X replay failure at 3.70 s, the evidence supports a contribution from online BFM state feedback to stability under that tested coupled disturbance. It does not establish force conditioning, intent inference, a contact expert, or LPP efficacy.

## Fixed protocol and provenance

- Original frozen `FBcprAuxModel`, original `ComplianceEnv`, `run.evaluate(..., baseline=True)`, Tether, observation/action interface and low-level PD. CPU only, seed 0, no learning or simulator rendering.
- K=80 N/m, D=8 N s/m, cap=5 N, activation=1 s, smoothstep ramp=2 s, release=4.5 s, episode=6 s. Failure remains root height <0.4 m or upright cosine <0.4, plus original termination/nonfinite checks.
- Common anchor origin is the archived no-force wrist world position at t=1 s: `[0.13686332656697187,-0.1968667867765935,0.848818048880822]` m. Offset changes only from `[+.03,0,0]` to `[-.03,0,0]` for the new contact condition.
- Original checkpoint SHA256: `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- New runtime SHA256: `a55c8a810b74577693fb0884bde78cc72620cb2ad1cf18b7aae25d746ac1b45c`.
- Archived +X JSON SHA256: `e03f6b23bb426c884a79a33c232e5c45aa0b06036590a2d416468577496fbd36`. It was reused, not rerun. Fresh no-force states/actions/observations/PD reproduce the earlier pinned baseline exactly, so both signs share the same reference.

## Full replay diagnostic

Fresh no-force BFM and no-force replay each produce 300 actions at 50 Hz and 1200 physics samples at 200 Hz. All actual forces are zero and Tether is inactive. Replay uses source indices 0–299 exactly once, with zero live BFM actor calls, through the same original wrapper and live PD.

| Check | Result |
|---|---|
| Initial state, standing latent and actor observation | Exact equality |
| Action values and action time grid | Exact equality; max delta 0 |
| qpos / qvel / PD torque | Max delta 0 / 0 / 0 |
| Complete recorded physics rows, including contact/force/root/wrist | Exact equality for all 1200 rows |
| Per-action observations and latent | Exact equality for all 300 rows |
| Original control rows and episode metrics | Exact equality |
| Full horizon / original failure | 6.00 s / no failure |

The runtime gate uses exact recorded-value equality, not an eased tolerance. The 1e-12 tolerance used below applies only to cross-Python recomputation of aggregate metrics, not to this replay gate. A failed replay gate would be classified as an implementation/runtime diagnostic and prevent the -X trial.

## Stability and actual forces

| Metric | No force / no-force replay | +X archived | -X new |
|---|---:|---:|---:|
| Simulated seconds | 6 / 6 | 6 | 6 |
| Live BFM calls | 300 / 0 | 300 | 300 |
| Original failure / fall | No / No | No | No |
| Whole-episode force RMS N | 0 | 0.584745 | 1.661049 |
| Force peak N | 0 | 0.901718 | 2.562635 |
| Active samples at force cap | N/A | 0% | 0% |
| Minimum root height m | 0.779834 | 0.779995 | 0.779934 |
| Maximum root tilt deg | 2.524867 | 2.564620 | 2.383388 |
| Double-foot contact fraction | 99.4167% | 99.4167% | 99.4167% |
| Foot XY max displacement from reset mm | 16.8611 | 16.8970 | 17.6113 |
| Action / torque saturation fraction | 0 / 0 | 0 / 0 | 0 / 0 |

After contact onset, both contact conditions retain double-foot contact at every physics sample. The five initial no-contact samples are reset settling at .005, .010, .015, .025 and .030 s. Foot normal forces, positions, root RPY and actuator torques are retained in the full traces; contact fraction alone is not used as proof of stability.

| Signed/vector metric (XYZ) | +X | -X |
|---|---|---|
| Active-period mean force N | `[+.193183,+.577551,+.065783]` | `[-1.932063,+.801977,+.274810]` |
| Whole-episode force XYZ RMS N | `[.374216,.443128,.074348]` | `[1.522892,.627441,.214939]` |
| Force impulse N s | `[+.677106,+2.024316,+.230570]` | `[-6.771882,+2.810931,+.963209]` |
| Late hold mean force N | `[+.523428,+.624847,-.027468]` | `[-2.003883,+1.000953,+.284502]` |
| Active fraction with Fx<0 | 31.2411% | 100% |
| Median angle to nominal anchor direction, for norm>=.05 N | 50.6720 deg to +X | 23.9827 deg to -X |

Both conditions start with zero position error and zero spring force. First total force is approximately `[-.227904,+.277371,+.059103]` N, entirely damping from the same initial wrist velocity; the first forces differ only at floating-point roundoff. There are 701 active physics samples under the original floating-point release timing. The -X force audit reconstructs the original Tether to 4.44e-16 N, using the actual **negative** anchor velocity. The audit does not drive control.

These are mirrored anchor displacements, not mirrored or equal force histories. The unperturbed wrist naturally drifts toward +X; this and the coupled dynamics make -X generate larger force. Lateral/upward forces remain substantial. No force-direction claim is inferred merely from the anchor's sign.

## Incremental response, separated from baseline drift

Late hold is the previously fixed [3.5,4.5) s window. All deltas below subtract the matched no-force sample at the same time.

| Late hold mean delta, XYZ | +X | -X |
|---|---|---|
| Wrist world mm | `[+4.4605,+6.6606,+.5942]` | `[-23.7518,+2.0199,-3.2585]` |
| Wrist minus root translation mm | `[+2.6930,+4.2471,+.3460]` | `[-10.2436,-.9222,-3.6432]` |
| Wrist in each condition's rotated pelvis frame mm | `[+1.1198,+3.0671,+.7026]` | `[-7.2165,+.7472,-5.7115]` |
| Root translation mm | `[+1.7675,+2.4135,+.2482]` | `[-13.5082,+2.9421,+.3847]` |
| Root roll/pitch/yaw deg | `[-.040935,+.113733,+.386038]` | `[+.206961,-.695922,-.700461]` |
| BFM action RMS delta | .00269940 | .01329438 |
| Maximum absolute action delta | .00808989 | .06062636 |

Both contact conditions first diverge physically from no force at 1.005 s and change BFM action at 1.02 s; standing latent remains fixed. This is state-dependent feedback, not a force-conditioned latent policy.

For -X, about 13.51 mm of the 23.75 mm wrist X increment comes from root translation, leaving 10.24 mm relative to root. Also, the wrist moves only **-4.7407 mm from its t=1 s anchor origin** during late hold: without force it would have moved +19.0111 mm. Thus -23.7518 mm is an intervention increment, not 23.75 mm of absolute backward travel or complete following of a -30 mm anchor. The corresponding +X absolute movement is +23.4716 mm, but its intervention increment is only +4.4605 mm.

Late-hold 3D anchor error RMS is 10.1308 mm (+X) versus 28.3661 mm (-X). Mean XYZ error is `[+6.5284,+7.7350,-.2640]` versus `[-25.2593,+12.3757,+3.5888]` mm. Stable direction-dependent deflection/feedback is observed; accurate symmetric guidance and force relief are not established. The original settling threshold of 5 N equals the force cap and is not used as success evidence.

After release, forces return to zero. At 6 s the paired wrist X / wrist-root X / root X residuals are +2.6006 / +2.3391 / +.2615 mm for +X and -7.2718 / -5.7171 / -1.5547 mm for -X. The 1.5 s recovery window does not establish asymptotic convergence.

## Attempts, failure boundary and validation

One numerical launch, exit 0; wall time 140.523 s, peak owned process-group RSS sum 6.882553 GiB. CPU affinity `[0,1,2,3]`, computation threads=1, GPU=0, cumulative wall limit=3600 s and RSS watchdog=8 GiB. This remains Trusted SSH mode, not native OS isolation. All owned processes exited; no resource boundary was hit.

**No new engineering failure or scientific failure was encountered.** The first previously observed contact-replay failure remains the original +X replay at 3.70 s, root height .396216 m < .4 m. It was neither rerun nor erased. The present full no-force replay resolves its specific missing implementation control, but no -X contact-replay ablation was performed.

- Mac and server: 9/9 meaningful unit tests, including wrong action timing, repeated source index, truncated prefix, late qpos/qvel/PD mismatch, accidental live actor, nonzero force and negative anchor-velocity sign.
- Independent saved-data verification on Mac/server: pass; aggregate recomputation maximum discrepancy 1.99e-13 with tolerance 1e-12. No model imported by these tests.
- 412 prior-file/asset hash checks pass; shared checkout and all prior personal tracked files/results preserved; checkpoint/data SHA rechecked. No shared environment, weights or original evaluation code changed.

## Core additive changes and reproducibility

Only new `research_notes/right_wrist_replay_minusx*`, test, verifier and plot files are added to personal `gaoe05/lpp-issue-19`. Old scripts/results remain untouched. The new runtime reuses the validated launcher/recording code and adds: (1) full exact no-force replay gate with first-difference diagnostics; (2) conditional -X rollout; (3) direction-aware force audit and archived +X comparison. It does not fix or replace the original controller.

Personal worktree:
`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-19-a8187804319761a4/repo`

Run and logs:
`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-wrist-replay-minusx-20261028T070932Z`

`attempt-01/outputs/` holds `no_force.json`, `no_force_replay.json`, `minus_x.json`, pinned `plus_x_reference.json`, `result.json`, `replay-gate.json`, standing latent, `replay_minusx.png/.svg`. The attempt holds stdout/stderr, per-process telemetry and copied launch scripts. Mac evidence mirrors these under `.bridge/issue19-replay-minusx/` in the existing v2 project.

Data-only validation:
```sh
python3 -B research_notes/test_right_wrist_replay_minusx.py
python3 -B research_notes/verify_right_wrist_replay_minusx.py RUN/attempt-01/outputs
python3 -B research_notes/right_wrist_replay_minusx.py --analyze RUN/attempt-01/outputs
```

Reproduction requires a new unique run ID; this command is documented, not run again:
```sh
python3 -B research_notes/right_wrist_replay_minusx_supervisor.py --run-id issue19-wrist-replay-minusx-NEWUNIQUEID
```

Only seed 0 and these fixed conditions are established. No new training, LPP/PPO, WBC, GPU experiment, push, queue replay, Bridge change or background worker. The coding session used verified `gpt-6.1-sol / medium` after the owner explicitly allowed continuation; existing Astra/Sol routing was not changed. Stop here for scientific review.
