# M11 — Privileged Contact Expert Learning: bounded training and failure audit

2026-11-02. Scientific objective remains learning effective privileged feedback over frozen BFM, then transferring through LPP. **This run achieved development improvement, but did not establish a reliable expert superior to M10. Do not promote it to the deployed model.** No new Student was trained. M8/M9B/M10 and all failures are retained.

## Actual implementation and provenance

Parent M10 `49639ed29d4cbf513c285633841426579d80d5bd`; personal branch `gaoe05/m11-privileged-expert`. Training runtime `b995f37bdcb01793adcaebd7b70a76c884210bfa`; final runtime `c8ec3f72cba26e7c252f317aea5799ab69792f76` (evaluation bookkeeping/comparator fix, same learned controller); frozen noise diagnostic `a3ce101f5ff875c434ae1604027e64aa961c0f8f`; read-only observation diagnostic `d9f1985b219207a7173354218b29b3677898ccd2`; memory diagnostic `ca1ba59afe6e221eba2b8f10e2a878a2fe72528e`.

- Frozen BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Frozen M10 **deployed Student** prior SHA256 `1742ba74e9ec871abf8577eb9a276d7a1e614cd3184330a008dd9c4a7031bcd8`. This is not the best handwritten privileged M10 expert. Using it simplified parallel queries, but inherited its quality/state-feedback limitations; that was our methodological choice, not a BFM requirement.
- Learned external actor/critic: separate 128–128 tanh MLPs; 1,002 privileged inputs (465 normalized proprio/action, 25 current error/velocity/force/support/body features, 256 prior-z/16, 256 previous own raw command). No future path, task ID or privileged reference phase. The prior itself retains physical force/proprio/action history.
- Actor outputs raw 256D diagonal Gaussian. `tanh` → tangent at current prior z → norm cap 8 → original BFM sphere norm 16. Correct raw Gaussian coordinates are retained for PPO likelihoods; transformed z is not treated as a Gaussian action.
- High-level decisions 10Hz, original prior and BFM/PD 50Hz, physics 200Hz. Frozen prior remains a state-dependent feedback policy, not a static reference. Shared environment/PD/weights untouched.
- Actual trainable parts: external actor, log-std and critic. BFM, prior, Tether, latent endpoints/reference assets were not trained. This is privileged expert acquisition, not PPO-vs-LPP research or a new Student.

## Actual training and selection

Formal training: **122,880 RL decisions**, **2,451,456 physics steps**, **612,864 control steps**, **532,440 preparation/visited-prefix physics steps separately**; 160 batches, **2,289 optimizer minibatch updates**, 783 completed training episodes and 63 development episodes. Incomplete episodes discarded at development resets are not counted as complete. No logged training or development termination failure. One learner seed: this is not replicated convergence evidence.

Training covers randomized local and walking distance/ramp/onset/yaw/curvature, no-contact, and actual M10-visited prefix states; 101 distinct logged onsets and continuous distance/ramp/yaw variation were checked. No arbitrary qpos reset. The final combinations lie within these marginal ranges but were excluded from development selection.

Fixed development selection: mean `force_RMS + 3*tracking_RMS_m + cap_fraction + 20*failure`, including initial policy. Best was **iteration 60 / 46,080 decisions**, not last160. Score 4.29839 initial → 4.09753 selected → 4.25361 last. Later training oscillated; no claim of convergence.

Selected `train-attempt-01/outputs/job/best.pt` SHA256 **`bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f`**. Last SHA `23f5355fa03ff190bc8422b95a714e0498d098ba302b4d0c8d34575d34667e52`. Final results were not used to select another checkpoint.

Development straight: error **12.661→7.832cm**, actual force **4.610→4.144N**, cap fraction **78.87→47.42%**, q_PD peak **44.33→41.23°/20ms**. Other development conditions were mixed; left tracking worsened 9.509→10.152cm. Budget completion is not broad success.

## Prespecified final physics: all results

Full-contact 3D tracking RMS cm / actual force RMS N. All 18 branches completed; exact initial qpos/qvel and partner anchor paths match within each condition; actual recovery prefix also matches. Release is followed for 6s, not truncated to improve metrics. No-contact/after-release geometric anchor error is not counted as active tracking.

| Final condition | Frozen M10 Student prior | M10 handwritten feedback + projected arm expert | M11 selected deterministic mean |
|---|---:|---:|---:|
| New straight: .91m, ramp8.6s, onset4.85s, yaw−8° | 8.493 / 4.139 | 14.265 / 4.388 | **37.667 / 4.588** |
| New left: .76m, ramp7.4s, onset4.65s, yaw+6°, curve+.17 | 15.352 / 4.618 | **7.586 / 3.819** | 10.869 / 4.662 |
| New right: .94m, ramp8.8s, onset4.35s, yaw−9°, curve−.19 | 9.351 / 4.403 | **6.134 / 3.337** | 7.956 / 4.502 |
| Local: .12m, ramp4.6s, onset4.75s, yaw4° | 1.602 / 1.346 | same prior; not an independent local expert | 1.669 / 1.386 |
| Recovery: .88m, ramp8.4s, onset4.55s, yaw−4°, curve+.08, M10 prefix to8.05s | 6.955 / 3.497 | 6.844 / 4.052 | 7.068 / 3.517 |
| No contact | max root drift2.698mm | same prior | 2.873mm |

Recovery whole-contact figures include the shared prefix; `physical-metrics.json` additionally supplies post-prefix metrics, avoiding dilution claims. This is only one physical seed per final condition, not robust multiseed generalization.

M11 force XYZ RMS: straight **(3.592,2.556,1.272)N**; left **(2.357,3.794,1.335)**; right **(2.008,3.547,1.912)**. Straight error XYZ **(30.702,20.274,8.074)cm**. These are real 3D burdens, not pure signed-X forces.

M11 versus prior cap fractions: straight48.99→80.13%; left82.22→82.17%; right59.38→45.95% (force RMS still increased); local0%; recovery29.77→30.59%.

Quality remains limited: straight q_PD jump **43.45→51.35°/20ms**, actual joint peak7.43°/20ms, wrist world-height drop **14.07→20.07cm**, mean torso pitch **9.44→12.61°**, minimum root height .6516m, maximum tilt17.01°. Two physics frames (10ms total) had no detected foot support; no fall/termination, but not perfectly preserved support. Left/right q_PD peaks51.22/45.85°, actual joints5.84/6.13°. No recorded action/torque saturation or joint-limit excess. Slip p95 .0159/.0221/.0240m/s for left/right/straight. Six-second recovery last-second root speeds .00158/.00359/.00286m/s respectively, with earlier recovery peaks .0455/.1130/.0650m/s; low final speed does not erase transient quality problems. Local max root displacement3.57cm, not the old unnecessary30cm walking failure. No new torso regression was run: M10 torso policy is preserved, not newly qualified for this expert.

## Code, reward and applied-force audit

1. **PPO data/likelihood:** old raw logprob reproduction max6.10e−5, ratio near1; worker-separated GAE terminal handling tested. Last20 old-GAE-value EV .8275 versus updated-value EV .8685. Analytic joint Gaussian KL .02106, clip fraction .6058 at±.15; KL stop is checked after an update and can slightly overshoot .02. These do not prove optimal learning, but do not support an old-logprob/GAE coordinate bug.
2. **Actual force kernel:** every active final physics sample reconstructed exactly from preceding qpos/qvel, current anchor/velocity and actual wrist point: `F_raw=80*(anchor-wrist)+8*(partner_velocity-wrist_velocity)`; `F=F_raw*min(1,5/||F_raw||)`. Maximum reconstruction discrepancy **0N** across15 active traces. World axes, point, units, cap match logs. This is maintained-contact Tether, not unilateral hand pressure.
3. **Force limit:** static spring alone reaches5N at6.25cm error; damping makes the dynamic threshold velocity-dependent. Robot mass35.112kg, cap≈1.45% weight. It is a small cue, not enough to drag the body into compliance; active motion is intended. Larger cap was not tested and cannot be claimed to fix behavior. Direction is retained under saturation, magnitude beyond5N is not. Correlated Y/Z demands also reduce the X share of the shared vector-norm cap.
4. **Reward is not an exponential force reward:** instantaneous main rate is `1.5*||F||²/25`, plus `.15*log1p(||F_raw||²/25)`, `.15*||error||²/.04` and velocity/stability/continuity terms. Therefore there is **no force-reward sigma** to shrink. The .08 sigma is policy exploration. A larger force weight cannot restore instantaneous magnitude distinctions once the norm is capped; time spent saturated and unsaturated behavior still affect reward.
5. **Not all reward information is lost:** failed straight raw-demand RMS30.322N versus prior7.020N; actual force4.588 versus4.139N. Active mean tracking cost .5328 versus.0272; raw-demand cost .4162 versus.1296. Failed episode return **−24.048 versus−12.914**. Reward correctly ranks this failure worse, even though the capped force cost is flat (saturated std≈3.6e−16). Weight inadequacy is possible but not established as the primary cause.
6. **Y/Z force origin:** learned left spring component RMS(4.239,7.101,2.724)N versus damper(.891,.760,.790)N; right spring(2.994,4.783,2.981), damper(.767,.706,.708). YZ burden exceeds X for71.5%/81.4% of active frames. Position/coordination mismatch dominates these traces, not simply excessive D or a reversed axis.
7. **Read-side effect disproved as primary explanation:** original `point_state()` calls `mj_forward`; per-caller audit isolates original physical force application from extra M11 observation reads. Extra reads change only warmstart at~1e−12, point/velocity exactly equal. Replacing them with cached read-only kinematics leaves development error7.832→7.829cm and failed final37.667→37.731cm. Prior final8.493→9.188cm shows contact dynamics numerical sensitivity, so runs are not bit-identical; the severe learned failure remains. Cached reads are cleaner implementation, not a successful scientific fix. Shared Tether physics unchanged.

## Frozen sampling diagnostic: stronger evidence than arbitrary weight tuning

Same selected checkpoint, no updates, fixed epsilon seeds101/202/303. The former final straight is now explicitly **exposed diagnostic**, not a new blind test.

| Condition | Deterministic mean | Half original std (3-seed mean±sample SD) | Full original std (3-seed mean±sample SD) |
|---|---:|---:|---:|
| Dev straight tracking cm | 7.832 | 9.325±1.004 | 8.969±.787 |
| Dev straight actual force N | 4.144 | 4.633±.038 | 4.488±.200 |
| Exposed new straight tracking cm | **37.667** | 26.338±17.738 | **9.329±1.167** |
| Exposed new straight actual force N | 4.588 | 4.503±.129 | 4.298±.269 |

New-straight individual half-std errors27.611/43.405/7.998cm; full-std8.718/10.674/8.594cm (seeds101/202/303). All retained, no favorable-seed selection. Full noise is not generally better than prior8.493cm/4.139N; smaller noise is not monotonically safer/better.

Same-state/observation action probes show mean tangent norm~.146 versus full-noise~1.28; at representative devt5.6s, mean produces q_PD RMS change .036° versus noise .391° (eight fixed eps); at t7.1s .066° versus .458°. Mean latent correction in formal final never exceeds .756°, far below cap8; feature clipping absent in tested observations. Exploration dominates the learned mean in executed-action space. Raw std barely changed .0800002→.0799972. This is evidence of a stochastic-training versus deterministic-deployment gap, not proof that the entire BFM cannot express a useful expert.

First sustained trajectory divergence from M10 prior: root difference>1cm at7.285s, >5cm8.065s, >10cm8.22s. At8s own residual angle .531°, but prior latent evaluated on the two *different* closed-loop states differs33.74°. State/history feedback amplifies changes around walking entry. These cross-trajectory differences do not by themselves isolate same-state BFM sensitivity or prove the prior is the unique cause.

### Executed-noise versus own-command-memory causal split

Six additional frozen real-physics branches, same three seeds, exact same initial qpos/qvel and anchor/velocity/time/connection arrays as final straight (programmatically verified). At each10Hz decision, independently control noise in the executed raw action and in the next previous-command feature. The latter can be fictitious and is **only an intervention for diagnosis**, not a deployable controller or new final success.

| Executed action | Next command-memory feature | Error cm, mean±SD | Force RMS N, mean±SD |
|---|---|---:|---:|
| mean | mean (original final) | 37.667 | 4.588 |
| sampled | sampled (original full-noise3seeds) | 9.329±1.167 | 4.298±.269 |
| sampled | mean | **9.173±1.202** | 4.395±.344 |
| mean | sampled | **37.476±.723** | 4.588±.000091 |

This **rejects memory-noise distribution as a sufficient explanation for this failure**. Executed physical perturbations are the stronger lever in this limited condition; noisy features alone do not restore locomotion. The small nonzero memory influence remains. It does not prove that random excitation is the desired final compliance mechanism, or that one specific gait phase is the unique cause.


## Implementation changes, failures and reproduction

Core files: `m11_env.py` (privileged observations/rewards/prior), `m11_policy.py` (legal residual/PPO), `m11_worker.py` / `m11_train.py` (real parallel sampling, dev selection), `m11_protocol.json`, `m11_diagnostics.py` and frozen diagnostic plans, `m11_force_audit.py`, `m11_audit_evaluation.py`, `m11_metrics.py`, `m11_render.py`.

Corrections retained: per-worker cwd and inherited Linux abstract IPC fixed two startup failures; final status now written on completion (old train status.json says running, but original result.json/ledger/exited workers establish completion); comparator memory advances causally at original50Hz; measurement reads now avoid extra dynamics solve. A diagnostic-plan parser typo caused `readcheck-attempt-01` to fail before loading any BFM/GPU, fixed in d9f1985 and rerun as attempt02. All failed attempts remain. No scientific changes to force, reward or selected weights after final testing.

Meaningful tests: raw likelihood and legal sphere/zero-residual preservation; worker/terminal-separated GAE; float32-equivalent physical history feature optimization; paired initial/prefix/anchor/finite/latent/force audit; exact physical force reconstruction; paired legacy/cached reads; real common-epsilon noise and memory diagnostics. `test_m11.py` passed. Remaining weakness: source is research code, not an exhaustively verified general robotics stack.

Use project .venv; no installation into shared environments. Read-only reproductions after setting `PYTHONPATH` to private repo:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$P/.venv/bin/python" "$ROOT/research_notes/test_m11.py"
```

The supervisor ledger stores exact commands/runtime/GPU UUID/PIDs for every run. Do not rerun a consumed stage in its original output directory. Any future scientific repetition needs a fresh output directory/protocol and live resource check, not overwriting this evidence.

All paths below relative to personal run `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-privileged-expert-20261101`:

- `train-attempt-01/outputs/job/{training.jsonl,best.pt,last.pt,dev-*/}`; `analysis/{training-analysis.json,learning-curves.png,first-divergence.json}`.
- `final-attempt-01/outputs/job/evaluation/`: all18 real qpos traces, `physical-metrics.json`, `paired-audit.json`, `force-audit.json`.
- `diagnostic-attempt-01/outputs/job/evaluation/`:12 full episodes and same-state action probes.
- `readcheck-attempt-02/outputs/job/diagnostic-summary.json`:6 full paired episodes, per-caller measurement audit.
- `memorycheck-attempt-01/outputs/job/`:6 frozen causal branches; artificial memory explicitly diagnostic only.
- `videos-attempt-01/outputs/`: real-state paired movies and source hashes. Mac: `~/Projects/compliance-bfm-diagnostics/videos/m11-privileged-expert/` contains **only3 MP4 + INDEX.md**, not raw trajectory clutter. 01=development improvement, 02=unseen straight failure, 03=left handwritten-expert gap. Normal1×,25fps, fixed oblique/overhead, actualforce/anchor/contact/heading. Local SHA, duration and full decode verified; frames inspected.

## Resources and research decision

All supervised attempts including startup failures, training, final tests, diagnostics and rendering used **13365.88s / 14,400s** cumulative wall (~3.713h), one GPU1 UUID `GPU-0cd3457d-8b3f-656a-cacf-0fd505369fe3`,8CPU, peak owned RSS22.42GiB /32GiB, max compute VRAM2194MiB /8192MiB; stage output~1.54GiB /8GiB. Rendering graphics VRAM is not measured by the compute-app query; total device free memory returned to baseline. All owned workers/supervisors exited; other workloads were not managed. Roughly17minutes of the predeclared wall budget remained, insufficient for another comparable full acquisition; **the budget was not exhausted**, and a short remaining-budget smoke run would not establish convergence.

**First scientific boundary:** deterministic deployment of the learned privileged mean is not robust across a new in-range joint condition, despite development improvements, correct force reconstruction and usable old prior. This is an expert-learning/deployment-quality boundary; no new Student transfer was justified. Current evidence does not establish BFM incapacity, force simply being too small, wrong force-reward sigma, or a single implementation bug as the cause.

**One recommended next acquisition:** preserve physical K/D/cap, reward, prior, final evidence and architecture; compare the existing fixed-.08 exploration recipe against a predeclared gradual exploration-reduction curriculum with matched seeds/sample budget and both deterministic and sampled development evaluation. Require improving actual mean-policy force/tracking/quality, not just stochastic return. Keep the exposed straight condition development-only and freeze a new combined final set. Do not simply switch deployment to a lucky noisy seed, and do not simultaneously raise force/reward weights. A fresh finite training allocation should be recorded before launch; no additional acquisition is running. Stronger feedback-expert warm start remains a supported alternative if this targeted consistency intervention fails, rather than endlessly increasing Student data or patching reference motion.

This is a completed bounded **M11 training-and-diagnosis report**, not closure of the larger scientific objective of obtaining a broadly superior learned contact expert.

