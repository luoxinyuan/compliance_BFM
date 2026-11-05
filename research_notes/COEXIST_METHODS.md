## Methods and evidence scope

Issue46 follows Issue45. The scientific target is one contact-conditioned Teacher that retains useful local reaching and long-distance locomotion, not task-specific checkpoint selection. This stage trained no Student. No BFM, PD, contact mechanics, architecture, history length or reward coefficient changed. Main physical cap10N;5N remains a separate regression. K80/D8 and original force feature scale /5 remain unchanged.

Two independent seeds4601/4602 started with the same actor and input normalization from I (`72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`). They use independently seeded fresh critics, fresh Adam and fixed raw-coordinate std=.008, one Gaussian sample per decision, original raw→norm16 Direct command,10Hz high level/50Hz frozen BFM/200Hz MuJoCo. Runtime `f6ca5c5f6c4de205b3457bef4a024345e2105009` was frozen in two separate worktrees. Raw logprob/GAE and transactional parameter+Adam joint-KL guard were reused. Reliable startup-only same-state raw-function retention coefficient=.002; it is not a whole-body motion freeze or a guarantee of physical preservation.

Privileged actor input includes current anchor error, current relative partner velocity, true contact force/history, contact-active status and current heading error. It does not include final distance, future path, task ID or a changing episode-clock feature. These are privileged Teachers, not deployable force-only policies. Natural arm-ready initialization is a real native-goal prefix before takeover, with physical/action/z/contact histories retained. It is not an online force-to-motion controller.

The mixture contains near/mid/far requests, matched-prefix stop/continue, natural arm-ready, zero demand, no contact, left/right waist rotation and old stop regressions. Actual per-family and per-phase counts, independent completed episodes, physical/control/prefill steps are saved separately. Paired initial deterministic dev rollouts were identical; distinct full-checkpoint hashes reflect independent critics, not different initial actors.

## Reward and exploration audit before training

Twenty saved same-condition physical runs were independently recomputed. Per-tick force/error/relative velocity/heading/uncapped terms agree with PPO reward (<1e-5 residual); inherited quality terms are read from recorded control statistics and explicitly distinguished from independently recomputed physics. Undiscounted return is a ranking diagnostic within a condition, not comparable utility across durations.

| Model | Exposed9cm | Exposed65cm | Prefix-stop | Prefix-continue |
|---|---:|---:|---:|---:|
| I | -3.452 | -12.024 | -7.658 | -13.764 |
| C0 | -1.265 | -47.692 | -12.020 | -45.845 |
| C1 | -3.737 | -12.198 | -10.263 | -12.293 |
| F4501 | -3.519 | -13.857 | -8.687 | -12.145 |
| R4501 | -3.349 | -17.567 | -11.558 | -15.193 |

The reward prefers C0's valid local response and I's valid long response. It does not require a new economy penalty to reverse these particular rankings. This is not proof that the reward is optimal for every motion-quality tradeoff. The main experiment therefore leaves reward unchanged.

At20 real I states,256 raw-noise samples each: same-state I↔C0 legal-z angles1.83–4.90°, random-noise mean angles5.4–11.16°; five-substep fixed-observation q_PD RMS differences .079–.493° for I/C0 versus .273–.652° for noise. Some noise has a component in the local-action direction, but large orthogonal components remain. Neither this projection nor a latent angle proves a multi-step recoverable C0 behavior. Two fixed-seed stochastic I runs produced9cm4.422cm/3.673N and65cm7.166cm/5.864N. These are limited sampling diagnostics, not statistical evidence of improved mean behavior.

Thus the simple claim 'std too small to change BFM action' is not supported. Temporally coherent feedback discovery remains an open limitation. Future reward-prompted sources are recorded separately; no native reward inference, composition search or F/SF planner ran here.
