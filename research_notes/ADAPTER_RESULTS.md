# Issue45 — Frozen-I conditional adapter versus full-actor training

2026-11-05 CST. Completed bounded stage; no PPO or subsequent training queued. [Issue45](https://github.com/GaoE05/robot-research-bridge/issues/45).

## Scientific conclusion

Freezing I's parameters did **not** solve closed-loop coexistence of the best local reach and long-distance walking. Both full-actor F and residual R retain useful walking, but neither consistently retains C0's local advantage. R is sometimes better at 30cm/zero demand, not consistently better across two seeds. This is a negative result for the proposed parameterization as a sufficient remedy, not evidence that residual policies cannot work.

Strict common-prefix tests establish feedback-dependent command changes after the partner demand diverges. They do not establish optimal arm-to-step allocation, and privileged F/R cannot be called force-only. A separate permission-limited ideal-force Student has a viable narrow closed loop, with a clear new65cm transfer gap.

## Fixed design and identities

Main training/physics runtime: `2673837a1f92cdc9ff0861c7ac1be767a930374a`; supplementary walking-source/render/analysis: `3f7a1e6be9d5f40326e01f0468cd3b8f2170a645`; initial audit: `07b85abbd90b5a536e7fdf20a1a4618d65b1e75c`. Final documentation commit is linked by Issue45; runtime is not inferred from that commit.

- BFM SHA256 `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- I `72877017c56fb9145e9fe6a016a06cd093e3ea5ba56302080132dffd57f8b32a`.
- F4501 `52956080217e6346b1a8b196ad411cbc84a07e248a8e747eeedf2dcc775e12f6`.
- F4502 `9e009414dcac884f8bfc39566e5c8ea6a9ca3831f47f5e56e0a29ea0421f3462`.
- R4501 `298af356d386f79377ca103b4fc82baecbf6e18a997a958633dade2cc7733a37`.
- R4502 `d940c66409c1cec7f6e1ffec7c25589c99abf227a597c3bdb67447d4ec9d0561`.
- Independent ideal-force Student `02f3c7685c41a3114814a537ae6a1bf861bdc686a65b89561bcda3cfe952fbae`.

F copies I; R uses `original_direct_command(frozen_I(obs)+adapter(obs))`. Only the final adapter layer is zero initialized. Initial deterministic R equals I exactly; last-layer gradient is nonzero. I actor/normalization and BFM remain unchanged. Both components read the actual combined executed history. No independent base trajectory, extra noise draw, or residual-radius cap. Training here is deterministic supervision, not RL.

Both have 188,928 trainable parameters; F has 156,289 frozen parameters, R 345,217. R adds a frozen forward, so equal updates are not equal compute. Paired seeds 4501/4502 use identical batch orders, labels, Adam 1e-5, 2000 updates and final checkpoint selection fixed in advance. Save 0/500/1000/1500/2000; no final-test checkpoint reselection.

750 slots: zero141, arm5/15 each93, walk/left/right each141. Replace 71 walk slots with 30 valid recovery labels from **three** B→I starts, not 30 independent failures. Reliable retention has 66 records. Same group-balanced batch128 and retention16. Loss is legal-z MSE + .05 raw-norm preservation + 2.473907620604606 times five-substep q_PD MSE. Recorded BFM observations and actual previous z are used; gradient flows through frozen BFM. This is fixed-state action replay, not differentiable physics rollout.

## Label and execution audit

All 750 inputs audited: six exact duplicate groups, identical z targets. No exact contradiction found; this does not prove sufficient observability. Nearby z differences are not automatically action contradictions. Local source uses current displacement mapping, not final-distance ID; precontact labels use I. Native source uses initial geometry origin/release bookkeeping not all explicitly in actor input; full history identifiability is not proved. Normalization clipping and same-state action audits are saved, not silently changed.

R frozen-base equality passes after training. Mean executed raw corrections are .131–.149; maximum .748–.803. Mean legal-z change from I is 8.20–9.28°, maximum 58.75–59.31° on learned-policy ticks, excluding native-source overrides. R4501 maximum occurs at armready22 t=10.0s. Thus failure cannot simply be attributed to an imposed tiny adapter radius. Large available correction also does not prove it is used effectively.

## Same-checkpoint physics

Values below: full active-contact 3D error cm / actual force RMS N. 10N physical contract; explicit 5N regression remains separate. Releases are excluded from active tracking error. All four final models are fixed at update2000.

| New/fixed condition | I | C0 | C1 | F4501 | F4502 | R4501 | R4502 |
|---|---|---|---|---|---|---|---|
| new9cm |4.644/3.829|2.784/2.291|4.821/3.995|4.671/3.873|4.820/3.992|4.566/3.778|4.780/3.964|
| new22cm |6.303/5.179|7.226/5.893|6.472/5.324|6.333/5.206|6.396/5.265|6.494/5.334|5.976/4.905|
| new65cm |5.495/4.534|32.024/8.605|5.581/4.552|6.012/4.894|6.292/5.083|6.657/5.446|5.135/4.235|
| common-prefix stop |4.280/3.516|5.374/4.335|4.820/3.928|4.479/3.647|4.462/3.624|5.283/4.312|5.624/4.588|
| common-prefix continue |5.719/4.710|25.178/8.387|5.440/4.459|5.452/4.457|5.726/4.671|6.035/4.931|5.879/4.823|
| armready22 |7.170/5.878|7.853/6.405|7.765/6.331|7.754/6.326|7.748/6.320|7.486/6.129|7.136/5.922|
| I prefix to6.5s,65cm |5.495/4.534|7.032/5.737|6.351/5.209|6.122/5.034|6.092/5.011|6.900/5.467|6.550/5.373|
| 5N regression8cm |3.944/3.279|2.497/2.037|5.245/3.763|4.915/3.697|5.213/3.801|6.322/3.906|6.223/3.624|

Development 80cm F1/F2/R1/R2 errors=6.684/6.712/6.610/7.065cm; 5cm=2.887/3.285/4.267/5.581cm; 30cm=6.292/6.250/4.762/5.011cm. Early/late stopping, zero/no-contact, 8/20/55cm and bilateral waist regression are preserved in `adapter_artifacts/physical-metrics.json`; no direction-specific checkpoint assembly.

In new9cm, C0 wrist-relative-shoulder forward change8.07cm, root1.58cm, left-foot path.38cm. F1:1.96/7.82/22.07cm; R1:1.79/9.50/29.33cm. This is a comparative loss of useful local behavior, not a universal prohibition on stepping for small requests. Neither low supervised loss nor unchanged frozen parameters establishes functional retention.

## Common prefix and natural source states

At branch7.05s, stop/continue share identical physical prefix, observations and commands for every tested model. First raw difference7.1s, the next high-level decision. R1 postbranch wrist travel stop/continue14.99/60.25cm and root29.26/72.91cm; F1 wrist11.65/56.76, root25.19/61.07cm. Demand response is causal in time, but F/R also see current privileged geometry; this is not isolated force causality or anticipatory distance classification.

Arm-ready is generated by actual native execution4–6s, contact6.85s, versus neutral contact4.85s. Same22cm demand: R1 relative wrist forward change−3.55→+5.47cm and left/right foot path36.23/36.99→22.56/1.40cm, but error6.494→7.486cm. State-dependent organization is present; optimal reachability-sensitive allocation is unproved.

The pre-fixed source6.5 condition was initially described as moving. Saved I velocity shows root vx=−.0402m/s: step preparation, not established forward walking. Original test is retained. One explicit **development supplement** hands over at8s, actual vx=.1606m/s; all three branches share exact800-tick prefix. I/F1/R1 errors5.495/5.494/6.646cm and force4.534/4.525/5.396N. No retraining or selection followed it; it is not a new blind test.

## Restricted-observation probe

Teacher I was fixed before outcomes, not chosen from F/R. Fresh1639D Student: BFM proprio/history465, actual previous z256, force history900, known contact mounting18. BFM features are joint q/dq, projected gravity, angular velocity, prior executed actions and their history. No anchor/error, partner velocity/yaw goal, global clock, future path, active truth, teacher features, privileged hidden state or task ID.

Seven whole training episodes/883 aligned decision labels; balanced episode sampling, seed4511,2000 updates. Split by full episode/path/source, not adjacent frames. Offline z angle.900°, endpoint q_PD RMS.136°. All teacher labels use the correct decision-time state. No unnecessary projection optimization of already legal z.

| Closed loop | Teacher I cm/N | Student cm/N |
|---|---|---|
| new9 |4.644/3.829|4.232/3.471|
| new65 |5.495/4.534|7.404/5.943|
| prefix stop |4.280/3.516|4.639/3.792|
| prefix continue |5.719/4.710|5.600/4.611|
| armready22 |7.170/5.878|6.798/5.522|
| oldwalk |6.585/5.207|6.734/5.519|

Student shared-prefix parity also passes, first change7.1s. This is limited positive deployable-information evidence and a real long-distance transfer failure, not generalization proof. Ground-truth simulated external force remains **ideal-force observation**; actual G1 sensing availability is unknown. No DAgger or fabricated Teacher history used here.

## Quality, cost, failures and reproducibility

132 full physical episodes:52development+56final+21probe+3walking-source. Zero safety terminations/support-loss ticks; max root tilt12.501°. Maximum20ms q_PD change60.287° at R4501 new65 t10.16s joint index10; maximum actual5ms joint change3.332°. These are different quantities, not a claim joints rotated60° in20ms. Complete XYZ, support/slip, phase, effort proxies and recovery are in metrics. No new smoothing or reward/cap changes.

Four Expert runs×2000 +Student2000=10000 supervised updates; RL0. Ledger job-wall3735.67s including conservative supervisor-failure charge, plus saved-data CPU aggregation89.77s and short audits/checks; roughly2.03GiB before final text artifacts, below16000s/10GiB. GPUs2/5/6 at most3; actual usage refreshed before launch, no other users' tasks touched.

R4501 supervisor encountered a transient partially-written ledger JSON. Child completed all updates/evaluations; process confirmed absent. OS exit status unavailable (`null`), not fabricated as zero. Preserved logs and `reconciled_completed`; charged677.53s upper bound, measured runner293.91s. Atomic ledger writes/read guards corrected monitoring only. No retraining, changed scientific runtime or duplicated jobs.

Run root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/conditional-adapter-20261105`. Models under F4501/F4502/R4501/R4502 `<name>-2000.pt`; Student `probe/ideal-force-student.pt`. Source/config: `adapter_policy.py`, `adapter_fit.py`, `adapter_protocol.json`, `adapter_evaluate.py`, `adapter_probe.py`, `adapter_source.py`. Independent fixed trees and complete raw trajectories remain on server. Public artifacts contain aggregates, no weights/raw trajectories/credentials.

Mac: `~/Projects/compliance-bfm-diagnostics/videos/conditional-adapter/`:01 same-model small/long;02 R1 shared-prefix stop/continue;03 R1 natural source states. Three real-qpos25fps1×MP4, full episodes, SHA and complete decode checked. Unequal-duration panels explicitly label held final frames. INDEX gives exact conditions, model identities and failures.

## One next decision (not executed)

Do not promote R, repeat adapter tuning, or demand perfect offline imitation before feedback learning. Next proposed experiment: a bounded **real-physics feedback optimization** with matched shared-prefix stop/continue and near/long requests, starting from one explicitly fixed retained-walking model, retaining reliable walking actions only where validated. Test whether physical reward can preserve the demonstrated local solution while resolving transition-state deviations, with a frozen supervised baseline and unchanged reward initially. First check that existing reward ranks C0's effective small-request response above unnecessary whole-body travel; if not, isolate an independently specified movement-economy reward comparison rather than hiding it in architecture changes. No automatic launch; new authorization and resource budget required.
