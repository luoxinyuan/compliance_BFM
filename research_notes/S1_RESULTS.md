# S1 — original BFM Steering qualification

2026-11-12. Issue61; actual runtime `f994b9b66907219edd69d8e94c77bd667d7898e9`. [Protocol](S1_PRE_REGISTERED_PROTOCOL.md), [implementation supplement](S1_PROTOCOL_APPROVED.md), [erratum](S1_ERRATUM.md).

## Conclusion and stop status

Execution complete; interface qualification **not passed**. C-S has useful causal stopping/holding response, but near/sustained regressions, loaded-slip tails and nonlinear speed response prevent promotion. BC500 remains frozen baseline. No training, search, sealed inputs or external weight downloads. S2/S3 not started.

**Budget exception:** experimental branches used33,600 physics ticks, but archival reconstruction/qualification brings known total≥72,280 versus literal45,000 total cap. The implementation's separate-overhead wording did not authorize this change. Exact all-process total is unavailable. Physical jobs stopped after detection; no additional physics used to repair outcomes. Time/output remain within2h/2GiB. This is not a fully budget-compliant milestone. All failure records retained.

## Actual interface and provenance

C-S fixed admittance reads current measured force/body history: v_next=clip((10*v+0.1*Fx)/(10+0.1*30),0,0.22). C-P reads current body-frame partner velocity. Neither reads future trajectory, stopping time or final distance. Both use frozen T:71-row/1.42s source bank, phase advances proportional to v/.214, standing/bank blend clipped at v/.214, legal norm16 z. Own phase starts0; real previous executed z and original five-substep interpolation remain intact. Bank SHA f3bb1f490aa5f90a4a78653e3f3f78af79d47ddadc92d502632deb8c5550e907 includes historical offline arm/waist corrections; no old high-level actor online in T.

C-S vs C-P changes guidance law as well as available information: not a pure information ablation. Fixed admittance is a diagnostic reference, not a learned final force-to-skill controller. [Permission audit](s1_information_audit.json). Physical/reset dependencies are explicitly quarantined in legacy_worker.py; pure Guidance/T contracts remain separate.

Seven exposed windows/six parents;48 branches including exact repeats/native controls;1,680 high-level commands and8,400 BFM steps. [59 passing matching checks](s1_checks.json) include all six archived BC500 first3s control/physics traces and repeated complete trajectories. Stop/continue observable demand branches at5.5s, first command difference5.6s for BC500/C-S/C-P: no anticipatory answer leakage observed. [Branch timing](s1_branch_timing.json).

BC500 control is loaded from unchanged F-RL-v2 candidate0 container SHA e1165d80cbce13b9248eba0ee816f4ecfb873b70056257c0d32f22c720e9c39d. This is not a new BC model or a claim its file hash equals historical BC500; exact archived output parity establishes identity here. Other entries in [model manifest](s1_model_identities.json) are reconstruction provenance, not additional comparison arms. BFM remains33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361.

## Matched first3s table

Error: active-contact3D RMS. Force: actual vector RMS. Released samples never dilute contact burden. Yaw: endpoint world-coordinate pelvis angle, not change from initial yaw. No-contact error undefined.

|State|Controller|error cm|force N|root dx m|slip max m/s|yaw deg|mean u m/s|
|---|---|---:|---:|---:|---:|---:|---:|
|P1-1|BC500|2.569|2.071|0.014|0.006|-11.644|—|
|P1-1|CS|5.345|4.375|-0.010|0.073|-6.262|0.106|
|P1-1|CP|4.864|3.948|0.013|0.008|-5.198|0.014|
|P1-13|BC500|7.373|5.594|0.393|0.318|-17.820|—|
|P1-13|CS|11.534|7.537|0.499|1.235|-18.559|0.084|
|P1-13|CP|19.264|10.000|0.173|0.794|-3.006|0.066|
|new_tail-hold-97ff10c4dcd9|BC500|17.046|9.210|0.240|0.179|-9.336|—|
|new_tail-hold-97ff10c4dcd9|CS|5.225|3.919|0.194|0.485|-21.928|0.002|
|new_tail-hold-97ff10c4dcd9|CP|5.210|3.905|0.190|0.554|-20.857|0.000|
|new_near-sustained-5f62e065b5ac|BC500|2.786|2.088|0.015|0.005|-12.116|—|
|new_near-sustained-5f62e065b5ac|CS|4.381|3.553|-0.009|0.010|-6.622|0.072|
|new_near-sustained-5f62e065b5ac|CP|4.160|3.411|0.003|0.009|-6.245|0.014|
|new_stop-sustained-5f6380cdec75|BC500|12.506|8.081|0.074|0.292|6.084|—|
|new_stop-sustained-5f6380cdec75|CS|5.191|4.271|0.125|0.302|-1.218|0.063|
|new_stop-sustained-5f6380cdec75|CP|7.567|6.185|0.021|0.012|-4.191|0.036|
|new_stop-sustained-5f6380cdec75-continue|BC500|7.619|6.277|0.342|0.396|-18.892|—|
|new_stop-sustained-5f6380cdec75-continue|CS|6.647|5.545|0.300|1.162|-7.991|0.068|
|new_stop-sustained-5f6380cdec75-continue|CP|8.747|7.138|0.137|0.907|-6.777|0.072|
|native-standing|BC500|—|0.000|-0.014|0.015|-0.793|—|
|native-standing|CS|—|0.000|0.000|0.001|-7.485|0.000|
|native-standing|CP|—|0.000|0.000|0.001|-7.485|0.000|
|native-standing|fixed:0|—|0.000|0.000|0.001|-7.485|0.000|
|native-standing|fixed:0.09|—|0.000|-0.026|0.006|-5.392|0.090|
|native-standing|fixed:0.18|—|0.000|0.858|0.359|-16.611|0.180|

## Full phases and slip interpretation

Stop hold: BC50015.849cm/9.753N, C-S3.380cm/2.529N, C-P8.018cm/6.408N. But C-S deceleration7.905cm/6.704N is worse than BC5005.271cm/5.509N. Improvement is mainly subsequent holding, not every stopping phase. Released root displacement BC500−.024m/C-S+.008m/C-P−.018m; no released anchor-following judgment.

Continue moving: BC5008.499cm/6.954N vs C-S7.382cm/6.236N, while loaded-slip peak .396→1.162m/s. C-P8.779cm/7.308N is worse and advances less. Tail-hold improves contact burden but fails slip/yaw reference criteria. Sustained P1-13 C-P remains at10N and strongly regresses. [Full phase XYZ metrics](s1_phase_metrics.json), [kinematics](s1_kinematics.json), [scores](s1_scores.json).

Peak events are loaded left ankle-roll foot contacts: P1-13 C-S1.235m/s at48.85N and C-P.794 at81.06N; continue C-S1.162 at41.86N and C-P.907 at31.18N; tail C-S.485 at36.68N and C-P.554 at25.98N. Not zero-load grazing. Exceeding/persistent tick counts: P1-13 C-S2/1, C-P1/0; continue C-S4/2, C-P2/1; tail both1/0. Persistent episodes here are brief5ms events, not sustained sliding. Original three-way slip definitions and failures remain unchanged; no normal-force threshold raised. No hard physical-boundary termination occurred. Legacy contact-success fields are not applicable to no-contact controls.

No-force fixed u=.09 produces mean root vx−.0088m/s; u=.18 produces+.286m/s; u=0 near stationary. Source can actively advance without dragging, but this coupled phase/blend mapping is not calibrated proportional speed control. Only one native-standing source tested: stronger two-demand-parent fixed-speed clause remains untested, no post-outcome expansion.

## Evidence classification and recommendation

Established: exact matching, causal common-prefix response, stopping-hold benefit, near/sustained regressions, nonlinear no-force response, loaded slip events, total-step accounting overrun.

Compatible: coupled phase-speed and blend amplitude may impair speed/support control; poor T may explain why partner velocity alone fails. Neither mechanism independently isolated.

Unproven: general Steering capability; information-only advantage; Contact Adaptation necessity; P/I-BFM physical superiority; impossibility of another original-BFM interface. Do not generalize this T failure to BFM incapacity.

Next proposal: bounded P-BFM native-ERI compatibility qualification, first without contact then matched load, preserving this baseline. Public inference-bundle metadata~166MiB, not downloaded. Only after no-contact steering qualification should load-specific adaptation be assessed. No next experiment is started or implicitly authorized.

## Cost, failures, deliverables

[Ledger](s1_cost_ledger.json): whole preparation/run/analysis envelope plus publication reserve, failures included; physical job464.168s. New output~358MB at accounting. Exact all-process physics unknown; known≥72,280, not just33,600. First reducer/renderer failed on own-runtime enum import; originals retained and corrected readers used new output directories, no dynamics rerun. A publication command also failed because generic python executable was absent; rerun with system python3, no environment change or physics.

Three actual-qpos1× videos verified by full decode and local hashes at `~/Projects/compliance-bfm-diagnostics/videos/steering-s1-20261112/`: sustained-negative3s, stop-response6s, C-S stop/continue6.8s. [Hashes](s1_video_manifest.json). Complete executed windows, negative outcomes retained; no prediction presented as execution. Raw states and failed jobs retained privately, only aggregate evidence/code published. Exact executed postprocessors are versioned under research/interaction_steering/reporting.
