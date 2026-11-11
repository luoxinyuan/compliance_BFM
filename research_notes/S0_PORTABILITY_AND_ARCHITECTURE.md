# Issue61 S0 — Portability and clean architecture

2026-11-12. **S0 COMPLETE; S1 physical execution NOT STARTED.** F1 closed first at `a7cba60`; no new physics, training, external model weights or replay downloads in S0. Scope is source/model metadata audit, original frozen-model replay and clean interfaces. [Issue61 contract](https://github.com/GaoE05/robot-research-bridge/issues/61); [F1 final](F1_RESULTS.md).

## Decision

Prioritize an explicit continuous **velocity/progress + heading + validity** Guidance interface. First falsify the inexpensive original-BFM `T(G,u)→legal z` interface in matched physics. P-BFM native ERI is the preferred alternative execution backend, not a feature we can append to the old actor. I-BFM is an independent interaction reference, not a drop-in human-wrist controller. **S0 proves legal interfaces and archived-action parity, not physical root-speed control or successful stopping.** Contact Adaptation is not yet established as necessary.

## Reproducible identities and actual tests

| Item | Exact identity / result |
|---|---|
| Original frozen weight | `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`, 3,384,968,272 bytes; unchanged after read |
| Original interface qualification runtime | `16e103e3bdc0dd53bb70fe6f7aee3920a885a263` |
| Final metadata/helper/test runtime | `168a91b` (full SHA in cost ledger) |
| Original action-contract SHA | `705fa1d4072b31523ef943c1121c8652f2ab09ba4d1bd4066b57adf8f30b36dd` |
| P-BFM source | `ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7` |
| P-BFM public browser export | pages `2401ffa16598c8efeb8512a774a71ee38bd8ae73` |
| I-BFM source / saved model metadata | `b4547263e925aedc16ade5427d03e9884f9fa3c2` |

Original replay: 21 archived real BFM control substeps across the three L3 windows, 42 frozen-model calls. Explicitly permitted actor fields versus the old full observation input produce **bitwise identical actions and q_PD**, also matching archived execution. No simulator instantiated; zero physics steps. Load/stage 2.214s, complete qualification 40.015s including checkpoint hash IO; actor-only GPU allocation peak 137,262,592 bytes. This is not full simulation throughput. [Machine result](s0_original_qualification.json).

Eight clean-contract tests pass: observation allowlist, legal latent/nonfinite handling, action coordinates, unsupported backend fail-closed, own progress memory, common prefix, yaw coordinates and valid zero-speed commands. 32 P-BFM velocity helper cases match upstream exactly. The I-BFM same-point spherical helper passes, but the antipode norm check failed its original `1e-5` criterion by producing absolute error `3.573103987e-5`; preserved, not relaxed. No I-BFM checkpoint forward or physical test was performed. [Metadata/helper audit](s0_external_audit.json), [erratum](S0_ERRATUM.md).

## Three backend contracts

| Contract | Original BFM | P-BFM | I-BFM |
|---|---|---|---|
| Available locally / executable evidence | Original weights + existing qualified MuJoCo chain; new wrapper archived-action exact | Source, config and public ONNX asset metadata; weights not downloaded/tested | Source, saved config, LFS pointers; weights not downloaded/tested |
| Actor inputs | state64, last_action29, history372 | Original proprio + heading + velocity_command + depth; estimated velocity internally | state98 = robot64 + object34, last_action29, history372; **indirect privileged B dependency** |
| B input | state64 + privileged463 (527) | robot state/privileged; no terrain/partner | state98 + privileged497, includes object/interaction context |
| Latent | Direct 256D, norm16 | 256D behavior intent plus separate ERI | 256D target; actor-mode LOGO sphere log using current B; not old Direct coordinates |
| Root interface | None native; proposed T only | Native yaw-frame velocity `[vx,vy,valid]`, heading-error features | No verified equivalent ERI; object-goal interaction context |
| Normalization | Stored per-key statistics retained | Export incorporates normalization, depth encoder, velocity estimator | Saved per-key normalization + sphere prompt transformation |
| Robot/action | Same 29 joint names/order | 29 names/order match, wrist scale/PD differ | 29 names/order match, waist/wrist PD differ |
| Simulation | MuJoCo3.3.3, original 50Hz/200Hz; 10Hz high-level | MJLab/MuJoCo3.8 stack; browser/Warp export differs | MJLab/MuJoCo3.8 object environment |
| Current use | Safe existing execution backend; root-control hypothesis unqualified | Preferred ERI candidate pending asset/backend physical qualification | Interaction reference; deployment permission and action mismatch unresolved |

### Original BFM

`humanoidverse/agents/fb/model.py` actor normalizes permitted observation and calls `_actor(obs,z,std)`; no ERI or LOGO. The 2048-wide, six-block model uses z256 norm16. Explicit action conversion preserves original operation order:

`q_PD = offset + clamp(action*5,-5,5) * stored_q_PD_scale / 5`.

The stored q_PD scale already contains the historical factor five. No new action saturation or PD change was introduced. The new executor rejects unsupported observation keys rather than relying on their names being ignored.

Behavior provenance matters: the existing `motion_walk_task` bank includes the M10 offline arm/waist-corrected motion asset followed by native goal inference; it is **not unmodified raw motion**. `goal_arm5` originated from an offline constructed/IK pose followed by native inference. The bank is71 frames/1.42s, with stored closure joint gap19.84° and latent angle35.55°; a cyclic call is not automatically smooth. [Source manifest](s0_source_bank.json). Neither loads I/C0/T30 actor online. T uses a versioned native bank, but is not prior-free learning or an established root-velocity controller.

### P-BFM: real ERI and available public export

[velocity_context.py](https://github.com/Roboparty/P-BFM/blob/ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7/pbfm/agents/velocity_context.py) uses current yaw-frame `[vx,vy,valid]`: `[0,0,1]` means valid stop, not missing command `[0,0,0]`. Reference future-mean context is reference-data processing, not permission to read future human trajectories.

[behavior_context.py](https://github.com/Roboparty/P-BFM/blob/ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7/pbfm/agents/behavior_context.py) represents heading as `[valid*(1-cos(error)),valid*sin(error)]`, not a raw yaw-rate field. [Actor code](https://github.com/Roboparty/P-BFM/blob/ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7/pbfm/agents/nn_models.py) estimates velocity from causal proprio/depth rather than reading its training target. Depth/history/calibration must be reconstructed, not filled with unexplained zeros.

Public [export config](https://github.com/Roboparty/P-BFM/blob/2401ffa16598c8efeb8512a774a71ee38bd8ae73/interactive/pbfm/config.json) describes step364822528, `direct_depth_full_v1`, policy 150,909,102 bytes / four parts; declared SHA `42872f98404abd7a6e624b871f8b00075ad266146908965afc4dfeb2f6fc910e`. Backward ONNX 820,629 bytes, contexts162,600, robot assets22,112,097: proposed minimum bundle **174,004,428 bytes (165.943MiB)**, excluding any optional runtime dependency. These are metadata sizes, not verified downloaded weights. Reassembly may require another150,909,102 bytes; reserve <0.4GiB for this bundle, not a replay.

The export consumes 8×18×32 depth frames, sampled ages35/30/25/20/15/10/5/0 from a37-frame history; max optical depth2.5m. Proprio history is four frames, newest first, grouped by keys. Native ERI code is available, but its response to wrist contact has not been tested here. No inference or training FPS claimed.

### I-BFM: hidden dependency must be counted

Saved [init_kwargs](https://github.com/Roboparty/I-BFM-Code/blob/b4547263e925aedc16ade5427d03e9884f9fa3c2/checkpoints/ibfm-166m/model/init_kwargs.json) and [model config](https://github.com/Roboparty/I-BFM-Code/blob/b4547263e925aedc16ade5427d03e9884f9fa3c2/checkpoints/ibfm-166m/model/config.json) are authoritative over generic YAML. 34 object fields include pose/velocity/target-relative terms, hand contact/spatial terms, presence and task identifier. They are not human-partner force observations.

Crucially [model.py:117–143](https://github.com/Roboparty/I-BFM-Code/blob/b4547263e925aedc16ade5427d03e9884f9fa3c2/ibfm/humanoidverse/agents/fb/model.py#L117) routes actor→actor_prompt→current B→sphere_log_map. Thus the small actor filter does **not** eliminate `privileged_state` dependence. Current-B or object truth cannot silently enter C-S Guidance. LOGO is not a compatible transformation to attach to the original checkpoint.

LFS parts are1,700,000,000 and1,699,326,624 bytes: total **3,399,326,624 bytes (3.16587GiB)**; [restore script](https://github.com/Roboparty/I-BFM-Code/blob/b4547263e925aedc16ade5427d03e9884f9fa3c2/restore_checkpoint.sh) declares whole SHA `25b0d98c83f059289e97a3634cd04dd211dafb674c341e783fd86cb1ffd41cd9`. Parts plus assembled file peak~6.33GiB before framework memory. No download performed. The210GiB replay is unnecessary for this S0 audit and existing prompt inference. Carry/push/kick scripts have task orchestration; their existence is not human force-guidance evidence.

## Action/PD incompatibility is material

The shared29-joint ordering passes exact comparison. This does not make the actuators interchangeable:

- Original wrists pitch/yaw: kp16.77833, kd1.06814, raw-to-PD slope0.37250429. P-BFM export: kp8.61103, kd0.5482, effective slope1.94517961 (~5.22×).
- Original/P waist kp300/300/300; I saved checkpoint waist kp40.17924/28.50125/40.0. I wrists also use8.61103/0.5482.
- Tiny other slope differences in JSON reflect floating representation; the audit retains all differences without calling every one a physical mismatch.

Accordingly no external weight was loaded into the old PD. A future P/I test must preserve its own robot contract and first be labeled an **independent positive control**, not a same-weight/same-physics superiority claim. F1 recovery qualification only covers its verified MuJoCo3.3.3 solver/settings; no transfer of that proof to3.8/MJLab/browser.

## Clean architecture actually delivered

`research/interaction_steering/` has no import of `research_notes`, no trainer, global CFG mutation or simulator creation at import:

- `contracts`: C-S typed fields, Guidance, backend identity; forbidden partner/clock/active/task/future fields fail closed.
- `source_adapter`: experimental forward-only T; owns phase, uses versioned native bank; rejects unsupported lateral/yaw commands. Goal-bank identity fixed, execution z varies. This is not proof that style or root motion is preserved.
- `guidance`: fixed continuous admittance reference, parameters explicit; no learned-U claim or force-to-skill classifier.
- `executor`: original frozen actor with exact normalizer and PD map.
- `backends`: P yaw-frame command encoding; unqualified backend cannot execute.
- `tasks`: exposed full-state reference and strict common-prefix checks. Existing physical worker is to remain a separate isolated process behind this boundary; **new physical adapter not yet qualified**.
- `evaluation/provenance`: actual/predicted identities and immutable hashes.

C-S allows robot q/dq/IMU, four actual action/proprio-history frames,25 ideal-force frames, actual last-z, known wrist installation and known guidance mode. No anchor, target velocity, future onset, simulator active truth or planner phase. The mode is a declared task interface, not a hidden future-answer label. Ideal force is still a sensing assumption, not verified G1 hardware.

The existing physical-state adapter and validated reset/solver/history mechanics are reused read-only via an isolated worker in S1; no copied nested experimental framework or changed callback is claimed as complete. S0 tests validate schema/common-prefix contracts, not new physical restore rollouts.

## Costs, limits and next decision

See [cost ledger](s0_cost_ledger.json). S0 charged below7200s, including preparation/IO/failed helper/test/reporting allowance; actual GPU weight replay40.015s and metadata audit18.682s are subsets, not extra additions. External model forwards0, physics0, training0, external weight download0. Source snapshots are small text/config files; no210GiB asset and no dependency installation. Both external repositories use CC BY-NC4.0; retain origin and asset-specific rights, no commercial-use conclusion.

Original T has the lowest entry cost and strongest same-backend comparison. P-BFM has the most direct native Steering contract and modest identifiable inference bundle; its real porting/physics cost is **unknown**, not inferred from file size. I-BFM requires more assets and object/permission/PD adaptation. [S1 draft](S1_PRE_REGISTERED_PROTOCOL.md) proposes a bounded original-backend test first; any P download or different backend run needs separate explicit approval. No S1/S2/S3 has started.
