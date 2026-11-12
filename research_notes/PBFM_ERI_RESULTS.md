# P-BFM native ERI: bounded independent physical qualification

2026-11-12, Issue61. User approved following S1 with “可以”. [Locked protocol](PBFM_ERI_PROTOCOL.md). Actual runtime `9d29641c4a502d75e6641045565b8150779adcb7`. **Complete and stopped: no-contact gate failed; contact tests not started.** No training, search, newsealedstate, replay download, shared environment change or S2/S3.

## Main finding

Native ERI is connected and changes low-level actions, but the selected public `forward08-arms-spread` preset plus this local execution backend does **not** provide the needed low-speed/stop control. Valid zero command still produces sustained~.41m/s forward motion. Positive commands slightly increase speed; removing them does not stop. This is a bounded negative result about one pinned preset/export/backend, not a rejection of every P-BFM behavior or ERI architecture. BC500 and prior BFM results remain unchanged.

|Command peak m/s|Actual moving vx m/s|Final2s vx m/s|12s root dx m|Slip peak m/s|Final yaw deg|Hard boundary|
|---|---:|---:|---:|---:|---:|---|
|0.00|0.4105|0.4108|4.752|0.921|-6.458|None|
|0.09|0.4319|0.4089|4.871|0.975|-6.270|None|
|0.18|0.4508|0.4098|4.954|0.897|-6.219|None|

All six12s trials completed without fall/nonfinite/joint-range+.02rad termination. Each repeated trajectory and all actor inputs/actions are exact. Correct speed ordering alone does not satisfy the predeclared gate: zero drift, final speed and slip conditions all fail. No contact or adaptation trial was started. No future partner trajectory was used; these are explicit no-contact diagnostic command schedules, not a claim that Guidance was learned.

## What was actually loaded

P source `ff8f45c7945a3a19b8755bfd62b4a9ed0efb2fc7`; public browser export `2401ffa16598c8efeb8512a774a71ee38bd8ae73`. Policy364822528 SHA `42872f98404abd7a6e624b871f8b00075ad266146908965afc4dfeb2f6fc910e`,150,909,102bytes. Explicit selected downloads173,802,787bytes (asset subset, not exactly the larger S0 suggested list), plus existing smallconfig/source and150,909,102byte reassembly; no replay. [Per-file hashes](pbfm_eri_assets.json).

Public contexts file git blob matches upstream exactly `1a2ab29fd6456ca69e21e4e871202845f369c2e8`; fileSHA9a475627faf6cd66a718bd8aa1f9462f866a3ab13678ce290bc8b22880d3e0f0. Executed z is unchanged published256D float32, norm16.00000057, bytesSHA f0f20adecfaeb8f400cbd1f4ad64560a525a3c001e7fcfe4454111d448fe986d. Its opaque upstream zSha256 b3ccab03… does not match tested raw-float/JSON digest conventions. Encoding/source meaning remains unresolved; do not claim that secondary field was verified or that downloaded contexts were corrupted. [Digest audit](pbfm_eri_z_digest_audit.json).

Preset provenance is native reward-derived forward.8/arms-spread; it is not ordinary arms-down walking and was not chosen by post-test searching. One fixed z throughout; ERI separately changes world-forward speed and current-yaw heading error. No original I/C0/T30/BC500 actor output enters this backend.

## Portability and input correctness

Original native training was MuJoCo-Warp. Public browser implementation explicitly identifies MuJoCo-WASM3.3.8 and an archived matching robot asset, not verified original training-path identity. Here: existing Python MuJoCo3.3.3, ONNXRuntime1.22.0 **CPUExecutionProvider**. This is a separate port; no numerical equivalence to browser/Warp claimed. GPU used only for video rendering. No inferred Pro5000 GPU inference throughput.

Robot29joint order, range, armature,200Hz DC motor torque-speed clipping, native PD/action scales and50Hz control are preserved. The native wrist PD differs from original BFM; no old PD was substituted. No external human force applied. Joint/action targets remain separate from actual q; hard physical margin unchanged. The default pose/root is the browser's explicit flat row4 reset; history then develops through real execution, no borrowed teacher history.

42 successive synthetic frames run through the actual extracted upstream JS functions and the local port: state, last_action,372D history, heading, independently transformed velocity command,8×18×32 depth history, z, scaled action and q_PD are **bitwise equal** (max error0; fixed criterion1e-7). [Parity](pbfm_eri_interface_parity.json). Live depth is calibrated optical-Z MuJoCo ray geometry with native crop/filter/quantization/history, not zero-filled; initial range .1333–.6745 normalized. Depth processing parity does not prove all renderer/engine dynamic equivalence.

ONNX graph consumes velocity_command directly in actor concatenation; no partner truth, future trajectory or policy hidden answer is supplied. Same stored inputs at0/2/4/7/10s, changing only command demonstrates nonzero action changes. Approximately .09/.18 command changes produce .26–1.08deg/.52–2.12deg max q_PD differences across these states; invalid `[0,0,0]` differs from valid stop `[0,0,1]`. Thus “the command was never wired into the network” is contradicted. This offline intervention proves command sensitivity, not successful stopping. [Full same-state table](pbfm_eri_same_state.json).

## Quality and interpretation

Loaded-slip peak .897–.975m/s, with peak normal load44–130N. Peak events are touchdown/transient by pre/post body-contact continuity, not unloaded brushing; p90 tick-slip .021–.025m/s and exceeding-tick fraction1.33–1.58%. Retain both tails and mostly low-slip behavior, never describe the whole walk as sustained sliding. All peak-risk failures retained without raising normal-load threshold. [Quality distribution](pbfm_eri_quality.json).

Maximum q_PD target jump~29.42deg per20ms, actual largest joint change~2.12–2.20deg per5ms: do not confuse target with physical rotation. No actual joint-limit excess: smallest observed remaining margin~.19rad. Max measured torque~35.1Nm. Final yaw−6.2 to−6.5deg is within the yaw reference, but stop speed is not. Error/force contact metrics are not applicable to this no-contact test; actual external force is exactly0.

**Established:** pinned public inference assets load; native input/action helper parity; commands affect network output; physical motion repeats exactly; this preset/backend keeps walking through valid stop.

**Compatible, unproven:** reward-derived forward latent may retain speed intent not removed by ERI, and .0–.18 commands may be outside its reliable execution range. Export/backend differences remain another possible contributor. No isolated source/engine comparison was performed.

**Not established:** native upstream browser/Warp would show identical behavior; allP-BFM latents fail to stop; contact dynamics cause this failure; Contact Adaptation is required; any training method fixes it. No attribution to force cap is possible because force was0.

Next scientifically useful proposal is one matched native-reference/export check of the same zero/low command, then a separately authorized neutral-goal vs forward-preset comparison if needed. Do not start contact adaptation before resolving no-contact command behavior, and do not launch training to conceal an unqualified execution interface. No further experiment automatically scheduled.

## Accounting and delivery

Six real12s trials:14,400 actual5ms ticks =14,400 globally pre-reserved ticks,3,600 actual50Hz decisions. No hidden reset/prefill stepping; all warmup included. Physical supervisor250.031s includes load/save; trial inference/control227.153s. Old restore qualification was not rerun. Limits2h/2GiB/30,000 allprocess ticks retained. [Stage ledger](pbfm_eri_cost_ledger.json), [gate](pbfm_eri_gate.json), [full phase outcomes](pbfm_eri_physics.json).

First render failed because ffmpeg was absent from PATH; native existing imageio_ffmpeg executable used in a new analysis02 output, with original failed script/output retained. No package installation and no physics rerun. Offline initial parity used supplied transformed velocity; the final independent-rotation parity above supersedes it. A legacy cgroup path read was unavailable; actual cgroupv1 quota was read correctly. Failures/preparation are inside the stage elapsed envelope, not omitted.

One three-panel12s actual-qpos1× video compares the three commands; full decode and local/remote hash verification. Mac `~/Projects/compliance-bfm-diagnostics/videos/pbfm-eri-20261112/pbfm-eri-zero-and-speed.mp4`. [Video hash](pbfm_eri_video.json). Plot: [command/actual velocity/slip](pbfm_eri_response.png). All raw histories and failed assets retained privately; public reports contain aggregate evidence and source only. CC BY-NC/source robot licenses retained; no commercial-use permission asserted.
