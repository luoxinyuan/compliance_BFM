# Adapter Envelope V1 — approved inward numerical revision

Supersedes only strict FP32 inverse representation requirement in c13cc69; old preflight failure remains in ADAPTER_ENVELOPE_V1_PREFLIGHT_RESULTS.md. Scientific constants/thresholds unchanged. Approval6827917070 plus numeric6828464806.

Full machine contract: adapter_envelope_lock_v1.json. Status qualification_locked_not_yet_executed. No new physics yet.

Original map g=offset+(action*5).clamp(-5,5)*scale/5. q0=g(clamp(base)); margin=max(.02,.05*range); dminus/minus and dplus/plus=min(.12,.25*remaining margin). Noise before raw final clamp, then original joint-space Envelope. Requested q may lack FP32 preimage. Execute nearest inward representable action using nextafter then ordered-FP32 bisection for offset cancellation; assert actual g(action)-q0 inside original bounds using exact FP64 comparison of FP32 values, numeric difference≤1e-6rad. No allclose substitution in execution/replay/critic checks. Zero/unchanged action bitwise retained.

Training derivative is original analytic projection with straight-through derivative only across representational rounding; forward is actual bitwise execution value. Current/target/critic and replay all use executedaction. A path unchanged. B/C constrained stochastic distributions differ from A; compare packages, not isolated unconstrained adapter effects.

Frozen state manifest:
- B19-release: parent 8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e, time 11.800, core 0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984; archived actual B policy state; current frozen model B19 for all branches.
- B16-near-active: parent 2670e363f8c204b64ab32dd614237f032a571e1cd03f4e5c50524c3caef42f79, time 5.700, core e0c086aa7c019daf8a43b0f628fe82e0f00a55e4ec3f14fd263b1617df723b1c; archived actual B policy state; current frozen model B19 for all branches.
- B2-mid-active: parent 06c173441f3204e6b90e5e5fb230e9356f15e30e91495171915d5a4f6897bd70, time 5.500, core 38f9f971b0aa2a3e6871604606ae8dbc19673544ae286ab770dfc7e59a27a005; archived actual B policy state; current frozen model B19 for all branches.
- A83-none-recovery: parent 6a2fe7552234ceb9dd5944b5601c46e3424d3995699b22bc02e6d37410dbc314, time 6.000, core 69a1a4ad87a6c6c18a60cb31534473a408cbd9b3e99767e380d985327a7793f3; archived no-contact A83 state following real native standing prefix; not claimed native-only after handoff; frozen B19 applied to identical complete state.

On/off/envelope each mean/sample from identical full physics/RNG; B19 only new envelope branches, verify archivedon/off identity before reuse. Four distinct already-exposed parents. No sealed reads. Actual A83 no-contact recovery is labeled precisely; original nativeprefix regenerated/hash-matched under all branches. No claim A83 is pure native afterhandoff. Source unchanged by runtime Envelope.

All §4 thresholds are machine-locked verbatim. Any qualification failure stops B/C. B19 mean/sample3s, actualexcess0. Two active parents require nonzero effective adaptation. Off-relative quality and original absolute.45m/s/20deg references separately reported; release excludes anchor error. Oldshortonfailures not samewindow fullquality.

After qualification: fresh B/C seed6201 each128×250/256updates; A128 preserved. Newprocess recovery tests and originalmanifest, std.01/clip.06. Five-way20episode dev thenBI probe. Qualification is not yet demonstrated; planned work not results.

Numerical unit test first nextafter-only attempt failed to reach inward float under offset cancellation; preserved under stage prep/envelope_attempt01.py and jobs/rounded-operator.log. Fixed by bounded exact representable search, not constant/threshold change. CPU/CUDA29696 entries: maxcorrection4.7683716e-7rad; unchangedzero identity and gradients verified. All runtime paths still require actual audit.
