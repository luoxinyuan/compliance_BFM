# Dual-track V1: resolved decisions and remaining scientific choices

2026-11-09; new stage [Issue56](https://github.com/GaoE05/robot-research-bridge/issues/56).
This is a concrete review packet, not evidence that a training job has started.

## Approval provenance

[Master approval](https://github.com/GaoE05/robot-research-bridge/issues/18#issuecomment-6739473031)
and the current user instruction approve the six scope/budget decisions and the
1M FB probe. Three explicit subsequent user replies in this session also approve:

1. High gamma .99/100ms, FB gamma .99**.2/20ms, existing per-contact physical costs
   integrated over actual duration, no added economy reward. True terminal,
   truncation and physical release are different events.
2. A97D privileged contact/partner field, old robot normalization retained and
   frozen, new fixed physical scales; zero added input weights. The discriminator
   remains robot-only. No future, episode clock, final distance or task ID.
3. Equal standing/arm/moving source probabilities; near/mid/long/stop/zero/none
   episode probabilities20/20/25/20/10/5%; no extra high-level BC penalty; only new
   actual execution in FB train replay; original native no-contact continuous
   snippets for CPR, retaining its existing coefficients.

These replies approve the stated principles; the third question explicitly did
not silently approve all ranges, thresholds, optimizer settings or FB z schedule.
The machine-readable file stores approval and proposal separately.

## Concrete remaining recommendations — pending Yi

|ID|Recommended resolved setting|Why / cost / limitation|
|---|---|---|
|P1 task ranges|Near4–12cm at average.03–.09m/s; mid16–45cm at.06–.15m/s; long50–80cm at one of.09/.15/.22m/s. Distance and speed drawn independently within each family. Onset handoff+.5–1.2s; continuous deceleration.5–1s; hold1.5–3s; released recovery4s. Stop/continue are one parent.|Includes the prior arm16–30cm positive-control domain and marks larger arm requests as extrapolation. A smoothstep mean speed is not its peak speed; report both. No silent cap/K/D change.|
|P2 native source generation / independence|Standing, goal_arm5, motion_walk_task through original BFM only; physically generated source episodes and support-stratified handoffs. Dev10/source, sealed20/source as targets. Source-command/history hashes, common-prefix and partner-parent IDs must not cross splits.|Changing a seed, cutting another frame of one canonical motion or relabeling one prefix is not a new independent parent. The90 independent parents are **not yet generated/qualified**. Bounded native entry variations must be enumerated before generation, not invented after seeing results. If genuine groups are fewer, report fewer.|
|P3 high policy and optimizer|Exactly the #55 engineering branch1088→128→256 and critic2727→128→1; both BC500 actor and zero branch trainable. Old BC500 center/scale and log_std fixed; std.008 sampled once. Fresh Adam actor3e-5/critic3e-4; GAE.95;768 decisions/batch,6 epochs, minibatch256; first-batch100 value-only steps; no BC retention.|Explicit adoption of tested topology and conventional update values, **not their automatic promotion from benchmark to science**. The two KL arms differ only by approved KL limits. Fresh critic avoids treating BC values as trained returns.|
|P4 exact matched schedule|Conservative interpretation:196608 Expert decisions **total**, split98304/arm=128×768. Same initial actor/normalizers/critic and paired seed5601; immutable models at0/every8/final; independent30-parent dev every8; sealed once after selection. Serial GPU blocks, immutable code per running job.|Master says total; running196608 per arm would double the approved stage. A later independent seed is not included automatically. Student98304 is a separate conditional target, not mixed into Expert or FB counts.|
|P5 development selection|Retain initialization. A physical/native termination excludes promotion for that candidate, not the direction. Family-balanced cost `force_RMS_N+.03*active_error_cm+cap_fraction`; select strict lower cost after tail guards. Tail guard: no family median error increase>max(2cm,20% of initial), force increase>.75N, loaded-slip peak increase>max(.20m/s,50% initial), or released yaw worsening>10°. Report all failed reference lines, phases and individual parents, not just the aggregate.|The tolerance values are proposals and not inherited M1 gates. No near-zero ratio scoring. Existing M1 values remain references, and no-contact uses drift/support rather than undefined active-contact error. This selection can retain initialization without denying partial gains.|
|P6 FB reward and regularization|Contact scalar auxiliary reward is the same signed physical integral as the frozen line. Recommend **identity** auxiliary-reward normalization to keep physical units; fresh aux critic (or migrated critic as an explicitly tested warm-start, not claimed contact-trained). Existing CPR reg_coeff1, aux reg_coeff1 and scale_regTrue retained initially and their actual actor gradients reported.|Original EMA auxiliary-reward statistics were learned for another objective; copying them would silently change force-task scale. Identity vs fresh running normalization needs explicit selection. FB still optimizes its FB/CPR objective in addition to task auxiliary reward, so this is not pure PPO-vs-FB equivalence. The approved common physical reward alone does not select that balance.|
|P7 FB online z schedule|First1M probe: source-associated stand/arm native goal z, or native walking's causal phase z, throughout each episode. No contact-driven hand-coded mode switch. Mutable29D actor is free to change its response using97D feedback; z is a traceable behavior prior. Existing training goal/uniform relabel mixture50/50 remains. Evaluate both native prompts and reward_inference using newly collected extended states separately.|A finite native prior can support stable collection but may restrict coverage. This is a **proposal**, not an already verified schedule. No I/C0/T30/BC500 online action source. Do not claim universal task discovery or a learned latent selector from this probe.|
|P8 FB replay / CPR / checkpoints|Fresh CPU replay capacity1M; all online actual transitions, actual interpolated z and low-level action, target z, terminal successor before reset, separate terminated/truncated. CPR snippets8 frames from actual original native no-contact source executions, balanced by source and never crossing reset. Immutable complete0/250k/1M packages: model/6Adam/normalizers/replay/counters/RNG plus exact physical reconstruction identity.|No fake zero-filled partner labels on unrelated historical robot data. Explicitly recorded no-contact source has a valid zero physical force. Replay size and resumability must be measured after expansion; current format is not yet a complete resumable online run. Engineering output ceiling64GiB and shared free-space guard proposed, not a throughput promise.|
|P9 FB cadence and initial collection|Use original six-optimizer update recipe except approved gamma, batch1024,16 full updates per1024 new native transitions. First1024 real transitions are warmup **inside**1M; then updates only against actual collected counts. Report final fractional block and exact updates; do not automatically label it15625 if warmup was skipped for optimization.|Existing benchmark does not establish online scheduling or terminal correctness. The endpoint must stop at exactly1M true controlled transitions; source/reset steps remain separate costs, not extra learning samples hidden outside the cap.|

For P2, a concrete bounded proposal is: mean native actor only; standing handoff
3.8–5.2s with a genuinely distinct native preconditioning profile (stand/arm5
spherical interpolation fraction.02–.08 for.3–.8s, then actual standing.3–.6s);
arm-ready handoff5.2–6.2s with goal_arm5 participation.85–1.0 and a distinct
native entry profile; moving handoff5.6–7.6s with actual native phase offset
within±.1cycle and rate.9–1.1. These are proposed source-generator variations,
not force-to-skill rules or learned actor inputs. They require bounded physical
verification and explicit approval before use. Rejecting an unmaintainable source
does not erase the failure or turn a new seed into an independent replacement.
Record the3 reused source assets separately from the number of actual independently
sampled/hashed source-and-partner parent processes. Frame cuts of one generated
source episode stay in its original group. Dev/sealed generator seeds56031/56061
and their full manifests are proposed; existing exposed regressions remain separate.

The numerical suggestions above are intended to be approved or revised as one
reviewable packet. Formal entry admission remains closed while any applicable
decision has `pending_Yi` status. Selecting an option in the earlier three
questions is not retroactive approval of this remaining table.

##97D field order and scales (approved)

|Slice|Meaning|Permission / transform|
|---|---|---|
|0:6|installation present/wrist/torso + local xyz|Known attachment, not contact-active truth|
|6:9|actual current XYZ force|Current pelvis frame, divide5N; no change when cap10N|
|9:12|current anchor minus point|Privileged, current pelvis frame, divide.25m|
|12:15|current partner minus point velocity|Privileged, current pelvis frame, divide.5m/s|
|15:18|point minus root position|Current pelvis frame, metres|
|18:21|current partner velocity|Privileged, current pelvis frame, divide.5m/s|
|21:22|actual physical active|Teacher-only truth, never Student input|
|22:97|25 chronological actual wrist-force samples|50Hz, expressed in current pelvis frame, divide5N|

Only97D is added to original robot keys. No MPC phase or expert hidden feature.
Original B robot input527→624, actor465→562, F/critics928→1025,
discriminator remains527 (before its z concatenation). These are filtered-input
dimensions, not all raw observations flattened into one shared vector.
Original robot privileged463 is still available to the FB representation/critics;
high-level2727 and FB actor562 are not identical feature vectors. Information
tiers and current-vs-history scope must be reported rather than claiming an
exact input-matched algorithm comparison.

## Critical migration detail discovered before training

`humanoidverse/agents/nn_models.py:Block` applies LayerNorm **before** its input
linear layer; residual actor/F/critics use this block. Plainly appending97 zeros
changes mean/variance of every old coordinate, even with zero added weights.

`dual_track_fb_extension.py` retains the original LayerNorm domain and inserts
zero contact matrices immediately after it, before the original activation.
The old robot/z/action computation and affine parameters remain intact. The B
map's first linear has no input LayerNorm and gets an equivalent zero contact
matrix. The proposed fix is an input migration, no added hidden width or
force-to-motion controller. Step0 equality, new-gradient flow and fresh optimizer
membership are tested separately. No pre-existing result is withdrawn on this
account: #55 did not extend the contact input or report contact learning.

## Four-cell comparison and current missing cells

Interpretation proposed for review: rows are frozen vs mutable BFM; columns are
privileged Teacher vs restricted ideal-force Student. The high-level KL pair is
inside the frozen/Teacher cell, not a substitute for the four cells.

| |Privileged current feedback|Restricted deployable-permission actor|
|---|---|---|
|Original BFM frozen|BC500-derived zero branch, KL.02/.08: **not yet trained**|BC500 historical baseline retained; new Teacher migration/RL not yet run|
|Independent mutable FB copy|97D extension under migration check;1M **not yet started**|Not implemented/trained; privileged actor cannot be relabeled restricted by hiding inputs|

Missing cells are not zero performance or failed experiments. The first two
formal learning jobs await completed scientific configuration and actual runtime
validation. New restricted migration is conditional on useful Teacher evidence;
this stage does not silently authorize another1M restricted-FB run.
