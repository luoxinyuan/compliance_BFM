# R2 engineering lock — not a scientific training recipe

Authorized by Yi on 2026-11-14 following ba8fd2c. Engineering costs are INCLUDED in the new16h/12GiB paired pilot; engineering upper limit3600job seconds. GPU0 serial with D1. No sealed or historical-result rewrite.

## Fixed initial function and execution

E/I: A128 U weights (d97f8c515fb76f34b7ff3d75b7a29208a68987afbf75de5cf727f8227180114b archive core), original frozen BFM33f410c1 function, independent trainable copy of Actor policy blocks5–6 only. Original embeddings/blocks0–4 and B_R/normalizer/PD remain frozen. Initial E/I action and legalz must equal original A128 action at the same input. Tail parameter count measured, not assumed. New32D public-goal slot enters a zero-initialized correction to U, preserving initial function; E plain Goal encoder vs I B_I. No old29D adapter or Envelope.

One std.01/clipped.06 noise draw after original tanh, final[-1,1] actualaction is physics/replay truth. Original physical boundary unchanged. Goal template is fixed, public zero-interaction/hold; no actual active/partner truth as actor input. Q sees declared current privileged state. B_I outcome is current successor contact_privileged_v1[6:22] under existing physical scaling, training-only; goal template does not read actual successor.

## Engineering-only probe

Fresh twin physical Q, Adam1e-4,batch32, targetEMA.01; physical reward/time discount/termination unchanged. Collect250 actual50Hz transitions, execute2 Q update groups and1 Actor/U group, save complete state, then5 real transitions plus1 full update. New process must restore identical checkpoint and reproduce continuation physics/rows/RNG/parameters/Adam exactly; no tolerance changes. Original BFM actor state hash must remain unchanged.

Engineering retention coefficient1 in q_PD squared rad units, SF actor coefficient0. These are explicit runtime measurement settings, NOT calibrated scientific values and NOT a formal arm. E/I implementation can train FI/BI with actual successor-density Bellman +.1 orthogonality; no frozen F_R term. Scientific retention dual rule and FI auxiliary coefficient require separately recorded train-only calibration before any formal run. Engineering models are discarded as training starts; no warm-start from this probe.

Original source/current episode are fully reconstructed through existing exact restore; complete mutable core includes tail/targettail/U/Q/goal modules/Adam/normalizers plus append-only replay/physics, RNG and state/history. The small existing pointer-layout auxiliary must be copied with identical hashes before launch.

## Gate and accounting

No hard physics failure, permission leak, unequal initial function, malformed nextstate or failed full recovery may be labeled a qualification pass. Preserve failures. Snapshot runtime before each job; actual sample/update/IO/failures charged. Measured online throughput below2transitions/s or projected output over12GiB requires a revised size/cost decision before formal training. This engineering probe has no authority to select a scientific checkpoint or alter the reported old failure.
