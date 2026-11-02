# M11 continuation — stochastic training versus reliable execution

User-authorized continuation of Issue32, not a new research milestone. Parent466266385fab2fec9852253d661ecd24cd675c64 and all original results retained. No new Student/contact point/reward/cap/network/BFM/PD changes.

## Fixed before physical probes or training

One actually free GPU, sequential A/B;8CPU,32GiB owned RAM,8GiB owned VRAM,8GiB outputs. New total supervised compute budget28,800s (8h), each acquisition up to12,600s training-process time,160x3x256=122,880 **additional** high-level decisions each. Same source model/optimizer/RNG; original best iteration60 SHA bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f. Common physical sampling RNG streams; episode timing may diverge, not claimed identical training trajectories. Registered in m11_continuation_protocol.json.

A holds the source's per-dimension exploration std (~.08). B holds it for20 batches, decreases log-linearly to one quarter by batch120, then retains nonzero~.02 for40 batches. Updates happen only before a complete sampling batch. Both freeze log_std (requires_grad=False, original optimizer group/state retained but lr0), skip the old learned-logstd clamp, and assert each worker's actual std equals the update distribution. The actor/critic and source optimizer moments otherwise remain identical at initialization. Test includes std below historical clamp and rejects post-sampling std changes through old-logprob verification.

Zero-mean residual with exact source std and seeds101/202/303 is applied physically on old development straight and exposed prior-final straight; compared with saved M10, learned mean, and learned+noise. Original legacy observation reads are used only for exact diagnostic comparability. No final-selection claim from this exposed condition.

Development: original7 conditions plus exposed failed straight; mean and sampled(seed202) at initial and every40 batches. Selection uses mean failure count first, then mean normalized per-condition cost plus half worst-two normalized cost. Costs retain actual force/error/cap, plus measured no-contact/root and local drift. Normalization uses newly paired M10 baseline, floor.5. Stochastic rows do not choose checkpoint but remain available. This makes tail deterioration explicit without requiring every history counterexample pass.

New final combinations are in protocol, not used by training or selection. Three sampling seeds401/502/603 plus mean for each selected arm; M10 deterministic baseline. No reselection after final. Family metrics, raw trajectories and failures retained. Fixed mean control schedule and causal own-command memory unchanged.

## Resume correction

Original m11_train.py now measures current resume weights before establishing a fresh output selection score; historical score is provenance, never attached to different current weights. If a distinct source best.pt exists, evaluate it separately and retain it if better, with its own optimizer state. Selection file binds score/evaluation/SHA. New continuation Selection helper atomically copies the actually evaluated candidate and records weight/evaluation SHA, score vector and protocol scope. Source artifacts never overwritten. This bug is not claimed to explain the previous from-scratch run.

## Execution / evidence

Run: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-continuation-20261102
Personal source: bridge-v2-owned/trees/m11-privileged-expert/repo/research_notes.

Finite sequence: zero probe, A fixed, B annealed, M10/selectedA/selectedB sealed final, actual-qpos three-controller videos. Supervisor records exact runtime, owned PID tree, resources and outputs. Stops on runtime/resource failure; no watcher, no unbounded retry or training extension. Source must remain fixed while processes use it. User's earlier permission to release the conversational turn while waiting on training still applies; a running job is not a completed result.

Training logs include actual std vector bounds, raw/tangent norm, latent angle, same-state q_PD delta, target step, reward components, old/post value EV, actor/critic/std gradient norms, common clipping factor, joint Gaussian KL, clip fraction, actor mean change and all completed episodes. Actor success is determined by physical mean-policy results, not EV.

Video rendering uses saved actual qpos/qvel; three1x synced comparisons (straight,left,local), each with M10/A/B, world oblique and overhead, actual force/anchor/heading. No Student/target reference animation. Copy only key MP4 and INDEX to Mac videos/m11-privileged-expert/continuation/ after completion and verify hashes/duration/decode; do not call remote-only videos delivered.

No new result claim at protocol registration. Both learning and non-improvement outcomes will be reported on the same Issue32; Project Master updates only after stage evidence is complete.
