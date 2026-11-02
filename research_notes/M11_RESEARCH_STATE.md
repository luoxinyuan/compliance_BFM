# M11 Privileged Contact Expert Learning — current state

2026-11-02: bounded training and diagnosis finished; expert objective only partially met. Read M11_RESULTS.md before starting work. M10 deployed policy remains the prior/baseline; M11 does not replace it. No new Student.

122,880 RL decisions / 2,451,456 training physics steps / 2,289 minibatch updates / 783 completed train episodes. Development-selected iteration60 improved straight12.661cm/4.610N→7.832cm/4.144N. New final straight failed8.493cm/4.139N→37.667cm/4.588N; left/right tracking improves over prior but force does not; handwritten expert stronger there. Best SHA bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f; final selection unchanged after exposure.

Audit: raw likelihood/worker GAE/legality valid; force reconstructed exactly;5N clipping flattens force magnitude cost, but raw-demand/tracking cost correctly penalizes failure. No force-reward RBF sigma. Read-only observation cleanup does not fix the failure. Full training noise on failed condition9.329±1.167cm, deterministic37.667cm. Executed noise with mean memory9.173±1.202cm; fictitious noisy memory with mean execution37.476±.723cm. This narrows the gap to physical execution/closed-loop dynamics rather than a memory-only bug; do not label fictitious memory as deployable.

Source branch gaoe05/m11-privileged-expert. Run bridge-v2-owned/runs/m11-privileged-expert-20261101; all failed attempts/raw results preserved. Final physics c8ec3f7, training b995f37, memory diagnosis ca1ba59afe6e221eba2b8f10e2a878a2fe72528e. Source-only publication is authorized; no main/force/merge/weights/data push.

All owned GPU work exited within predeclared4h wall/8CPU/32GiB RAM/8GiB VRAM. Shared BFM/PD/weights untouched. Mac videos/m11-privileged-expert contains3 paired real-state1x MP4 and INDEX. No Bridge/watch/LaunchAgent.

Next recommended work: one matched-budget fixed-noise versus gradual-reduction acquisition to address stochastic-to-mean deployment gap, preserving physical force and reward. New final conditions must be sealed; existing final is now exposed diagnostic. No new run automatically launched; do not retry consumed stages or overwrite results. Do not claim M11 has delivered a reliably better expert or blame all remaining issues on BFM.
