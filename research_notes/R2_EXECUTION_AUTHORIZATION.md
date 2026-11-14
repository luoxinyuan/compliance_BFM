# Issue63 D1 + R2 execution authorization

2026-11-14: Yi explicitly approved “可以。启动D1和R2吧” after the ba8fd2c D0/D1/R2 deliverable. This supersedes DESIGN ONLY for the bounded implementation and execution described there, not historical conclusions.

- D1: R2_D1_MINIMAL_PROTOCOL.md unchanged science: P2/P3 exposed recovery states; V1/Control64/Return64/base-only first20ms, common V1 continuation to3s; max18000 physics ticks including reconstruction/failures,5400 aggregate job seconds,1GiB output. No training, active-state expansion or sealed.
- R2: R2_INTERACTION_FB_DESIGN.md new E/I partial-Actor online execution adaptation, max32000 actual50Hz transitions/arm,57600 aggregate job seconds,12GiB output. First3600 seconds engineering/qualification are included, not additional. Single GPU0 heavy jobs serial. New independent identities, original shared BFM/PD and old runs unchanged.
- R2 exact train-only calibration for functional retention and interaction auxiliary normalization must be recorded and locked before training. No dev-based coefficient search, no implicit reuse of old Return lambda. Initialization and complete checkpoint qualification precede real training. A128 is a shared U initialization, not a proven qualified actor.
- Original project240h aggregate remains; new stage ledgers are additions, not resets. Output excludes source worktree but includes all new data, failures, logs/checkpoints. Shared free space guard100GiB retained.
- Source baseline ba8fd2c86ec2994fc66f37e1dd5a156c3f89328f. D1 reuses archived exact-state restore incl efc_JT and original valid-array rules, without relaxed comparisons.

At this authorization commit: no new physics or optimizer steps yet. Live GPU0 empty, shared statvfs free242455937024bytes, cgroup CPU210 cores, available RAM~1TiB. These are preflight snapshots, not future scheduling guarantees.
