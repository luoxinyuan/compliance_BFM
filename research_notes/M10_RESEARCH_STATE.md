# M10 — Task-appropriate whole-body compliance

Status: completed, partial scientific improvement; M10 Student is research-only.

See [M10_RESULTS.md](M10_RESULTS.md) for the final physical comparisons,
source attribution, failed same-state expert handover and next decision.
The initial plan below is preserved as historical pre-execution context.

This personal review branch inherits the actual M9C source commit
`82f7eb7786ad569ae89473cdd4a1e42252daf79b`. All M0–M9C reports retain their
original runtime/checkpoint identities. M9C's transferred Student remains a
research model, not a replacement for M8/M9B.

## Publication and provenance

The 167 commits absent from the fetched team branches were inspected across
all reachable unique blobs and commit messages: 1,204 objects, 4,942,948 text
blob bytes, no binary/large asset/credential matches. Reports contain compact
metrics and server-local provenance paths, not model weights or physical
trajectories. Existing experiment histories are preserved. Upstream main
`c002f394f0b77d2f8a85356667fec5eda0f4a1af` adds optional accumulated latent
commands; it is not silently merged into the verified experimental runtime.

## Research question and execution plan

Locate the earliest source of contact-hand lowering, persistent walking lean,
and incidental arm raising during torso lowering. Compare saved references,
native physical execution, experts/projection and old/new Students before
choosing the intervention. Distinguish pelvis and torso orientation from their
relative waist motion; quantify full XYZ load and actual hand motion.

Improve the faulty source layer while retaining frozen BFM and original PD.
No force-cap increase, partner re-anchoring, hidden deployment targets, arm
locking, or exclusion of adverse trajectory intervals. Small-wrist regression
must be reported even if locomotion improves.

Initial finite M10 compute allocation: one available GPU, at most 8 CPU cores,
32 GiB owned RSS, 8 GiB owned VRAM, 8 GiB new outputs, 7,200 cumulative supervised
compute seconds. This is an operational allocation, not a scientific success
criterion. Recheck live GPU availability before launch; no other user's process
may be interrupted. No model execution on CPU as a GPU substitute.

Use separate, immutable attempt outputs and committed runtimes; retain failures.
Dev choices and held-out evaluation are fixed before fitting. Final checkpoint
selection must not use held-out outcomes. Only task-appropriate improvements
justify Student transfer; additional optimization alone is not evidence.

Outputs: personal `bridge-v2-owned/runs/m10-task-appropriate-20261101`;
source: personal `trees/m10-task-appropriate/repo`. A small set of actual-physics
1x comparison videos plus INDEX will be delivered to the Mac's
`~/Projects/compliance-bfm-diagnostics/videos/m10-task-appropriate/`.
