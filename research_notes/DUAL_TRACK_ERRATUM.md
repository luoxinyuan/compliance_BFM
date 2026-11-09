# DUAL_TRACK_ERRATUM — new engineering entry only

2026-11-09. Old research outputs, old runtime, BC500, original BFM/PD are unaffected. Failed directories are immutable and counted. No failure was bypassed by loosening probability/KL/physical exact-repeat assertions.

|Attempt|Observed defect / evidence|Correction / boundary|
|---|---|---|
|high-01 @36666fe|New entry omitted existing abstract-socket shim; Manager failed AF_UNIX bind `Operation not supported` before physics/update.|Import existing m11_ipc, commitf48e3bb; no installation/shared changes. Original log and ERRATUM.json retained.|
|high-02 @f48e3bb|Static inspection found generated configurable update referenced `contract` without declaring it.|Stopped positively identified owned PID202214 before benchmark; explicit argument/default addedda0b530 and old/new numeric regression2312784. No result used.|
|high-03 @6fdef77|Old H100 cache comparison rejected maxqvel4.4303666e-6.|Initial TF32-only explanation was a hypothesis, **not a proven cause**.|
|high-04 @6c7232f|Same old-cache mismatch persisted with explicitTF32=false.|TF32-only explanation falsified. Independent current-backend cache pathde06aac. Original Pro5000 historical prefix actually matches current run exactly, so cache creation is unnecessary on that branch. No threshold change.|
|high-05 @005e092|Parent exact-prefix/branch/RNG/disk-history checks passed. Child wrapper incorrectly required generated cache file, although original-exact-prefix branch legitimately writes none.|Wrapper now accepts existing empty prefix directory and lets original backend checker choose its verified path. Old cache untouched. Isolated failed child, no physical labels/training promoted.|
|fb-01 @2f37cc1|New benchmark tried assigning immutable Pydantic `cfg.train.batch_size`; validation failed before build/update.|Keep original1024 default and assert it. Segment2 replaces config through model_copy forcompile, commit3fa9b8e. No frozen official config mutated.|

Successful high-06 runtimea2efe6c executes both exact update regressions, original probability mismatch rejection, raw/normalizer identity, real cross-reset next-step identity and parent/child isolation. fb-02 runtime3fa9b8e executes128eager+128compiled full updates and exact model/Adam readback. resume-01 independently recreates actual prefix and allRNG/model identity.

The final JSON loader was added after measurement with identical dataclass values; final source regression is recorded separately. This is not claimed to be the same Git commit as the earlier physical benchmark.

A read-only accounting helper initially used system Python metadata and found no torch distribution; it wrote no completed ledger. The final ledger uses the project venv. This operational failure did not affect any experiment or package installation.

Remaining unverified: pure isolated compiler CPU time, cold-storage bandwidth, full online mutable-base actor+environment throughput, arbitrary-state/interrupted-minibatch exact resume, contact-expanded base topology, Isaac batched Tether. These remain unknown and are not silently treated as achieved.
