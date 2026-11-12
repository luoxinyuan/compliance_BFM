# Issue62 Pilot launch packaging record

2026-11-12. Yi approved the fc6531c revised lock in conversation; public approval record is Issue62 comment6816360266. Pilot stages are independent of engineering preparation; all earlier failures/results remain.

First attempt runtime8697b29704685e5ee072e45d776fb6b529f322ac stopped during import, before source/replay initialization or any Pilot update: the new stage lacked the stage-local `prep/pointer_layout.so` used by the qualified F1 MuJoCo state contract. Its full process cost105.2319s is retained in the job ledger. No probability, physical or numerical assertion changed.

Ordinary packaging repair: copied the already-qualified helper and pointer_layout.json from preparation, no new compiler/install or model change. Both .so files have SHA256 `8e7d74e1ffb947f1ccfc82b083dbf91b6433e2da847c29dc8e96ef1709dfb439`. A shell quoting attempt failed locally; another unchanged queue invocation was rejected by the existing-job assertion before creating any GPU child. Both command traces remain; current launch/CPU allowance covers this work.

Retry runtime `cf247d5f87beab89ff8be1a6eeaca764f0f93599` uses distinct pilot-A-r1/B-r1/C-r1 output names. It changes only orchestration names; same approved manifests, model initialization, mathematical updates and evaluation. The first failure remains in aggregate costs. No previous output/checkpoint deleted or overwritten.

Queue: single GPU0; each arm128 completed rounds×250 actual50Hz transitions,2complete updates/round. Failure/normal stop blocks subsequent jobs. Separate dev processes follow the three arms, plus shared initial/BC500 references and CPU-only grouped probe. No sealed execution or subsequent stage. Startup/CPU allowance600s is conservatively charged in addition to preparation2560.4251s; live jobs/IO/failures are then added within8h/8GiB. Output soft guard reserves1,200MiB for final save/failure; hard cap and ≥100GiB shared free guard remain. This can end below32k if real cost exceeds projection; no silent expansion.
