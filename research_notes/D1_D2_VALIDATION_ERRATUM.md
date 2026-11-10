# Issue57 short validation erratum

2026-11-10: e0c5c6b new-process resume-01 restored the model/six Adam/replay/RNG/full physics exactly, and subsequent collection was bitwise identical. The next actual FB update had identical logged metrics/RNG/replay but non-identical model and Adam tensors. The first run remains failed under the original exact comparison. Full tensor differences are retained privately; no tolerance was relaxed.

A new engineering-only entry enables deterministic Torch backward algorithms and an explicit CUBLAS_WORKSPACE_CONFIG, records/asserts that numerical backend contract, and fsyncs the published pointer directory. A paired short rerun is pending; this is not a formal FB restart or reward/boundary change. Old Issue56 lost512-update weights remain unrecoverable regardless of the new result.

Second pair d990dff control-02/resume-02 also failed exact post-update equality, with the same model/Adam digests as pair01. Deterministic algorithms therefore did NOT resolve the discrepancy; a generic GPU nondeterminism explanation is not established. Both readback and subsequent collection were exact. Long-run admission remains blocked. Original failed outputs are retained, including original result.json left at its launch status by the old diagnostic entry; jobs sidecars are authoritative. The new entry writes independent failure/finish evidence.

Final: third attempt8ecd211 through the original factory still produced identical mismatch hashes.
Recorded module training/requires_grad, normalizer/filter order and config matched; cause unresolved.
Seven complete engineering updates total, no further training. External-stop fallthrough gate was
subsequently added and CPU-tested, not physically requalified. Old original erratum remains unchanged.
