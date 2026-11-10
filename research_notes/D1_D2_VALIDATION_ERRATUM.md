# Issue57 short validation erratum

2026-11-10: e0c5c6b new-process resume-01 restored the model/six Adam/replay/RNG/full physics exactly, and subsequent collection was bitwise identical. The next actual FB update had identical logged metrics/RNG/replay but non-identical model and Adam tensors. The first run remains failed under the original exact comparison. Full tensor differences are retained privately; no tolerance was relaxed.

A new engineering-only entry enables deterministic Torch backward algorithms and an explicit CUBLAS_WORKSPACE_CONFIG, records/asserts that numerical backend contract, and fsyncs the published pointer directory. A paired short rerun is pending; this is not a formal FB restart or reward/boundary change. Old Issue56 lost512-update weights remain unrecoverable regardless of the new result.
