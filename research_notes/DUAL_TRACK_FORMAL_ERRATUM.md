# Issue56 preparation errata — original records retained

|Attempt/runtime|Observed defect|Impact and correction|
|---|---|---|
|stage-smoke-01 /546de08b2e058055a9a1d0e0ac08fa9176d4a1cd|New deferred-reset wrapper did not accept original `target_states` keyword; first Task.reset raised TypeError.|Before new source execution or training.0 formal samples/updates. Failed job104.4608s and ERRATUM/log retained. Wrapper now forwards `*args, **kwargs`; regression is a complete reset/source/terminal/replay smoke. No old experiment or published behavioral result affected.|
|stage-smoke-02 /68d4f569cc5c64a210d125307eefa54ae76fee24|After native-source and injected actual-terminal checks, the test compared a CUDA terminal tensor with its CPU replay copy directly.|A test-device mismatch, not a changed physical successor.0 formal samples/updates.125.2848s charged. Compare both on CPU; no equality tolerance changed. Original failure retained.|

No probability assertion, physics threshold, source split or scientific quality
gate has been relaxed to bypass the error. This file does not overwrite #55's
separate historical engineering erratum.

## New FB full-package validation failures

|Attempt/runtime|Observed defect|Impact and regression|
|---|---|---|
|fb-online-smoke-01 /658f69f1d36e8da5cc952be635323290eace9a2f|The loader constructed a second Collector/Task on the already-used simulator, stacking reset/callback bindings. Original strict reset-prefix comparison failed (max qpos0.132094, qvel12.2291).|Disposable engineering only; no formal model or prior paper result affected.576.5885s charged. Reuse the same Task graph for in-process readback; standalone load constructs one Task in a fresh process. Full actual-action prefix regression passed in fb-online-smoke-03; no equality tolerance changed.|
|fb-online-smoke-02 /cbbc65427bb11499eb98a3a2e83b3d62bdc184d4|Strict JSON writer rejected original Gym Box metadata containing -Infinity while packaging original init kwargs.|Not a nonfinite reward, gradient or policy-probability escape.260.5964s charged. Copy the archived init-kwargs bytes verbatim; retain strict finite JSON checks for measurements. Full save/readback passed in fb-online-smoke-03. Partial package is not a resumable or formal checkpoint.|

Successful full-package regression: runtime
`b9862178a6355043a19a809b08c27c5aa6a2fe94`,262.3471s total job.
256 actual native online controls;2 eager+2 compiled full six-optimizer updates;
model, all six Adam states and new replay read back exactly; actual29D-action
physical prefix reconstructs full solver/BFM/force/action/z history exactly.
These updates are discarded engineering weights, not the first4 formal updates.
All failed outputs and original logs remain in isolated private stage directories.

## Final pre-launch checks

- Course-provenance-01 failed due quoting in the disposable shell harness before
  any generator/physics call;0.8795s charged. Replaced with a committed test file.
- Course-provenance-02 assumed stopping must release later than continuation.
  That is not a required invariant: late deceleration can outlast a short finite
  continuation. Test now requires identical pre-branch parameters and equal
  post-motion hold, with unchanged approved deceleration ranges.27.6612s charged;
  course-provenance-03 passed in27.2666s. No probability or physics assertion relaxed.
- Code review found the legacy executor marks wrapped timeouts as a native
  termination. New stage entry preserves them as truncations, with real successor
  and no fall tail. Actual injected wrapped-timeout regression passed in
  stage-smoke-04 (147b22b684b0c03a613559fa6d5e5881d380ab7f,133.9472s).
  No formal samples existed before this correction. Native time limit45s exceeds
  this course; no evidence currently shows a historical result triggered it.
