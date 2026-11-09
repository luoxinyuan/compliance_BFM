# Issue56 preparation errata — original records retained

|Attempt/runtime|Observed defect|Impact and correction|
|---|---|---|
|stage-smoke-01 /546de08b2e058055a9a1d0e0ac08fa9176d4a1cd|New deferred-reset wrapper did not accept original `target_states` keyword; first Task.reset raised TypeError.|Before new source execution or training.0 formal samples/updates. Failed job104.4608s and ERRATUM/log retained. Wrapper now forwards `*args, **kwargs`; regression is a complete reset/source/terminal/replay smoke. No old experiment or published behavioral result affected.|
|stage-smoke-02 /68d4f569cc5c64a210d125307eefa54ae76fee24|After native-source and injected actual-terminal checks, the test compared a CUDA terminal tensor with its CPU replay copy directly.|A test-device mismatch, not a changed physical successor.0 formal samples/updates.125.2848s charged. Compare both on CPU; no equality tolerance changed. Original failure retained.|

No probability assertion, physics threshold, source split or scientific quality
gate has been relaxed to bypass the error. This file does not overwrite #55's
separate historical engineering erratum.
