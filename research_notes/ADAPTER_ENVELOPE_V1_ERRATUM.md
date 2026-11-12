# Envelope V1 qualification harness erratum

Runtime e4c2d3ff08fc0a29152dc7b1f0ee00201b077012, first B19sample job, exit1. The full600tick physics branch and150 exact execution/replay/qPD checks finished and trace was saved; the disposable update audit then attempted backward under the enclosing physical driver no_grad. This is a harness defect, not a model/replay/optimizer restoration mismatch. No completed new optimizer step.

Minimal source correction: torch.enable_grad around test.update; persist explicitly physical-only summary before attempting audit. Syntax checked only; full update replay not rerun because actual slip qualification failed independently. Do not call the engineering correction closed/passed. Original raw trace, failed stack, sidecar and e4c2d3f retained. New posthoc partial-evidence JSON is labeled reconstructed; original job not relabeled success.

Impact: actual measured physics and passed exactaction/replay/qPD assertions remain usable; target/bootstrap/gradient qualification missing, no B/C retraining authorized by this run. Gate false for loadedslip peak and p95; mean/otherstates not run. The correction changes no sampler/reward/Envelope constants/physical gate. Future use needs source relock plus full test under new approved scope.
