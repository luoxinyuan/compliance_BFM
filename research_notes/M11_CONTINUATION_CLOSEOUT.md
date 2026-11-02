# M11 continuation — owner-requested early closeout

2026-11-02 16:17 CST. **Development evidence only; no completed equal-budget final A/B comparison.**

Owner requested termination after persistent quality problems. Fixed A completed 160 batches; annealed B stopped safely after 114. The owned trainer saved an interrupted checkpoint and exited with all owned workers. GPU1 was released; no other task was stopped. The sequence and local finite video waiter both report stopped. No sealed final evaluation or new video rendering ran. Original M11 and all continuation attempts remain intact.

## Budget and identity

|Arm|Completed batches|Added RL decisions|Completed logged episodes|Logged physics / control steps|Last actual std|
|---|---:|---:|---:|---|---:|
|fixed|160|122880|788|2451480 / 612870|0.080000|
|annealed|114|87552|562|1746652 / 436663|0.022038|

Physics/control totals are completed logged episode sums, not counts of optimizer updates or all in-flight work. A/B added optimizer updates: 2236 / 1100. Total supervised continuation wall time including attempts and evaluations: 19,204.39 seconds (5.33 hours). B stopped before its final low-std hold. A had an authorized throughput restart at batch114; preserve the recorded worker-reset protocol rather than claiming bitwise continuation.

Runtime: `c98abd7912e173f1228afaca996ae43ed6009ac1`; earlier A runtime: `38c7b25dcb6a3a1124acd9cf44c4eb116364ac60`. Same original model/optimizer/RNG initialization, frozen BFM/PD/M10 prior, task distribution and reward. No Student trained in this continuation.

## All completed mean-execution development evaluations

Each cell is 3D tracking error cm / actual force RMS N. Exposed straight was already a known failure condition, not a blind test. Selection uses the predeclared mean-execution failure/tail-aware rule; lower score is better. No final-test model selection occurred.

|Candidate|Selection score|Straight|Exposed straight|Left|Right|
|---|---:|---|---|---|---|
|fixed dev-0000|1.5076|7.829 / 4.143|37.731 / 4.588|10.115 / 4.594|8.566 / 4.658|
|fixed dev-0040|1.4344|9.620 / 4.673|10.455 / 4.336|8.671 / 4.324|9.761 / 4.676|
|fixed dev-0080|1.5251|7.546 / 4.548|34.900 / 4.582|9.804 / 4.664|8.832 / 4.644|
|fixed dev-0120|1.5473|11.369 / 4.655|34.359 / 4.587|11.849 / 4.679|7.659 / 4.569|
|fixed dev-0160|1.5128|7.412 / 4.116|42.720 / 4.577|10.197 / 4.614|10.658 / 4.674|
|annealed dev-0000|1.5076|7.829 / 4.143|37.731 / 4.588|10.115 / 4.594|8.566 / 4.658|
|annealed dev-0040|1.4785|9.875 / 4.679|13.416 / 4.590|9.098 / 4.557|9.833 / 4.657|
|annealed dev-0080|1.5311|8.471 / 4.641|31.272 / 4.585|12.196 / 4.676|9.058 / 4.604|

Matched M10 baseline (straight / exposed straight / left / right): 12.662/4.610, 9.188/4.342, 9.509/4.605, 10.986/4.673. These differ from the original M11 sealed-test conditions; do not merge their numbers.

Both selected checkpoints remain batch40. A exposed-straight error temporarily fell from37.731 to10.455cm, then rose to42.720cm at160. B reached13.416cm at40, then31.272cm at80; batch114 has no development evaluation. Force stayed about4.3–4.7N in these conditions. Benefits are condition-dependent and do not establish a reliably improved low-force Expert. B cannot be ranked against A160 at equal budget. No convergence or universal failure of annealing is established.

Full mean and stochastic evaluations, local/no-contact/heading/recovery regressions, saturation, boundaries and episode metrics remain in the server JSON index below. Aggregate scores are not substitutes for task-specific performance.

## Scientific boundary and next decision

Persistent deterministic deployment quality and task tradeoffs remain the first practical failure boundary. Earlier zero-mean residual + matched noise controls showed learned feedback contributed to tracking, but not a corresponding force advantage. Previous audits did not identify a force/GAE/raw-logprob hard error explaining this failure. This is not evidence that BFM or LPP is fundamentally incapable, nor proof that reward weight alone is the cause.

Recommended next step (NOT started): narrow to one trusted straight-motion/contact feedback domain; test whether privileged residual learning can reproducibly improve deterministic execution and actual XYZ force over M10, then expand speed/distance. Keep real closed-loop feedback rather than fit a timed replay. Reassess representation/exploration/optimization with this reduced task; do not launch another broad run on the assumption that more steps will fix it.

Deployment caveat: current Student force comes from simulated applied Tether force with known contact metadata. This excludes anchor/future targets from Student input, but does not prove real G1 force sensing or contact localization. A sensor/estimator and noise/delay validation remain required before real deployment.

## Preserved artifacts

Root `R=/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-continuation-20261102`.

- `owner-early-stop.json`, `sequence-status.json`, `compute-ledger.json`, `owner-closeout/summary.json` preserve stop reason, runtime and full metric index.
- A: `fixed-attempt-03/outputs/job/`; B: `annealed-attempt-01/outputs/job/`. Earlier attempts are unchanged.

|Arm|Snapshot|SHA256|
|---|---|---|
|fixed|last.pt|`ab2cec712754508940de82a69267dfed5f4bbb8a0ab8249bed625083b2abddb1`|
|fixed|latest.pt|`80df6ab136e37b555dd31e0d823b644d7cd77981291ddbaf081931837dd6b830`|
|fixed|best.pt|`e979f44d60f16abfe3c3644d64eb410717c1717a3f0a4ebb879843f1313da601`|
|annealed|interrupted.pt|`aca42857e291f023a52cdf90ce6d5a83bc4c2417365c3989c0856ad5e39d514f`|
|annealed|latest.pt|`51ac69884698deb03c8e706225dff5e5631887390ffe6674a8139c013565313b`|
|annealed|best.pt|`9c4196065365cfd5219caf6308e97ed21e0852eb9d99616bb511598bf5d84a10`|

Original M11 best initialization SHA256: `bc305de6917103c5fbc64a6c6639f91bb517d3c7a48d689e7158739a6168cc8f`.

No new continuation videos delivered: final rendering was intentionally not started after owner stop. Existing M10 and first-round M11 videos remain at `~/Projects/compliance-bfm-diagnostics/videos/m10-task-appropriate/` and `~/Projects/compliance-bfm-diagnostics/videos/m11-privileged-expert/`. Local compact evidence: `~/Projects/compliance-bfm-diagnostics/m11-privileged-expert/continuation/closeout/summary.json`. No weights or raw trajectories are included in Git.

This closes the current run. No new training, final evaluation, Bridge/watch, or resource expansion is queued.
