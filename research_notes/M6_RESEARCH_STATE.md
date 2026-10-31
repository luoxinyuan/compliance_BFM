# M6 Multi-Behavior Contact LPP

Authorized independent milestone; Issue27. M5 formally closed.

Candidate: +X15cm causal forward native-goal plus bounded arm feedback versus -X3.5cm standing-centered backward arm feedback, same robot initial state, right wrist, K80/D8/cap5. Original expert algorithm unchanged and queryable; native progress uses actual force rather than task identity. Different motion and response demand, not timing variants. Need real teacher and online latent-projected physics before shared training.

Use existing native-path+tangent interface only if second task physically expressible. No taskID, direction label, anchor or expert mode in Student. FIBER-inspired action projection, real visited-state queries and shared learning; not replication of AH-BFM or flow planner.

Budget:5400 cumulative compute seconds,8CPU,32GiB RSS,8GiB ownedVRAM; serial GPU4 initially free. No downloads/shared changes/push/Bridge. Negative expert qualification first; then fixed two aggregation rounds and final round2, independent negative control matched capacity/updates, M5 independent positive retained. New test_pos/test_neg sealed in protocol before first simulation. Historical M5 tests used only archived regression, not relabeled new.

Failure and progress evidence: compute-ledger.json, each attempt logs/outputs. Source inherited unchanged; M6 added files only.

Qualification complete: native/negative teacher/projected negative full-contact RMS1.65148/1.14107/1.08221cm; force1.34739/.93168/.88307N; two teacher onsets consistent. Projected per-frame arm RMSmax.000578deg/nonarm.035398deg. Teacher wrist-root extraX relative native -1.04025cm, native progress0. Same interface can express negative arm response; proceed fixed training.

Initial policy for all new methods is frozen M5 final Projected-z (7000 inherited updates); each M6 method receives7000 additional updates. Main shared methods same available data and balance, independent negative-only control excludes positive updates by design. No target label is an input. Final new tests remain unopened.

Training fixed round2 complete:7531 initial samples (6825M5+706negative); two actual13-episode Student state-query rounds add5153 each; final17837, each method3000+2000+2000 new updates. No test data. All training runtime b9e165a. Final source adds only presealed evaluation baselines and audits, not model/training changes.

Signed-force rule fairness baseline calibrated from train0 negative projected labels only. No anchor, taskID or direct joint control, one fixed template scaled by measured force. No tuning from final outcomes.

Engineering recovery: final-attempt01 six completed episodes preserved; onset6.65 incompatible with inherited50Hz-only anchor lookup. Fix exact200Hz physical-state reference with parity atold5/6.2/7.1s, unchanged task/weights. No re-training or repeat of completed evaluations. Old failure traceback retained; original handler did not save partial interrupted trajectory. Future exception capture added.

## Final milestone decision

M6 completed. Two shared final policies,7000 new updates each,17837 aggregated samples,69 complete new physical episodes/1297.88 simulated seconds. Sealed +14/-3cm Alignment error2.682/.821cm versus native8.275/1.491; Pz2.722/.914. Actual-force ablation Pz7.680/1.386. Small+3cm shared regresses5.78/7.51% versusM5 and has larger release transients; all complete branches stable and no-contact bitwise native. Not unseen-body or within-episode transition proof.

Full evidence in m6_results.md and Issue27. Runtime training b9e165a; final resume c077fad. All model/physics work stopped; no push. Unique next proposal: real within-episode forward-unload-reverse contact, qualify expert at actual transfer states, then shared latent aggregation; preserve small+X quality regression, preseal new timings. Do not start until next milestone authorization.
