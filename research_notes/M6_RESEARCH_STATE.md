# M6 Multi-Behavior Contact LPP

Authorized independent milestone; Issue27. M5 formally closed.

Candidate: +X15cm causal forward native-goal plus bounded arm feedback versus -X3.5cm standing-centered backward arm feedback, same robot initial state, right wrist, K80/D8/cap5. Original expert algorithm unchanged and queryable; native progress uses actual force rather than task identity. Different motion and response demand, not timing variants. Need real teacher and online latent-projected physics before shared training.

Use existing native-path+tangent interface only if second task physically expressible. No taskID, direction label, anchor or expert mode in Student. FIBER-inspired action projection, real visited-state queries and shared learning; not replication of AH-BFM or flow planner.

Budget:5400 cumulative compute seconds,8CPU,32GiB RSS,8GiB ownedVRAM; serial GPU4 initially free. No downloads/shared changes/push/Bridge. Negative expert qualification first; then fixed two aggregation rounds and final round2, independent negative control matched capacity/updates, M5 independent positive retained. New test_pos/test_neg sealed in protocol before first simulation. Historical M5 tests used only archived regression, not relabeled new.

Failure and progress evidence: compute-ledger.json, each attempt logs/outputs. Source inherited unchanged; M6 added files only.
