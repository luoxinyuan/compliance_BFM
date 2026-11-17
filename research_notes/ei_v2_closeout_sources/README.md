# E/I v2 closeout analysis sources

Exact versions of bounded CPU archive analysis, including failed versions. These are not a new training framework.

- `final_readonly.py`: first pass; 100 case/stage/continuity/label tables were saved, then failed while trying to read an auxiliary update as an Actor line-search record. Original output and failure retained.
- `final_training.py`: training-only corrected reader; separates `auxiliary` from Actor attempts. Reads exact original manifests, cores and actual on-policy data segments.
- `final_stage_quality.py`: failed before data analysis because the scene XML uses includes; empty joint list was rejected.
- `final_stage_quality2.py`: uses the original MuJoCo model without dynamics to verify all29 qpos addresses; failed on a case-index variable in the label audit.
- `final_stage_quality3.py`: completed100 cases/445 phase-Goal windows, same locked continuity rules. Probe eligibility uses actual current measured observation force, not next physics force.
- `final_probe.py`: frozen causal predictive-context diagnosis, ancestor-grouped linear classifier only; no future-label encoding as input, no policy/representation update. Selective frozen-module decoder cross-checked against original complete-core reader on an archived real fixture; does not claim physical/Adam training recovery.
- `final_curves.py`: original per-block optimization records and actual command coverage only.
- `build_report.py`: local Markdown/aggregate JSON assembler. Local `data/` is copied from the above immutable summaries; no checkpoints or trajectories enter Git.

## Placement / reproduction

The remote scripts were copied to the existing formal stage's `prep/`, so `Path(__file__).parents[1]` resolves the stage and imports its frozen `eval-runtime`. Run under the already-validated project venv, CPU-only (`CUDA_VISIBLE_DEVICES=""`, one OMP/BLAS thread), into a new separately versioned output directory; do not overwrite any archived directory. Every actual run's source SHA and cost is in the final machine-readable report and original stage sidecar. Failed variants are evidence, not recommended entrypoints.

The original training/evaluation worktrees, checkpoint chain, probability/physics contracts, public Goal, reward, splits and sealed set were not changed. All diagnosis states are already exposed Dev or original training states.

The metadata-only `coverage-literal-audit` additionally reads the final core's literal episode/instruction records (not a physical reset) to confirm44 unique completed specifications and actual Goal-transition parent coverage. Its exact result remains in the original archive and final report; no new sampling.
