# M4 Continuous Contact Feedback — stage record

User supersedes midpoint-centered M4. Stage Issue #25. Base M3 72851904. Independent private branch gaoe05/m4-continuous-feedback; inherited files stay unchanged.

## Scientific contract

Primary task is one continuous forward-guidance, body-response, stop/recovery episode, not a sweep of interpolation labels. Natural u=.25 is a fixed existing preparation; main task +X15cm /4s ramp /3s hold /8s full recovery. Old u=.5 is preserved as a matched-state failed-policy regression, not the stage objective. Teacher, pose-base local teacher, original deployable rule, native and frozen M1/M3 are retained. No new controller, bounds, network or memory has been added.

The three qualifications are real teacher feedback, deployment information/maintainable intent, and physical teacher/latent-projected recovery. Same-observation label comparisons use global z and action/q_PD, never raw redundant coordinates. Exact initial context can matter without being leakage; counterfactual context swaps do not prove conflicting training labels.

All three must justify learning before the conditional fixed two rounds x750 Projected-z updates. The final model would be round2, never a heldout-selected checkpoint. No training has been launched. If expertise is only scheduled paths or cannot recover, stop distillation and describe the missing feedback expert.

## Precommitted resources

3600s cumulative model/physics/training wall,8CPU,RSS32GiB,ownedVRAM8GiB; expected output<=3GiB in personal project. CUDA BFM only, original CPU MuJoCo; no CPU-model fallback. Any resource-safe GPU may be selected, exact owned process cleanup only. Existing safety boundaries retained, no additional per-pose tracking threshold or arbitrary mandatory stepping.

## Versions / failures

- e315538: fixed protocol, source/label audit, matched-state recovery runner, 4 protocol tests.
- f98cb3a: saved physical analysis and analytic exact-zero-gate identity; active projection still original160steps at50Hz. No control change; dead zero-gate optimization omitted with identity guaranteed by original constrained().
- Local prelaunch syntax check found Python reserved keyword `global` used as a dict keyword. Renamed `global_command`;4tests passed before any launch. No failed model/physical launch for this mistake.

Runtime ledgers and complete evidence: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m4-continuous-feedback-20261031A/ . Original M1/M2/M3 results unchanged. Status: completed at expert-semantic capability boundary.16 real branches/426.4 simulatedseconds includingprefixes, three same-state recoveries via teacher and projectedlatent succeed. No Studentupdates. Main frozenM3 has severe unforced reconfiguration. Highlevel teacher still uses initialconfiguration/clock/mode; no unified reachability-aware objective feedback yet. See m4_results.md. Stop; no further experiments authorized by this file.
