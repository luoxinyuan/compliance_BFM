# M9 experiment index — bounded milestone finished (partial qualification)

Private source: `gaoe05/m9-contact-locomotion`, inherited M8 commit `554324f22ebedcee3498257c4348434c5d937d59`.
Run root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9-contact-locomotion-20261101A`.
Stage Issue: https://github.com/GaoE05/robot-research-bridge/issues/30

## Reproduction and evidence

`m9_protocol.json` records resources, assets, fixed candidate batch, training and sealed physical conditions. `compute-ledger.json` records source commit, exact command, GPU UUID, processes, wall time, failures and outputs for every launch. Each outputs directory retains `result.json`, compressed 200 Hz physics / 50 Hz control traces, initial state and same-state expert labels. `m9_analysis.py` and `m9_report.py` analyze saved traces without models or new rollout. A finished stage is not replayed by the supervisor.

Manual stage command (requires live resource check, no watch/Bridge):
`python3 research_notes/m9_supervisor.py --stage STAGE --gpu-index AVAILABLE_GPU --execute-after-owner-gpu-notice`

## Experiments and non-selection

1. `infer`: original same-weight replay, exact inherited sample indices/seed; reward family, not name-based skill certification.
2. `qualification`: batch reward entry/execute/exit; all candidate failures and semantics retained.
3. `tracking`: three geometry-selected LaFAN reference windows, exact official backward observation / model encoding.
4. `contact`: finite native reference with real curved Tether; failed first launch was a stale M8 attempt path, corrected from ledger, no physical evidence overwritten.
5. `expert`: kinematically closed cycles; force-only and privileged-current-guide feedback. Cyclic expert removes reference exhaustion as an apparent learned stop.
6. `diagnostic`: one fixed +/- .03 native turn-z composition; retained drift and large qPD change.
7. `turncheck`: predicted .06 positive composition and common-phase feedback. Native left turn exists; feedback left-turn qualification remains distinct.
8. `train0`: 4000 updates, last checkpoint fixed; native legal-z/action identity verified. Old M8 executed labels retained; no identity optimization disguised as projection.
9. `collect1`: seven real Student trajectories with same-state queries. All complete. Straight/right dev move qPD disagreement with expert 6.819/7.688deg; hold 11.504/11.268deg despite offline ~.6deg. Genuine visited-state coverage needed; no final-test selection.
10. `speedcheck`: single sustained .25m/s profile, original5N/K80D8 and unchanged controller composition. Tests operating range, not isolated causal effect of speed (path/profile also differ). No gain scan. Result: sustained left still negative heading, high5N saturation; not added as a qualified turn expert or training source.

All native and expert behavior, learned behavior and held-out evidence will be reported separately. Finite reference phase, cycle phase and feedback memory are expert command state, not hidden Student inputs. Student deployment sees only sensor provenance, force/proprio/action history. No hidden anchor, future reference, time or task answer. Negative control includes an explicit all-zero-force-history execution permit, so it is not independent evidence that the MLP learned every no-contact case.

Known support limitations: per-sample link/frame identity tested on real saved sensor streams; same-episode contact-link switching not yet physically qualified. Rigid initial-yaw final condition is separately recorded and refreshed by actual BFM controls, not a new learned posture. Shared source, PD, assets and M8 models remain unchanged.

11. `train1`: 2000 updates on 12346 accumulated samples; fixed last checkpoint `4974f8e71205d65946401f66fab3219397ac038c5e2b9406c449e3eddb9596d5`.
12. `collect2`: seven further complete real Student trajectories. Right development root Y changed from +.129 to -.099m; move error .2125 to .1618m. Straight move error .1695 to .1960m and hold .1807 to .2596m: regression retained, not a reason to select train0 on final data.
13. `train2`: predetermined2000 updates; final checkpoint pinned before opening final conditions: `f003afd8953b55e8f96ae1f47aea2beb80622f257cef24c451ea34fe28ab48a6` (18094 accumulated samples,8000 total updates). No extra gains, teacher variants or high-speed data introduced after speedcheck failure.

14. `final`: 30/30 complete once under runtime `decf2929cbb87bb81ddcdf0db1d1d91237d3b90d`; full findings in `m9_results.md`. No further training after sealed evaluation. Original qualification failures remain in the same index. 83 complete physical trajectories including semantic/quality negatives; 1312.34 simulated seconds; one pre-physics model-run path failure preserved. Supervised compute4043.55/7200s.
15. Saved-trace analyses: `m9_report.py`, `m9_final_audit.py`, `m9_plot.py`; outputs `m9-results.json`, `final-audit.json`, per-stage capability metrics and four full trajectory plots. Only plot labeling/layout adjusted after final; no model or physics changes.

Final model `train2-attempt-01/outputs/projected-z-final.pt`, SHA `f003afd8953b55e8f96ae1f47aea2beb80622f257cef24c451ea34fe28ab48a6`; frozen M8-PZ dependency retained. Final pairing, native source/weight, no-force, contract and cleanup audit passed. Runtime has exited. Do not automatically start another round, overwrite old evidence, push, or replay consumed runs.
