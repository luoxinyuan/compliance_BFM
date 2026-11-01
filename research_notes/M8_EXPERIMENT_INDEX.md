# M8 experiment index

Scope: new upper-torso downward maintained-contact behavior plus old right-wrist LPP. Frozen same-weight BFM and original PD. This is not unilateral surface pressing or locomotion qualification.

Canonical code: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/m8-contact-body/repo/research_notes`.
Canonical artifacts: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m8-contact-body-20261101A`.

`compute-ledger.json` is the complete launch/PID/source/GPU/resource record; no Bridge task consumption.

- `asset-audit.json`: initial purely geometric ranking, retained. Broad stance and original motion speed motivated a static-entry refinement before any BFM evaluation.
- `asset-audit-static.json`: one bounded stride10 scan of40existing motions, zero latent search. Two distinct natural candidates fixed before physics.
- `qualification-attempt-01`: source SyntaxError before model load,0GPU/0physics/0updates; preserved.
- `qualification-attempt-02`: identical standing prefix; standing14s, candidate0partial4.03s at conservative pelvis20deg review threshold (not a fall), candidate1full14s. Candidate1dance1_subject2/frame1690 selected for subsequent contact, original goal inference norm16/82.14deg fromstanding. No live qpos goal reset.
- `contact-attempt-01`: downward8cm maintained upper-torso guidance: standing, raw proportional rule, new signed-force integral native feedback expert; expertfree. WorldXYZforce and actual contact wrench logged. Raw-rule release15.81deg qPD jump retained; final fair comparator applies SAME native-progress entry/return rate bounds as expert, not a joint smoother.
- `expert-attempt-01`:10cm feedback expert plus8/10cm actual legal projection physics. Online projected and expert qpos/qvel/force/point traces exactly equal. Body expert is already native-latent feedback, so this does NOT establish arbitrary non-native expert action representability.
- `train0-attempt-01`: both methods3000updates, identical widened M7PZ initialization, original23242labels plus3newbodyepisodes. Causal contact-site encoding applied before training; earlier teacher raw input logs retained, only sensor encoding reconstructed, no physical targets changed.
- `collect1-attempt-01`: both real Students,8/10cm torso and original wrist continuous reversal,6completeepisodes; actual50Hz expert queries, alternateframes fortraining.
- `train1-attempt-01`: 2000 updates/method,27053 cumulative samples. `collect2-attempt-01`:6 complete real Student episodes. `train2-attempt-01`:2000 updates/method,29689 samples. Both rounds completed, finalround2 fixed before testing; no held-out feedback into optimization.
- `final`: pre-sealed torso9cm/3.5s ramp/onset6.6 and7cm/2.8s/onset5.7 with localX+2cm; oldwrist+15/-3.5/+3 andM7sealed_mid are REGRESSIONS. Full8s release, no-contact bothsensorcontexts. No finaldata enters training or modelselection.

`m8_protocol.json` fixes conditions, budgets, training, final checkpoint identity beforefinalexecution. `analysis` contains traceaudit, physical comparisons, plots and finalprovenance. OriginalM1–M7andallfailedcandidates remain unchanged. The M7 shared native forward path is a retained skill, not the new research goal.

Final status: COMPLETED; no further experiments authorized by this completed M8 run.

- `final-attempt-01`:28/28 complete physical tests, source `f2511b2ae9f89c5c9dd0c2379c11caf404c3f687`; one complete run, no test rerun/model selection. New body tests and old wrist regressions are separated in `m8_results.md`.
- `analysis/final-comparison.json`:all phase metrics and paired body response/no-contact parity. `behavior-audit.json`:post-test saved-trace diagnosis only; release and residual-gate jumps retained.
- `analysis/trace-audit.json`:49complete/1partial,816.81recorded seconds,966.81including3s physical prefixes. Real wrench/force/history/location/norm and exact native body projection parity verified.
- `analysis/final-audit.json`:current archive identity, preserved sources/weights/oldcheckpoints, no surviving experiment PID. `tests-final.log`:6 tests passed.
- Actual supervised compute2579.9078s/7200s,10attempts including pre-load SyntaxError; peakGPU1206MiB/RSS7.3373GiB, approximately2GiB output. Offline analysis is CPU-only and separately logged.
- No shared checkout changes, push, installs/downloads, PPO, Bridge or background tasks. See `M8_RESEARCH_STATE.md` for the one proposed next milestone; it has not been started.
