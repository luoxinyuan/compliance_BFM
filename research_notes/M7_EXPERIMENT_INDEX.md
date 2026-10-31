# M7 experiment index

Research record: https://github.com/GaoE05/robot-research-bridge/issues/28 (not queue).
Source: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/m7-continuous/repo/research_notes
Run: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m7-continuous-20261101A
Branch: gaoe05/m7-continuous, base2d3ed08b.

Protocol m7_protocol.json contains immutable task definitions and presealed final combinations. compute-ledger.json records runtime revisions, commands, exact owned PID/start ticks/resources/failures. Prior stage protocols copied to each attempt outputs.

qualification: 10 full episodes using original expert and frozenM6, runtime c271e165. Original teacher fails reverse task (alpha retained/reentered); neither model retrained.
expert: 2 episodes, runtime1b612f2d. M6 progress-only native+original local feedback; effective but loses learned residual, not chosen as final teacher. All evidence retained.
expert_refined: bounded teacher acquisition, runtime d98bd04. Full frozenM6Alignment executable command+original bounded privileged feedback. Projection remains native-progress-centered. No increase in joint authority, original M6 teacher task/limit structure preserved. Two matching reverse-aware signed-force rule baselines included, thresholds/rate symmetric with original positive rule, fixed before tests. This is explicit algorithm development, not an engineering bug.
projected: same frozenM6Alignment physical prefix until reverse restart, then teacher vs privileged per-state projected latent, no state reset. Checks same-state takeover, causal history, actual output equivalence and physical follow-through; temporary optimization is not deployment Student.
train0/collect1/train1/collect2/train2: conditional on teacher/action qualification, fixed 2500/2000/2000 per method, two real visited-state rounds, original17837 M6 examples retained. Shared samplers40%transition/20%positive/20%negative/20%free. 4 episodes/aggregation: transitions short/long, each bothmethods; original old-skill/negative data retained. Finalround2 fixed. M6 own final model initializations preserved, not identical initial weights between methods.
final: presealed two combinations + oldpositive/negative/smallpositive/free/pulse regressions and one explicitforce-history ablation retaining actual physical contact. No selection/retraining on final.

Launch: source project env.sh; python research_notes/m7_supervisor.py --gpu-index <live available> --stage <stage> --execute-after-owner-gpu-notice
Analysis: python research_notes/m7_analysis.py [--stage stage]; saved physics only. Future reproduction must use new isolated output root; never replay completed stages into canonicalRUN.

FIBER Appendix B/C distinction documented in M7_RESEARCH_STATE.md. This is frozenBFMZero/smallMLP LPP, not AH-BFM/FlowPlanner/SF-adapter replication.

Engineering failure retained: train2-attempt-01 exited before projection/update because concurrent supervisor ledger truncate/write exposed an empty JSON to done(). No extra optimizer updates. m7_io atomic same-directory replace plus concurrent-reader regression fixes the IO race; retry uses the same completed collect2, same prior train1, fixedseed/budget. Original failed directory/ledger record retained.

train2-attempt-02 completed at0f4a4b8: samefixed2000/method; totalnew6500/method, no update in failedattempt01. Final checkpoint hashes pinned in protocol before any sealed execution. Final shadow expert labels are diagnostic only, not replayed into training.

Post-final mechanism clarification (no training/selection): longer-gap reverse has only16–20% actualFx<−.1N and positive holdFx. One same-prefix M7-Pz physical partner-hold counterfactual is fixed after canonical final, preserving oldtest and finalSHA. Holds +14cm anchor fromrestart12.15 throughrelease17.35; same prefix, gap,50Hz controller and8s recovery. No fabricatedforce. Distinguishes demand-sensitive response from obligatory unload/retraction. Included in original5400s budget; not blindtest.

COMPLETE: 54fullphysical/1124.40recordedseconds,1286.40withprefix;12supervisedstarts,one pre-updateledgerIO failure;4076.37s budgetused. formal final27episodes@30f8d371, posthocpartnerhold1@a6312c72; no more training or selection. Derivedall-summary54,traceaudit54,feedbackaudit27,exactcounterfactualprefix and2same-stateprojectiontakeovers. m7_results.md is the final scientific entrypoint. Seven tests plus real causal/weights/source audits; oldM6 andall failures preserved. Nextstage proposal only, not authorized by this completion.
