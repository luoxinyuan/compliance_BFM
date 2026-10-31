# M3 closure and next-session state

Read `m3_results.md` and Issue24 final report first. M3 is complete and stopped; one1500-step extension, frozen evaluation, no additional fit authorized by this report.

- Genuine limited progress: unseen u=.625 keeps intermediate body before contact; no-contact wrist RMS1.396cm vs old15.076cm. Error3.147cm/native4.531, force2.578N/native3.690. Real no-contact-z replay error4.735cm proves contact-command task value. Complete8s return improved.
- Main failure: known training/devu=.5 real rollout error28.273cm/hold38.802cm, wristZ−37.913cm, late wronggoal. Originalbounded teacher3.750cm. Do not promote as reliable replacement for M1 or claim robustcontinuousgeneralization.
- Same trained model succeeds at u0/u1 and new.625 but fails.5: failure is a closed-loop state/history coverage issue, not evidence that two remembered endpoint poses are all the BFM can express. Exact cause/observability remains to be separated. No arbitraryoutput-u gate: u andreturnr are redundant coordinates.
- Source branch gaoe05/m3-continuous-body, base d07a616. Fit/final da69c597; causaldiagnostic7f5da11. SharedHEAD6023c6e andcoreunchanged. No push.
- Root data: bridge-v2-owned/runs/m3-continuous-body-20261031A; finalcheckpoint train-attempt-01/outputs/projected-z-final.pt SHA81300c1a1d4abcf25eb58a36a0a92924f92526771ac2c2bbbf9c8eb8f578cf81. OldM1SHA preserved. Saved0.625test is now exposed; never relabel as freshblindtest.
- Final records:35physicalbranches,1500newStudentupdates,0BFMupdates;9GPUstages2070.07s/GPU4,peak710MiB/RSS7.45GiB,allownedexited. Completefailedregression,source/protocolamendment andpost-finaldiagnostic included.
- Unique next proposal, NOT executed: finite continuous-state on-policy expert requery/aggregation eligibility using .5 asdevelopment failure. First check deployable-history/teacher-label consistency, preserve negatives/fullrelease, sealnewconfiguration+timing. No blindoffline data growth,force-rule patches,latent expansion orPPO.

- Last bounded failure-state check@15302b72:4x5 frames at known .5, all original-expression tests pass. Maxarm.01105degrees/nonarm.19236degrees versusStudent3.419–35.559degrees. No further physical orfit, owned523761 exited after97.26s. Allstage resources total2070.07s.
