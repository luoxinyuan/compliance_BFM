# M5 experiment index

Stage Issue26; M0–M4 retained. Personal branch gaoe05/m5-contact-lpp.
Code: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/m5-contact-lpp/repo
Data: /mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m5-contact-lpp-20261031A
Mac mirror: ~/Projects/compliance-bfm-diagnostics/m5-contact-lpp

## Version ledger
- Base f538511: completed M4, no inherited source modifications in M5.
- 5967a73: causal task expert, deployable257-output projector, two-round protocol, tests. qualification-attempt-01 completed6episodes.
- 02b80f0: common no-force anchor reference for Student/evaluation; final Student has no expert/Jacobian queries. train0 completed3000updates per method.
- collect1-attempt-01 retained failure at t3 first Student forward: CPU Linear weights versusCUDA input; no contact experiment completed, no scientific result claimed.
- 3f2f062: place entire loaded Projector onrequesteddevice. collect1-attempt-02 complete, train1 +2000 each, collect2 complete, train2 +2000 each. Same training/physical definition, nocheckpoint reselection.
- 73613b4: frozen round2 SHAs, final main25 plus2 precommitted residual-OFF branches. Analysis/audit/plots only added. No new fit.

## Evidence
- compute-ledger.json: every attempt, runtimecommit, exactownedPIDs, resources and exits.
- qualification-attempt-01/outputs: teacher/rule/native physical contrast, teacher late/free/pulse.
- train0/1/2-attempt-01/outputs: new-data-projection.json, cumulativedataset.pt, training.json, alignment-final.pt,distill-final.pt. Final6825realstateexamples. 3000+2000+2000 perpolicy, BFM frozen.
- collect1-attempt-02 andcollect2-attempt-01: each policy early/late/free truephysics +privilegedexpertqueries onactualvisitedstates.
- final-attempt-01/outputs: sealedcontact combinations, regression/negative/pulse, residual-OFF physicalablation, sourcecontract andfrozenmodelchecks.
- analysis/physical-summary.json, final-audit.json, m5-final-curves.png: derivedoutputs; rawtraces remainauthoritative.

## Reproduction
Use the existing project env.sh and original fixed assets. Exactcommands storedperattempt. The staged sequence isqualification,train0,collect1,train1,collect2,train2,final; supervisor runs CUDA BFM, records/caps resources and neverrepeats acompletedstage. For independentreplication, use aNEW isolated run directory and sourcecopy with RUN adjusted; do not overwrite thisledger or historicaldatasets. Analysis itself is read-only and repeatable: python research_notes/m5_analysis.py; python research_notes/m5_plot.py.

The final27 tests are not trainingdata. The two new conditions were fixed beforefirst modelrun; two samecheckpoint residualOFF diagnostics were appended beforefinal exposure, without altering either maincondition orweights. Papercomparison ismethodinspiration, not exactreproduction: qPD uses originalratio5 (not paperdeployment'sratio6), original50Hz interface, nativeprojectedz prior rather than B(s), smallMLP notflowplanner.
