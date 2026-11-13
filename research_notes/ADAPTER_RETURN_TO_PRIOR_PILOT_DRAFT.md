# B-only Adapter Return-to-Prior — review-only falsifiable pilot draft

**NOT APPROVED / NOT STARTED.** This is a new proposed learning experiment; no B/C continuation, no new Envelope or oldslip-rule revision. Issue62 V1qualification remains failed. No Student/PPO/fullFB/128roundrestart. Evidence and limits in ADAPTER_RETURN_TO_PRIOR_AUDIT.md.

## Question and intervention

Can a *same current state, same current legalz* training constraint remove harmful recovery residuals in an independently tested domain while retaining useful activeactions? Current evidence supports trying this narrow question, not declaring the entire repair route successful.

Two B-only clones, one seed, same frozen B19actor/adapterAdam as parent; exact originalEnvelopeV1, std.01/clip.06, original5/20/100ms controls, BFM/PD/Tether/reward/hardqallowance and physicalgates unchanged:

- B-control:64 adapter-only steps under original B actor objective, no new Returnterm.
- B-return:the same64steps, minibatches and optimizer initialization, plus fixed λ_R=.0008 Returnterm.

**Both arms freeze U, Q, discriminator and their targets**, plus originalBFM; only adapterAdam steps. This common intervention deliberately isolates adapterrecoverability and prevents U from compensating for the newloss. It is **not** a replication of full B jointRL or an on-policy learning claim. Existing Q is a fixed localsurrogate; the pilot's physicalevaluation can falsify it. A128/BC500 stay frozen references; no Atraining and no Ctraining. Any later full B/C recipe must share the exact newloss/data/masks/operator and separately regainqualification.

## Loss and fixed coefficient

For genuine recordedinputx and its correct realpreviousz/target interpolation:
`delta_raw = adapter_phi(x, stopgrad(z_current))`;
`L_R = mean_j[(PD_scale_j * delta_raw_j)^2] / (0.1rad)^2`.

Reference is originalBFM at the *same currentz/robotobs*, not a storednativez or standaloneTeacherhistory. Thisraw joint-equivalent correction is an auxiliarytrainingproxy; actualphysics/Q/replay still use originalV1 executedaction. Penalizing only postEnvelope difference would have zero local outputgradients on79.31%ofcomponents at the knownstate; rawproposal return preserves gradient without changinganyclip.

`L_total = original_adapter_actor_loss + .0008 * mean_validated_return_batch(L_R)`.

The extra term receives a separate32row parent-balanced batch, so its coefficient is not silently rescaled by currentbatch inactivefraction. Originalactorreplay batch32 andnativeCPR batch32 are identical acrossarms; Returnbatchindices also fixed acrossarms (controlcoefficient0). No runtime force-gatedadapter, no forcedallinactiveoutputs0.

Coefficient basis is pretrial read-only localgradientmeasurement, not search: at genuine11.8s input, projected taskgradientnorm .048558, rawreturngradientnorm120.862366;2×ratio=.000803526, roundedto **.0008** (weightedgradient≈1.99×task at thatpoint). CopyingnativeCPR.1 is unjustified. One state is not a globalcalibration; additionalqualifyingstates may expose unsuitable scaling. Log per-parent gradients and saturation but do not adjust λ afterdev; evidence against thisfixedrecipe ends thepilot rather than triggering a sweep.

## Data and parent separation

No newMPC/Teachersearch. Old immutableB19replay/native retained. Mainactorbatches use only the selectedtrainingparentIDs (8dde57ea,2670e363 plus reliable7f3a7da6 oldreplay); all listeddevparentIDs excluded from **new** replay/CPR sampling forbotharms. PretrainedB19 has seen these historicalgroups, so this is **exposed mechanisticdevelopment**, never newsealedgeneralization. Nativeprefixrecords must inherit their realparentgroup; do not random-split adjacentframes.

Returnlabels are permitted only on an actual fullstate from which frozen same-z base-only continuation has validatedsafe recovery, plus the base's **actually executed qualified successortrajectory**. Two trainingparents, mean/samplebranches count as the sameparent, notfourindependentgroups. No propagation of onetrajectory's recovery validity to every failedV1state. The exactB1911.8s off controls alreadyexist and can bereused afteridentitychecks; existingfullinput atitsfirststate isavailable. Otherlabels require the explicitly approved future qualification below. Currentaudit produces no newlabels.

Trainingmask/whitelist is trainingmetadata. Actor inputs remain deployproprio/realexecutedaction/lastz/knowncontactinstallation/idealforcehistory; no active truth, source/taskID, onset/releaseclock, anchor/partnertruth or futurepath is added. Return's samecurrentz uses realownhistory, no fictitious I/Teacher/MPCphase. Currentzero-force alone is not the whitelist.

### Draft complete-state manifest

| Role | Parent (independent group key) | State time | Existing archive round | Reconstruction / window |
|---|---|---:|---:|---|
| return_train | `8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e` | 11.8s | 19 | existing exact offsample/offmean3s may be reused; complete current input saved |
| return_train | `2670e363f8c204b64ab32dd614237f032a571e1cd03f4e5c50524c3caef42f79` | 10.0s | 17 | reconstruct prefix only to10.0 from archived actions; do not use10.7+3s beyond originalend |
| recovery_dev | `fea32998d2c1c42c5dad39c9e186641b29671f6e63f8973bcde5389186f77c93` | 11.9s | 12 | reconstruct prefix only to11.9;3s ends at original14.9 |
| recovery_dev | `eb77738422ceb4787fad231c3b9b9a7eec4d690ecfe6fdaaa21cacd31e63323d` | 9.7s | 13 | stop branch;3s ends12.7 beforeoriginalend13.4 |
| active_dev | `b69d7a23e0491199d8a8aa55d741a9878bc2b136b9d2c29b0a7c48a3527d916f` | 6.0s | 1 | near ongoingcontact3s |
| active_dev | `06c173441f3204e6b90e5e5fb230e9356f15e30e91495171915d5a4f6897bd70` | 6.0s | 3 | mid ongoingcontact3s |
| active_dev | `d52717d57a1504880c53e84b5707998e4f21614551ef369335ffe6d3eb2afa3b` | 7.0s | 5 | stop branch contact; stop/continue remainoneparent |
| none_dev | `6a2fe7552234ceb9dd5944b5601c46e3424d3995699b22bc02e6d37410dbc314` | 6.0s | 83 | archivedA83no-contact actualstate; true source/history preserved; not native-only afterhandoff |

Manifest SHA `224fb0d39b2adccbbc8c58e6a11c7dcd66736d50d7a142ad15d0855f42bb71a9` is a reviewdraft identity, not a launchapproval. Toexecute, lock exactarchivedcore/actionprefixSHA and perbranchRNG in a newapprovedruntime beforephysics. Truncatedprefix states use actualstoredactions and compareexistingphysics/history atthatboundary; neverqpos-onlyreset. Require originalspec.end to contain thewhole3swindow; no silenttaskextension. A83 correctlynames its actualAvisitedstate after nativeprefix, not an inventedpure-nativecontinuation.

Two trainingreturnparents and six distinctdevparents; stop/continue commonprefix entirelysamegroup. No sealedread. No per-direction or perconditioncheckpointcomposition.

## Proposed sequence and stopping rules

1. Engineeringgate: copied-weights disposableupdate in explicitgradcontext, prove actualsame-zreference, nonzerofiniteadaptergradients, identicalcontrol/Returnbatchorder, unchangedbase/U/Q/disc/targets/RNGcontract. Prove λ0 reproduces the adapter-only control update, and execution/replay/qPD stillshareactualvalues. Failedno_graderratum is notclosed by syntaxalone. Allengineeringstepscount; originaldatauntouched.
2. Qualify the two trainingreturndomains using frozenbase-only continuation. ExistingB19mean/sample controls may bereused; near10.0s needs up to2new3s branches withfullinputhistory capture. Completewindow, originaltrueq.02rad hardprotection andsupport/safety unchanged. Off itself failingabsolute.45m/s does not become an absolutequalitypass: here it supplies a tested recoverydirection, finalcandidate stillmustmeet originaloff-relativegates. If base fails to give an evidence-backed recoverydirection, stop: no Returnlabels or training on thatdomain. Do not replace a failedparent.
3. Fixed64adapterupdates perarm, no physicalsamplingduringtheseupdates. This is offline regularizedactorimprovement on archival/Qdata, **not on-policy PPO**. Save init/16/32/64, no checkpointoverwrite. Final64 is the primary comparison; no dev-driven earlycheckpointselection. Store parentB19update38 andincrementaladapterstepcount separately; no assertion that Q/F/B received64updates.
4. Matched real3sdev from six independentparents: recoverydevmean/sample; active/none mean. FrozenV1, B-control64 andB-return64, same restoredstate/RNG. Also B1911.8s postlearningmean/sampletoverify theknownboundary. Release doesnottrackoldanchor. Originalseven-candidateTeacher/searchnotused.

Hardstops: any physics/probability/data/restoreviolation immediatelypreserveandstop; .02rad trueqallowance unchanged. Any loss nonfinite/zero-invalidgradient, frozenmodule change, wrongexecutedreplayaction, parentleak or budgetstop invalidateslearning. No failureparent skipping, no RNGreroll, no clip/reward/gatechange.

Physicaljudgment retains exactV1§4 offrelative slippeak/off+max(.10,.2off), p95/off+max(.020,.2off), activeerror/off+max(.005m,.1off), force/off+max(.25N,.1off), progress≥90%, yawworsening≤5°, heightloss≤.03m, no addednosupport, plus originalabsolute.45m/s/20°reported separately. This draft doesnotretroactivelypassoldV1.

Falsifiableprediction: Return reduces same-z adaptercorrection on independentrecoveryparents and reduces true slip/rootmotion **while meeting the originalgates and retaining activequality**. Rawlossfallingwithoutrealrecovery doesnotcount. If only trainparentsimprove, activebehaviourdegrades, or independentrecoveryfails, conclusion is “currentReturnrecipe did not close the gap”; no λincrease/morerounds. If rawproposalsremain outsideprojection after64steps, report finite-budget optimizationfailure, not a proof allreturnconstraintsareimpossible. No automaticfull B/C restart evenifthissmallpilotworks.

## Complete conservative cost projection — proposed only

One RTXPro5000, heavyjobsserial; no assumedvectorizedphysics. Currentprovenbound: firstV1B19process277.474s includedcoldload, severalfullprefixreconstructions,150controls/3srealphysics,IO andfailedaudit. Use conservative **8statebundles×600s=4800s**, eachactualbranchrestoredexactly. Up to30new3sphysicalwindows: neartrainingoff2; tworecoverydevparents×3models×2modes=12; threeactiveparents×3models=9; none×3models=3; knownB19twonewtrainedmodels×2modes=4. ReusedB19offarchivecost0newphysics. Counting allnewphysics **includingeveryreset/source/replay**, no misleading90s-onlyledger.

| Work | Upper job seconds | Notes |
|---|---:|---|
| Newsource/restore/actualphysics,all30windows | 4800 | includes8coldloadbundles, restore,metricIO; stopifprojectionunsupported |
| Two64stepofflineadapterjobs +engineeringupdate | 600 | fullNVIDIAload/forward/backward/optimizer/IO; newrawlossunbenchmarked, boundedfirstpass |
| Protocol/statehashes/newloss correctnesschecks | 600 | includesimplementation/failedwork |
| Report/curve/checkpoint/IO validation | 600 | allpositiveandnegativecases |
| Failure/stopreserve | 600 | noautomaticretrybeyondcap |
| **Total newproposed cap** | **7200 (2h)** | notdrawn/started inthisaudit |

Outputproposedcap **2GiB**, fullallbranchactualqpos/controls/stateidentity kept; smalladaptercheckpoints and referencestoexistingfrozenweights, nevercopyfullBFMeachcheckpoint. Sharedfree≥100GiB. New128adapteroptimizersteps total, plus≤2disposableengineeringupdates; newRLtransitions0 becauseofflineonly; actualevaluationcontrolticksandqualificationcost separatelycounted. Stage/240h/Pilotbudgetallocation mustbeexplicitlyaccepted beforelaunch; unchangedPilotoldledger is10.279462h. No retroactivecostreset or promise that600s coversanunmeasuredcase.

## Recommendation boundary

There is specificmechanismevidence to justify thisone **adapter-only isolation test**, conditionalon qualifyingindependentrecoveryreferences. There is notenoughevidence to resumeoldB/C, strengthenEnvelope, or trainanallinactiveReturntarget. If preparatoryoffrecoveryfails outsideB19, or thisfinitepilotcannotpreserveactivebehaviour, end thiscurrentrepairrecipe andreturntothe higher-level question of controlfunction/recoverydomain ratherthan anothercoefficientround.
