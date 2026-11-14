# Return-to-Prior pilot — negative, fixed recipe stopped

2026-11-14. Approved [protocol](ADAPTER_RETURN_TO_PRIOR_PILOT_PROTOCOL.md); historical V1failure unchanged. **Two64-step adapter clones completed, but independent recovery was not retained. Do not resume full B/C or increase λ/steps.** This is a single-seed offline adapter/Q-surrogate isolation test, not PPO, fullFB training or a claim that every Return mechanism is impossible.

## What actually ran

Original BFM33f410c1…/PD/Envelope/reward/std.01/clip.06/physical gates unchanged. Same B19actor and adapterAdam; U/Q/discriminator/targets jointly frozen. Mainreplay1255/native205 rows from selectedtrainingparents; six developmentparents excluded from newupdates. Returnbatch32 is balanced16/16 across two qualifiedtrainingparents. OriginalB19firstfullinput1row plus newnear base-off actuallyexecuted300rows; not301independentgroups. Mean/sample remain oneparent each.

Near10.0s reference mean/sample completed3s: loadedslippeak .023305/.023946m/s, p95 .012404/.013045, actualqexcess0, no-support0, yaw −.549/−.844°. ExistingknownB19off controls reused after matching model/task/firstBFMobs/z. [Reference qualification](adapter_return_pilot/reference-qualification.json) qualifies these directions, not all inactive states. Full archivedendpoint restore and targetprefixactualphysics/actions passed exact checks. Intermediate RNG explicitly uses the archivedendpoint stream for prospective paired evaluation; it is not falsely named historical10.0s RNG.

Engineering two disposableadaptersteps passed λ0 baseline exactequivalence, finite nonzeroadaptergradients, actualEnvelope→Q action identity, unchangedfrozenmodules/otherAdams/targets/normalizers/RNG. No fullAgent.update called. Parentcompleteupdates38 remain distinct from64incrementaladaptersteps. ExistingfullB no_graderratum is not retroactively passed by this adapter-only check.

Training proxy at pairedstep64: control **13.777044**, Return **0.146568**, reduction **98.94%**. It is a sampled trainingproxy, not closed-loop quality. Initial/16/32/64 models retained; final64 fixed before development. [Learning curve](adapter_return_pilot/learning.png), [loss/gradient endpoints](adapter_return_pilot/learning-summary.json), [engineering](adapter_return_pilot/engineering.json).

## Actual independent recovery

Two exposeddevelopmentparents: fea32998… at11.9s and eb777384… at9.7s, each originalV1/control64/Return64 × mean/sample, full3s each. Four matchedmode comparisons certify exact first fullobservation, legalz, originalBFMaction and actualq/qvel equality. Whole prefix/solver/history restore and actualaction→replay→qPD checks passed. Later states are closed-loop consequences and are not treated as identical states.

Mean results; root displacement measured first5ms row→last. All twelvebranches had **qexcess0 / no-support0 / no hardboundary**. All windows are genuinely released, actualforce0; anchorerror is N/A, never presented as task success or force reduction.

| Parent/model | loadedslip peak(m/s) | p95(m/s) | final world yaw(°) | root XY displacement(cm) | realized discounted3s return |
|---|---:|---:|---:|---:|---:|
| 2 / V1 | 0.0336 | 0.0225 | -15.14 | 6.42 | -0.0369 |
| 2 / control | 0.8490 | 0.1056 | -147.13 | 41.45 | -1.0378 |
| 2 / return | 0.7843 | 0.1033 | -145.73 | 42.31 | -1.0059 |
| 3 / V1 | 0.0181 | 0.0146 | -15.37 | 4.82 | -0.0232 |
| 3 / control | 0.8069 | 0.1025 | -150.35 | 39.91 | -1.0338 |
| 3 / return | 0.8348 | 0.1073 | -148.46 | 39.52 | -1.0023 |

Sample results reproduce the pattern: Return slippeak .828826/.891941m/s andfinalyaw −146.57/−147.85°, versus V1 .036904/.020033 and−15.26/−15.03°. This is not ordinary small footrecovery: highslip and about130°extra yaw accompany roughly40cm movement. Gates have not changed. Originalabsolute .45m/s/20° references fail for the learnedclones. Exact off-relative gates for these two newstates are **unmeasured** because matchedbase-off references were not added; V1 is not silently renamed off. Failure is established by the huge physical regression versus the frozenV1 and absolute coordinates, not by inventing missingoff results. [All12rows, XYZ/safety/identity/traceSHA](adapter_return_pilot/physical-comparison.json), [CSV](adapter_return_pilot/physical-comparison.csv).

Once the first complete independentparent falsified retention, the not-yet-started queue was stopped. The already-running secondparent was allowed to save its full matched result and corroborated failure. No parent was replaced or rerolled. Threeactiveparents, no-contactparent and knownB19postlearning branches remain **not executed**; no active-preservation, no-contact or trainingstate physical improvement claim. Fourteennew3swindows completed (two reference＋twelveevaluation), ≤30cap. All reset/prefix/failed physics counted separately.

## Mechanism evidence and its limits

Same first physicalstate/z/base acrossmodels isolates learnedadapterfunction changes as the intervention. First-step values:

| Parent/model | raw joint-equivalent RMS(rad) | executed qPD correction RMS(rad) | Envelope component trigger | frozenQ CPU local score |
|---|---:|---:|---:|---:|
| 2/V1 | 0.1255 | 0.0989 | 65.5% | 0.198134 |
| 2/control | 0.3740 | 0.1092 | 89.7% | 0.201406 |
| 2/return | 0.1453 | 0.1090 | 89.7% | 0.201415 |
| 3/V1 | 0.1869 | 0.1027 | 75.9% | 0.173614 |
| 3/control | 0.4653 | 0.1098 | 89.7% | 0.176095 |
| 3/return | 0.1862 | 0.1094 | 89.7% | 0.176087 |

**Confirmed:** the rawconstraint improves its trainingdomain but does not retain these two recoverydomains; both offlineupdatedclones degrade. Atparent2, Return rawRMS .145 versus control .374, but executedRMS remains ~.109 forboth under heavyprojection. Return is also worse than frozenV1 rawRMS .125 at thatindependentinput. Shrinking rawproposal in its referencebatch neither guarantees unsaturation nor preserves the actual controlfunction elsewhere.

The finalpairedtrainingbatch has nativeCPRloss .000318 forReturn, yet thisdoesnot certify visitedrecovery. WeightedReturngradientnorm was .05156 versus fullactor .05165 atstep1; bystep64 it was .004620 versus .028083. These are logged batchgradients, not evidence to retuneλ afterdevelopment.

**Compatible, not uniquely proved:** limited recoverycoverage/negative transfer, functiondrift under fixedQ optimization, and Envelope saturation masking useful executionchange. Atparent2, frozenQ prefers Return(.201415) toV1(.198134), while actualdiscounted3s return is −1.0059 versus−.0369. These are different horizons/contracts, so this is a local mismatch warning, not proof of a critic implementationbug or unique cause. OriginalnativeCPR protects anotherstate/z distribution; allinactiveadapterzero is not justified. InitialU parameters/z are equal, but laterU outputs can differ as the actualbody/history diverges.

**Not proved:** standaloneUcausality, everyReturnloss ineffective, currentrewardwrong, or interactionF/B unhelpful. No newC, U/critic learning, Student or sealedtest occurred. This pilot only falsifies the approvedfixed.0008/64step adapter-only recipe.

## Identity, costs and stopping

Return64 SHA **f2056e30087eb6748602ffd4df3815486dbfda23566d73e431d3a28c2de20fec**; control64 **c4e1c1525b37701929b0a0b33b3add420d2d5ca52155f80cf1b0ff5ba112d284**. Theirinitialfile hashes differ by serialization, but completeinitialagent/Adam digests match exactly. [All checkpoint identities](adapter_return_pilot/checkpoint-identities.json).

Fit runtime `1c06170c8932af7fbd95b8d843cc505fc814ab01`; physics `8afe076c9362c74cdb37fc6e936780a959cc6567`; reference `0338837994960b87317a2b6e7d4c3117eeac1da8`. [Source hashes](adapter_return_pilot/source-manifest.json). Rawqpos/qvel, controls, fullstate/solver/history/RNG, parentprefixes and allfailedruns stay in the isolatedstage; no weights/rawtrajectories published.

Charged conservativeupper **3504.110s /7200s (0.9734h)**, includingprep1260s, every failed/successchild job and its preflight, deliberately duplicatedcontrollerwaitingupper, failedtimer9.1s andreport/IO reserve600s. CPUanalysis7.868s included in reportreserve, not doubleadded. Newoutputs atanalysis **0.4215GiB /2GiB** (includes independentworktree); physicsactual **27584ticks**, adapterupdates128＋2disposable; newRLtrainingtransitions0 / Teacherqueries0. OldPilot37006.06334757905s unchanged; this independentstage debit belongs to project240h, not a reset of priorcosts. [Full aggregate ledger](adapter_return_pilot/final-ledger.json).

GPU0 released; queue stopped, no autonomousresumption. The failedone-shot localLaunchAgent was unregistered; originalerrorlogs remain. Nextmostevidenced question is why actorimprovement on the fixedQ surrogate damages a previouslystable recoveryfunction, and which reliable functiondomain can be retained. Do not answer with anotherλ/clip sweep or extrapolate thissmall offlinefailure into a verdict on all interactionRL. Any newintervention requires a separate scoped decision.

[Execution/scheduler errata](ADAPTER_RETURN_PILOT_ERRATUM.md). OriginalV1 failure and all oldresults retained. No main/sharedBFM/PD/SoftSONIC/process/physicalthreshold changes.
