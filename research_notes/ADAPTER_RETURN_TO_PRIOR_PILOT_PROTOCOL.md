# B-only Return-to-Prior pilot — approved protocol and preparation status

Checked 2026-11-13T02:21:05.552117+00:00. Yi approved the [unchanged draft](ADAPTER_RETURN_TO_PRIOR_PILOT_DRAFT.md) at `ce5cd6ea6345f6121612ad4a6915b3fb833164e6` with “可以。按 Return 小试验草案走”. The original draft and all previous failed qualifications remain historical records. This file records approval and preparation, **not a completed experiment**.

## Locked intervention

Two clones of B19: core SHA `0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984`, parent complete updates38 / real50Hz transitions4750. Copy the same adapter weights and adapter Adam. Freeze original BFM/PD, U, Q, discriminator, all targets and their optimizer states. Only adapter receives64 new updates per clone; common seed6262013. Save immutable0/16/32/64; final64 is the primary comparison, no development-selected earlier checkpoint.

B-control uses the original B adapter actor objective. B-return adds fixed λ=.0008 times raw same-current-state/same-current-z joint-equivalent Return loss: mean((PD_scale×adapter_raw)^2)/(.1rad)^2. Current z/previous executed z and BFM inputs must be genuine combined-policy history. Same replay32/native32/parent-balanced Return32 index order in both arms; coefficient0 still uses the same batch schedule. No Q/U/discriminator/target updates. This is offline adapter surrogate optimization, not PPO or a full B continuation. Coefficient basis and proxy limitations remain exactly as in the draft; no sweep.

Preserve original Envelope V1, std=.01/clip=.06, original reward, controls5/20/100ms, Tether and physical criteria. Actor permissions unchanged; no force-gated deployment switch, hidden task/time/active/anchor/partner inputs or new observation fields. B/C formal training remains stopped. BC500/A128/B19 and all prior assets untouched.

## Groups and actual archive checks

| Role | Parent | Time(s) | Archive round |
|---|---|---:|---:|
| return_train | `8dde57eac4325d7701fafe97d48f57cb962cd8ff121a9e4770a1bbdffa10db4e` | 11.8 | 19 |
| return_train | `2670e363f8c204b64ab32dd614237f032a571e1cd03f4e5c50524c3caef42f79` | 10.0 | 17 |
| recovery_dev | `fea32998d2c1c42c5dad39c9e186641b29671f6e63f8973bcde5389186f77c93` | 11.9 | 12 |
| recovery_dev | `eb77738422ceb4787fad231c3b9b9a7eec4d690ecfe6fdaaa21cacd31e63323d` | 9.7 | 13 |
| active_dev | `b69d7a23e0491199d8a8aa55d741a9878bc2b136b9d2c29b0a7c48a3527d916f` | 6.0 | 1 |
| active_dev | `06c173441f3204e6b90e5e5fb230e9356f15e30e91495171915d5a4f6897bd70` | 6.0 | 3 |
| active_dev | `d52717d57a1504880c53e84b5707998e4f21614551ef369335ffe6d3eb2afa3b` | 7.0 | 5 |
| none_dev | `6a2fe7552234ceb9dd5944b5601c46e3424d3995699b22bc02e6d37410dbc314` | 6.0 | 83 |

CPU verification checked the draft SHA, all referenced core/manifest identities, actual source and original task end. Available assets **do not certify successful physical reconstruction**. Truncated prefixes still require complete actual-action replay, archived boundary comparison, full solver/history/last-z capture and explicitly matched evaluation RNG before physics. A83 is the real A-visited no-contact state; it is not a fabricated native-only prefix. No task-window extension.

Actual B19 lists:4750 replay rows/760 native rows. New permitted main batches contain **1255 replay rows and205 native rows** after excluding all six development parent IDs. Native grouping uses10 contiguous real source blocks matched to actual online episode/spec order, source/time/slot checks; not a random frame split. Training parents are the two Return parents plus existing reliable `7f3a7da6…` for the original objective. Full exact indices remain in the isolated stage; [public provenance](adapter_return_pilot/preparation-summary.json) includes their manifest hash. Previously pretrained B19 saw these groups, so all development is exposed mechanism evidence, not sealed generalization.

Return labels only from physically validated same-z base-only recovery and its actually executed qualified successors. Existing B19 off mean/sample3s may be reused after identity checks. Its first complete high-level input exists; other old off successors lack full deploy/history records and must not be fabricated. New near10.0s off mean/sample needs full capture. A failed training recovery parent stops before learning; do not replace it. Group-balanced sampling must not count mean/sample or adjacent rows as new independent groups.

## Sequence and original judgment

1. Copied-weight engineering check in explicit gradient context, ≤2 disposable adapter updates; λ0 equivalence, same batches and frozen-module/optimizer/RNG identity; actual action/Q/replay/q_PD contract. Existing no_grad erratum is not closed by syntax.
2. Recovery-reference qualification for both training parents. Source/history/solver restoration must pass; full3s and original hardq allowance.02rad unchanged. Existing B19 off remains an imperfect reference, not an absolute quality pass.
3. Only then run the two fixed64-step clones. No new physical data collection during those updates.
4. Matched3s exposed evaluation: two recovery parents mean/sample, three active parents mean, one none parent mean; frozen V1/control64/Return64. Known B19 postlearning mean/sample also checked. Same state/RNG, no directional/condition model stitching. Up to30 new windows as itemized in the unchanged draft, all reset/replay/failed ticks counted.

Original V1 §4 judgments remain: off-relative loaded slip peak≤off+max(.10,.2off), p95≤off+max(.020,.2off); active error≤off+max(.005m,.1off), force≤off+max(.25N,.1off), progress≥90%, yaw worsening≤5°, height loss≤.03m, no added no-support; absolute .45m/s and20° separately report. Release does not retain anchor tracking. If an exact matched off reference is absent for a development state, its off-relative gate is **unmeasured**, not passed; do not substitute a different parent or frozen V1 and call it off. No expansion of windows to fill missing cells without accounting/scope review.

Loss decrease alone is insufficient. Independent recovery improvement plus retained active quality is required; train-only improvement, degraded active behavior or failed recovery ends this fixed recipe. No coefficient increase, extra rounds or automatic full B/C restart.

## Budget and current resource block

New cap **7200 cumulative job seconds /2GiB**, including preparation, failed work, physics, IO and publication; part of project240h, separate from unchanged stopped Pilot37006.06334757905s. Preparation provisionally charges a conservative600s upper bound inclusive of the measured8.45s CPU archive calculation, not an additional600+8.45. Remaining≤6600s; future actual jobs and output must fit it. Draft4800s physics/600s update forecasts must be rechecked against this remaining cap. No uncharged retries.

Live checks: GPU0 RTXPro5000 is occupied by another project's SoftSONIC queue (observed smoke followed by formal job); CPU quota210 cores, available RAM about612GiB, IO pressure0, shared free about348GiB>100GiB. These are dated checks, not future launch permission. **No GPU job, physics step, optimizer step or Teacher query launched here.** No process from that project was touched. No background resource watcher or automatic queue was installed. Repeat live resource verification on resumption; approval persists and need not be requested again.

Full original trial is authorized but currently **RESOURCE_BLOCKED_NOT_STARTED**. Model outcomes and original V1 failed scientific judgment are unchanged. Executable training/physical harness qualification and final full-state/RNG launch manifest remain pending; do not infer them from this CPU availability check.

## Pre-evaluation execution lock (2026-11-14)

Recovery reference near10.0s mean/sample completed3s each with full archived replay/strict state restore; knownB19 off reference reused after exact identity checks. Fit finished64adaptersteps perarm plus2disposable engineeringsteps. Frozenmodules/targets/otherAdams and λ0 baseline equivalence passed. Final64 fixed before evaluation. Exposed evaluation order locks caseindices2/3/4/5/6/7/0; within each Return64, control64, frozenV1 (case0 only the two newclones); mean then sample for recovery, mean for active/none. Order is a logistical choice made before any postlearning physical outcome, not model selection. Any hardphysics/correctness event stops currentjob/queue and leaves subsequentcells missing; no parent replacement or threshold change.

Restore erratum: archived non-NULL efc_JT bytes were omitted by the old replay-only restore. Newruntime copies the exact archived array (same shape/dtype/writable), alongside existing two archivedarrays, then requires the unchanged fullcanonical byte equality. No array was masked or declared irrelevant. Originalfailedtrace kept. IPC first-launch error also kept and fixed by importing existing m11_ipc at child-entry module level. Scheduled08:11 resumption failed on same-thread writer conflict; no experiment ran then. Costs retained.
