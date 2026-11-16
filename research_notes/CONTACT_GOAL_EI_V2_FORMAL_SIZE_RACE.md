# Formal attempt03: atomic-size supervision failure

2026-11-16. Engineering interruption; no physical/scientific gate failure observed.

## Actual progress and evidence

Runtime `646ba2531e5ef418a93d7e990bb4bf2850616e6c`, E attempt03.
`runs/formal-E-03/progress.json`: committed P2 block1,1024 formal50Hz rows,
2 accepted Actor updates; oldlogprob error0, final jointKL0.01888550703709082.
Collection120.724255s, update177.052782s, commitIO2.325869s.
First accepted-update sentinel none/zero/released all PASS (`jobs/formal-E-03.log`).
No fixed node or Dev complete; I not started. No behavior-success claim.

`jobs/formal-E-03.json`: FileNotFoundError statting physics-counter.pending during
atomic replacement. Supervisor requested SIGTERM; child saved full independent
`runs/formal-E-03/stop-packages/block-stop-1095` plus formal-stop.json.
1095 rows include71 unoptimized block2 rows. Sidecar cost618.868478s,
16461 actual/reserved physics ticks, already charged. Full stop manifest exists;
this particular package has NOT yet passed new-process restore validation.

## Narrow repair

Only vanished atomic temporary paths are tolerated by directory sizing.
Missing committed evidence, permission failures and other IO errors stay fatal.
The scan is not an atomic filesystem snapshot; published output is counted at the
next scan and the1GiB complete-save reservation remains. Six CPU fault checks PASS,
see contact_goal_formal_size_race_check.json. Training code/physics/thresholds unchanged.
Original runtime/logs untouched; repair committed only to personal maintenance branch.
First CPU patch attempt failed before writes because maintenance had sparse research_notes
checkout; expanding the authorized personal checkout resolved that path issue. No samples lost.

## Remaining continuation work

Original driver has no resume entry. Complete that orchestration before relaunch:
restore model/Adam/RNG/physical/history at1095, retain71 pending rows, block2 target2048,
never rerun block1 or label replay as new on-policy data. Reuse passed sentinel only with
identity evidence. Stop if exact restore fails. This report does not claim restart.
GPU0 has other-project processes at live inspection; no concurrent heavy launch.
Shared free space about602GiB; approved5GiB floor unchanged. Old failures retained.
