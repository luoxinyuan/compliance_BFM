# M8 final research state — information only

M8 is completed. Read `m8_results.md`, `M8_EXPERIMENT_INDEX.md`, `m8_protocol.json`, then the canonical run's `analysis/` and `compute-ledger.json`. This file is not a queue task; do not restart training or polish the old wrist timing task.

## What is established

One shared small Contact Policy uses actual force/proprio/action history and causal measured contact geometry to execute old right-wrist ±X and new upper-torso downward maintained-contact guidance. The latter uses active knees/hips/root through an original BFM native-goal endpoint. Bilateral Tether, not unilateral pressing. Same BFM/PD frozen.

Projected-z was the predeclared primary. Two new body conditions:3Derror2.778/2.618cm,force2.261/2.139N,versusStanding6.910/5.511cm,4.231/4.147N; forcecap0. Oldwrist+15/−3.5/+3/continuous errors1.863/1.059/.678/1.503cm versusM7-PZ2.160/1.064/.744/1.645cm. Four Student/native comparisons across six no-contact episodes have exact native trajectory parity. 49 full physical episodes plus1 partial candidate, two rounds/12real Student visits; each method7000newupdates, inheritedM7 20500 separate. No final data used to fit or select.

## Method and scope

1430→128→128→258 MLP, same widenedM7-PZ initialization forPZ/AA; two continuous qualified native coordinates plus original .3 tangent and norm16. Body expert uses signed realFz feedback with causal integration/return, no anchor or clock. Already-native body expert projection is nearly identity; online physical parity exact. Wrist expert still has the original bounded independent local correction. This does not prove arbitrary nonnative whole-body actions representable. Full action-aligned label construction and real visited-state aggregation are present.

## Remaining evidence boundaries

Single seed/standing start, one contact point per episode. Body path leaves lateral error/force; T1PZholdXYZRMS2.868/1.684/.081cm and actualmeanforce2.292/−1.353/−.025N. Returnrootpeak.164m/s and qPD1.921deg, despite final stability. Oldcontinuous qPD3.755→4.778deg(PZ),5.761deg(AA),at gate0→1. Small+X AAerror+13.96%,force+34.94%. Oldreleasehas smallnewbodyprogressleak.0186/.0262,finallyzero. These are not hidden or repaired from test feedback. No arbitrary position/direction/generalstate/multicontact/locomotion/realrobot claim. No explicit channel-ablation performed inM8.

## Canonical identities

Branch `gaoe05/m8-contact-body`; physicalruntime `f2511b2ae9f89c5c9dd0c2379c11caf404c3f687`; final archive = Git HEAD containing this file.
Code `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/m8-contact-body/repo/research_notes`.
Run `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m8-contact-body-20261101A`.
Final models `train2-attempt-01/outputs/distill-final.pt` SHA `74f624a96abc3fd4ea63da101dfb0177bf8f426d6fa4da8717d7b219aaee086a`; `alignment-final.pt` SHA `dddb6b536ca33410f300c59dba4b1a7e608d061ba7e9990ebb96b74c1d5ed1d9`.
M7 and earlier assets/models remain unchanged. Sharedrepo6023c6e clean, all owned GPU experiments ended. No push/Bridge/LaunchAgent or nextphase.

## Only proposed next milestone

Sequential right-wrist↔upper-torso contact-site change in one real episode, one active attachment at a time. First qualify feedback in naturally visited transition states; explicitly associate each historical force sample with its measured link/localposition (current single site code is not a validated multisitehistory interface). Then decide small sharedLPP aggregation under new authorization. PreserveM7regressions, not resume wrist-timing-only tuning; do not force walking before trustworthy support-transfer capability.
