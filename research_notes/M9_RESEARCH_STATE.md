# M9 research state — 2026-11-01

Status: bounded stage completed, partial scientific qualification. User authorization is fulfilled for this stage; no automatic next stage. Read `m9_results.md` for full evidence and `M9_EXPERIMENT_INDEX.md` / `m9_protocol.json` for reproduction provenance. Stage Issue: https://github.com/GaoE05/robot-research-bridge/issues/30

Achieved: one shared frozen-M8-based PZ contact policy has actual walking support transfers and limited right-curve response, alongside retained wrist/torso behaviors. 8000 new updates, two real visited-state aggregations, 18094 rows. 83 complete physical trajectories (quality negatives retained),30 sealed final comparisons. Same-state native-z/action identity verified; no redundant native projection optimization.

Final checkpoint SHA256: `f003afd8953b55e8f96ae1f47aea2beb80622f257cef24c451ea34fe28ab48a6`. Path: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m9-contact-locomotion-20261101A/train2-attempt-01/outputs/projected-z-final.pt`. It depends on original M8-PZ SHA `74f624a96abc3fd4ea63da101dfb0177bf8f426d6fa4da8717d7b219aaee086a`. Terminal evaluation runtime commit: `decf2929cbb87bb81ddcdf0db1d1d91237d3b90d`.

New straight: Student error12.057cm/force3.998N, Standing59.846cm/4.721N; root+.841m, single-support29.9%. Right curve:15.967cm/4.629N, root(1.112,-.078)m, move heading-9.12deg; not better than simple force rule. Left: heading+14.36deg but only+.184m forward/error32.015cm; not qualified. New+15deg heading: Student24.147cm/4.726N versus effective expert10.478cm/3.878N; genuine Student generalization gap.

Limits: native motion execution does not establish feedback steering; left expert wrong heading; high force saturation, XYZ coupled burden and qPD jumps remain. No contact-link physical switching or arbitrary front-posture walking entry claim. Old wrist/torso errors regressed6-19%; M8 remains archived and must not be silently replaced. Final no-contact Student and M8 traces identical, but explicit zero-history execution guard exists. No blanket neural intent/generalization claim.

Next decision: prioritize an actually queryable, phase-consistent low-speed velocity/heading feedback expert and then reuse this LPP pipeline, retaining useful straight movement. Request exact same-checkpoint team reference with full initial history/PD/physical steering-stop traces if available; otherwise bound expert acquisition for this family. Do not resume reward/gain scanning, expand contact points, alter shared PD, or secretly train on sealed final data.

Resources: serial GPU2 freshly checked at each launch, 4043.55s supervised total,1146MiB VRAM peak/7.502GiB RSS/~2.5GiB outputs. All owned processes ended, shared checkout6023c6e clean, all frozen source/weight pins intact. No push, Bridge, background worker, new large assets, PPO or real robot. Normal local analysis is CPU; original BFM inference/training remained CUDA.
