# Frozen local / whole-body qualification

No further alpha interruption fitting. S0/S1 and A0–A2/B0–B2 remain unchanged. The new comparison uses S1 and A2 (direct alpha distillation), not B2 chosen after a favorable result. `native` means the frozen A2-controlled standing/native-goal path, not a clock replay or instantaneous full-goal command.

## Interface audit and comparison

Both controllers use the original verified standing state, frozen BFM actor and normalization, norm-16 256D latent coordinates, original 29D network action, original normalized G1 PD and 200Hz physics. The older S1 source, original model/PD/Tether and source pins are compared before execution. Stored S1 decisions and all five actor frames per decision must reproduce raw output, executed latent, action and q_PD.

S1's 474D input is 465D normalized actor inputs plus the last three completed 50Hz force samples divided by20N. The network and contact detector run at10Hz; raw output is held for five actor calls. Its original 0.2s release gate is evaluated at50Hz. Its component-bounded tangent residual uses `project_z(standing + .3*gate*tangent(tanh(raw)))` and never receives anchor/time. No arm mask is added: non-arm latent effects must be measured.

A2's 1425D input is the same 465D actor inputs plus320 completed50Hz force samples divided by5N. Its frozen alpha runs at50Hz along the original66.96degree native-goal slerp path. Learned exit and original memory remain intact. These histories/scales/gates are different and are not interchangeable. No teacher400ms rule is newly imposed on A2.

Both start from the exact originalt3 full standing snapshot. At onset3.8s the common anchor is aligned once to the saved unforced wrist. Small+3.5cm and large+18.681928110688906cm each use4s smoothstep,1s hold, release8.8s and2s recovery, K80/D8/cap5N/seed0. Same contract across the six branches; actual force is endogenous, not forcibly shared. This changes the small-task partner schedule relative to historical S1 training, deliberately qualifying the existing skill on a common deployment task without claiming new generalization training. The large A2 condition matches its olddev_late regression and is checked for reproduction; it is not an unseen test.

## Fixed decision and conditional switching

Before results: require small local error better than native by5%, large native better than local by5%, each winner better than standing by5%, winner force no more than5% worse than standing, smaller small-task torso rotation with local, all original physical safety checks,2s recovery speed<=0.02m/s, and20ms q_PD max step<=3degrees. These are bounded deterministic qualification criteria, not statistical significance or a new universal human-interaction benchmark. Full XYZ/hold errors and force saturation remain reported.

Only if the comparison qualifies, run one large-task fixed scheduler: local until5.3s, one-second smooth spherical blend toward the contemporaneous native command, then one-second spherical return toward standing from release8.8s. Both frozen controllers retain continuously warmed histories on the actually executed trajectory, with their own original update/release rates. No concatenation/addition of latent vectors, no direct q_PD interpolation or joint correction. The scheduler uses time solely for interface qualification and is not a learned contact selector. Same-state instantaneous skill-swap q_PD differences are logged as a counterfactual, not applied to the robot.

If complementarity or safety fails, stop before this switch; no threshold/controller retuning or training. SerialGPU inference only,8CPU/32GiBRSS/8GiBownedVRAM/1800s cumulative, exact owned process cleanup. No CPU model fallback, downloads, shared checkout changes, push or Bridge.
