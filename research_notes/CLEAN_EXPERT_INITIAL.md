# Clean Expert: initialization evidence, not RL success

2026-11-02. User authorizes structured/direct learned motor control. No old six-job proposal or old continuation is restarted. Code runtime999494f7a25224b27780f504af797c32b6ca6879; preparation completes, then20-batch PPO per interface launched under separate supervision. Current progress is in stage ledger, not this snapshot.

|Policy on identical82cm task|3D RMS cm|actual force RMS N|cap|no-contact max root displacement|
|---|---:|---:|---:|---:|
|Standing|54.550|4.710|see raw|native|
|M10 feedback positive control|7.796|4.363|see raw|native|
|A structured after1500 supervised updates|32.672|4.674|84.48%|4.14cm|
|B direct after1500 supervised updates|7.502|4.059|46.45%|0.38cm|

A/B teacher physical trajectories match; same seed101/capacity/timestep/initialization count, but parameterizations and target losses differ. Each has a fresh PPO optimizer. All preparations completed/no physics boundary. Old effective Expert is used only to generate labels, never an online component of learned rollout. B already acquires useful closed-loop behavior from data; whether PPO adds genuine feedback gains is not yet known. This is neither an independent novel-skill discovery claim nor a held-out result.

A labels algebraically fit: max tangent13.656/16, angle40.48deg. But teacher-forced own-memory prediction does not guarantee closed-loop phase/participation control. A has phase.105 cycles before contact, alpha~.23/.32 at9/11s despite task deficit, ending about49cm behind partner during hold; no fall. A's successful offline MSE is insufficient. Short20-batch training is an explicitly bounded optimization test, not a decision to long-train this initialization or widen latent bounds.

B movement RMS8.195cm/4.308N, hold3.746cm/2.898N; mean hold signed anchor-minus-wristX−1.16cm vs teacher−4.46cm. Contact root motion~.83m, meaningful walking retained. Peak tilt11.66deg vs teacher9.67; q_PD step43.03deg vs46.67. Force remains substantial; no claim of fully natural motion. No-contact0force is accompanied by measured root displacement, not used alone as proof of no unsolicited action. Local15cm regression4.352cm/3.395N; not an independent final.

Actual physical traces include all XYZ force/error, feet/support, applied wrench, qpos/qvel and q_PD. Release interval begins at14.05 but actual physical activation may persist at its boundary tick; don't turn the one boundary-active sample's RMS into a claim that force persists for the six-second release. Use full trace/active mask and recovery curves.

Initial checkpoints:
- structured: SHA256 `de653d2962af5378d32375cfc9dbc86b49b3003bd74dc2dac870bf9724d7a539`
- direct: SHA256 `a86ff5f9ca71ed2ae8dc95ceeab0ac980f82ab6ed8d3759c039eb03a62efe488`

Data: `bridge-v2-owned/runs/clean-contact-expert-20261102/{structured-prepare-01,direct-prepare-02}/job`. `qualification/teacher__data.json.gz` contains actual observations/commands and coordinate labels; `dev-0000/` contains full mean/sample/local/no-contact. Preparation supervised wall~245/229s plus process startup; exact ledger charged. Source/notes contain no assets. BFM/PD unchanged. First RL training `structured-train-03` and `direct-train-04`, init/best/latest/last and actual updates recorded independently; no result invented in advance.
