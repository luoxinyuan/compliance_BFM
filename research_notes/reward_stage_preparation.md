# Native reward-skill qualification: preparation, GPU execution pending

This is a preparation record for Issue #21, not a claim of qualified skills.
The owner's latest instruction is to wait for their GPU-available notice and
preserve the verified GPU model/evaluation chain. No CPU model or physics fallback.
No real BFM forward, new reward latent, MuJoCo step, or policy update ran here.
Do not update the stage conclusion in Issue #18 until numerical qualification ends.

## Identity and verified download

- Private branch: `gaoe05/bfm-reward-skill-qualification`, based on
  `695a310f5d63776ccab337a943060b59c0b0c483`.
- P: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM`.
- Worktree: `P/bridge-v2-owned/trees/issue-21-reward-stage/repo`.
- Evidence: `P/bridge-v2-owned/runs/issue21-reward-stage-20261030A`.
- Frozen model SHA256:
  `33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.
- Official HF revision: `62b4206d68e026de5e5dc7efb1529bccfb95164c`.
- Sole large download: `new_model_for_training_code_inference/checkpoint/buffers/train/buffer.hdf5`.
- Destination: `/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets/62b4206d68e026de5e5dc7efb1529bccfb95164c/train/buffer.hdf5`.
- Size **26,675,212,288 bytes**; complete local SHA256 equals published LFS SHA:
  `46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b`.
- One HTTP attempt, single connection, 20 MiB/s cap; completed and hashed in
  1328.69 seconds. Owned downloader PID 4072538 exited; no background experiment.
- Accompanying 548-byte config SHA256:
  `5f6ba56ad7c4ec281c75f8bc488f02b7d702fdfa5cf2224aee5f6c94c4d38c20`.

HDF inspection passed: 5000 x 1024 records; observation dimensions 64/463/29/372,
action 29, qpos 36, qvel 35; float32, flags bool. Ring cursor 1250, full=true.
Total array bytes 26,675,200,000. Full-file SHA verifies bytes, not all-row finite
values or physical semantics. The small read-only audit inspected 32 fixed rows.

## Material temporal-contract finding and isolated correction

Initial same-row observation/physics comparison did **not** pass a noise-only
explanation. Preserve `buffer-contract.json`, `buffer-alignment.json`, and v2/v3
outputs; do not hide the original mismatch behind a final passing summary.

31 of 32 sampled rows remain after excluding reset/cursor neighborhoods. In those
rows, observation[t] matches raw physical fields[t-1]:

| Check | same raw row RMS | previous raw row RMS | previous-row max abs |
|---|---:|---:|---:|
| joint velocity (rad/s) | 1.513086 | 0.288448 | 0.499285 |
| projected gravity | 0.032206 | 0.028419 | 0.049675 |
| scaled body angular velocity | 0.170524 | 0.027673 | 0.048718 |
| privileged root height (m) | 0.006463 | **0** | **0** |
| last_action vs 5 x raw action | 0.408553 | **0** | **0** |

Joint-position previous-row max difference is 0.029319 rad (noise plus default
joint offset randomization). No observation is reconstructed from clean qpos.

An old official source revision provides a reproducible mechanism:
`aa9d33989d50f563b5d7a89e22b4603792163359` returns persistent `self.extras`, the
wrapper replaces `new_info['qpos'/'qvel']`, and training later reads `info` from
the preceding call. The dictionaries alias. Current source copies the dictionary.
Evidence excerpts with source blob SHAs/URLs are in `legacy-source-proof.json`.
This explains the measured one-row offset; the precise buffer-generation commit
is still not recorded by the published asset, so do not claim it was proven.

The local adapter is pinned to this exact buffer SHA. It preserves the original
trajectory sampler and observation[t+1], but pairs that successor observation
with qpos/qvel from raw row t (already the successor), and action[t]. It does not
use qpos/qvel[t+1], which are one physical step later in this file. It does not
modify the HDF, model, native reward definitions, reward weighting, or boundaries.
Reset-edge semantics were not separately reconstructed; full inference remains
unexecuted. This is an explicit preprocessing bug correction, not a successful
reward-skill result or an unreported change to the original command.

## Other implementation fixes and checks

- Original `DictBuffer.load(device='cpu')` ignores its device argument; saved
  config says CUDA. Local adapter changes the constructor placement to CPU while
  retaining original HDF loading and sampling. Never load all 26.7 GB onto GPU.
- BaseConfig is frozen. Copy only `inference_batch_size=4096`; do not assign the
  frozen field or change architecture, normalization, or weights.
- Synthetic tests cover actual small HDF loader/sampler pairing, legacy info
  aliasing, ring boundaries/seed repeatability, global softmax aggregation vs
  chunks, placement, and frozen config copy. Real-model execution remains pending.
- Original t=3s full standing snapshot and norm16 standing latent identities
  checked by file/metadata only (`physical-input-check.json`); no reset/step run.
- GPU launcher default is preparation only; execution requires the later owner
  GPU notice and explicit invocation. No watcher, Bridge, `npm run once`, or push.

## Fixed resumed protocol and limitations

See `reward_skill_protocol.json`: seed 20261030, 150,000 original sampled
transitions, same batch for two named rewards, one z each, no best-of-N.
Original global softmax(10r), reward weighting, backward model and projection;
FP32, 4096-row GPU chunks. Data loading/relabeling use CPU as in the original
chain, but BFM inference never falls back to CPU.

Resource plan: 8 CPU cores, 64 GiB owned aggregate RSS, one suitable GPU selected
from all cards, 4 GiB owned VRAM, 3600 seconds cumulative numerical work. Recheck
resources after the owner's notice. Inference supervisor is prepared; numerical
timing/memory and physics supervision still need actual runtime verification.

Planned physics uses the existing original GPU BFM/MuJoCo chain with full common
snapshot restore, verified standing baseline, fixed 0.75s entry / 2.5s execution /
0.75s exit / 2s settle, Tether off, no arm/root residual. The new skill schedule
has not yet run. Do not claim either reward name has the intended arm/body
semantics, safe transitions, or task success before this test.

Next action is to wait for the owner GPU notice, then finish the fixed milestone
continuously within its scientific/resource stop conditions. No Student/PPO.
