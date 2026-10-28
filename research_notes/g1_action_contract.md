# Issue #19: original frozen-BFM standing baseline recovered

Recovery ID: `issue19-recovery-20261028T045520Z`. Successful run: `attempt-06`.
Base `6023c6e257fab8fe58d13d9207fb5e85c32ee9c3`; personal branch `gaoe05/lpp-issue-19`.
This is a fresh manual owner-authorized recovery, not a replay of the consumed Bridge queue task.

## Actual CPU evidence

| Measurement | Actual result |
|---|---:|
| model_loaded | true; original FBcprAuxModel |
| original_wrapper_used | true; HumanoidVerseVectorEnv via original ComplianceEnv |
| simulated_seconds / physics seconds | 6.0 / 6.0 |
| actor calls / physics samples | 300 / 1200 |
| survival / failure | survived_6s / false |
| minimum and final root height | 0.779833750446 m |
| minimum upright cosine / maximum tilt | 0.999029196457 / 2.524866658 deg |
| any-foot / both-feet contact fraction | 0.995833333333 / 0.994166666667 |
| maximum foot XY displacement from reset | 0.016861118813 m |
| network action saturation | 0 |
| YAML torque saturation / XML ctrl out-of-range fraction | 0 / 0 |
| external force RMS / peak | 0 / 0 N |
| model eval / all parameters frozen / no gradients | true / true / true |
| all model parameters CPU / CUDA initialized | true / false |
| peak sum of owned process-group RSS | 7650643968 bytes (7.125217 GiB) |
| successful attempt wall time | 131.488027 s |

The initial root was `[0,0,0.800000011920929,1,0,0,0]` (xyz + wxyz);
leg joint angles were `(-.1,0,0,.3,-.2,0)` on each side, other joints zero.
Actor inputs actually had shapes state `(1,64)`, last_action `(1,29)`, history_actor `(1,372)`.
The existing default-reset `goal_inference` standing latent had norm 16 and dimension 256;
there is no saved reward_locomotion.pkl in this downloaded release. Original actor mean,
normalization/history, action scaling, clipping and PD/simulator path were all retained.
Low-level control was 50 Hz, physics 200 Hz, high-level wrapper 10 Hz.
Original no_force.json thresholds remained root height >= .4 m and upright cosine >= .4.
There was no fixed-PD substitute, new WBC, teacher optimization, or training.

Contact fractions count foot/ground contact presence at every original physics substep,
not force-bearing/support-polygon feasibility. The five no-foot-contact samples were at
[0.005, 0.01, 0.015, 0.025, 0.030000000000000002] seconds during initial settling. Foot displacement is not a contact-conditioned
slip estimate. XML range comparison is commanded ctrl vs range, not measured actuator torque.
No boundary of the unchanged standing protocol was crossed in this one seed-0, six-second run.
This establishes the recovered no-force standing baseline, not contact compliance, teacher quality,
long-horizon stability, or LPP effectiveness. The earlier Issue #17 PD fall remains no evidence
against this actual BFM baseline, and no causal ablation of their differences was performed.

## Asset paths and minimal recovery changes

Existing full LaFAN: `/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof.pkl`
(209659488 bytes), SHA256 `f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11`.
Existing clipped LaFAN: same directory `lafan_29dof_10s-clipped.pkl` (205117479 bytes),
SHA256 `7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`; it was inventoried,
not substituted for the original wrapper's full-file fallback.
Existing BFM release: `/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference`.
Checkpoint `checkpoint/model/model.safetensors`: 3384968272 bytes, SHA256
`33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361`.

`/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/config.json` is a personal metadata copy; its only semantic delta
is `env.lafan_tail_path` pointing at the verified existing full LaFAN. `model-view/checkpoint`
links to the original checkpoint directory. The unmodified original ComplianceEnv receives
this model folder. No weights/data were downloaded or copied; original metadata and all assets
were rehashed unchanged afterward. Reads are normal trusted-SSH reads, with Python output guards,
not an OS-enforced read-only mount.

The preserved original uncommitted launcher SHA was
`4faab68f5f5ce93827373dffeeecb8c6d9530e3849c2ecb4cd8bdc018a88beb8`;
its bytes remain in `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/original/launcher.py`.
Final executed launcher SHA256: `96384beb7b6c4025a883f0fc761e4ad3d902c114a92180c9196a9e402e8b3b42`.
Only this launcher and this Markdown report are intended for the feature commit.
Eight original source/config pins are checked before each launch.

Changes to the existing launcher: explicit personal asset view, precise model-load milestone,
original wrapper/actor/physics observation without additional mj_forward calls, full raw metrics,
nonzero exit on unexecuted/failed baseline, optional --unit-only, and IPC compatibility only.
The original motion library creates multiprocessing Managers. A light forkserver is started
before Torch/model import to avoid copying the loaded process at worker creation. Linux abstract
AF_UNIX addresses preserve the real Manager/Queue protocol without CPFS socket inodes or IP networking.
No scientific function or shared Python installation was edited.

## Failure history, report-call fix and budget

Old Bridge attempts 01/02 and their original needs-approval result remain intact. Their PASS
status only concerned launcher checks; no BFM standing episode ran then. The missing snapshot
LaFAN path did not mean the project dataset was absent on the server.

| Recovery attempt | Result / first actual boundary | Peak group RSS | Wall |
|---|---|---:|---:|
| 03 | model_loaded=true; Manager child AF_UNIX path too long, parent EOFError before wrapper completion | 6.919350 GiB | 128.330 s |
| 04 | short path; guard stopped a transient parent+spawn-child RSS sum above 8 GiB before model wrapper completed | 10.221416 GiB | 117.632 s |
| 05 | early forkserver; filesystem socket bind failed with Errno 95, Operation not supported on CPFS; before model import | 0.041965 GiB | 4.654 s |
| 06 | original BFM/wrapper loaded; 6 s survived; exit 0 | 7.125217 GiB | 131.488 s |

Attempt 04's summed RSS counted the parent's resident pages again during subprocess creation;
physical unique/PSS memory was not sampled, so a unique-physical-memory overrun is not established.
The measured sum really crossed the guard and was stopped; it is not omitted or called a pass.
Later helper processes were verified gone, including those still exiting in older final telemetry.

The old report error was independently reproduced at line 27 of the saved functions.exec call:
`const doc=String.raw` used a template literal containing unescaped Markdown backticks, so parsing
terminated before the path token `bridge` and raised `SyntaxError: Unexpected identifier 'bridge'`.
The failed parse never reached project_write. Structured JSON serialization now transports the
Markdown; syntax and byte-for-byte roundtrip checks passed. It was not a BFM or Python error.

CPU affinity is four logical CPUs, BLAS/Torch one thread, GPU visibility empty; no render call.
Manual recovery retains an 8 GiB sum-of-owned-process-group RSS watchdog sampled every 0.1 s;
there is no hard instantaneous OS memory quota. No RLIMIT_AS cap is used because file mappings
and fork address space are not physical RAM; the successful peak summed VMS was
19197759488 bytes. Trusted SSH has os_isolation=false.
The cumulative old-check plus recovery-attempt wall time was 435.188376 s, below 3600 s;
small IPC probes and metadata/commit handling are separate. The owner explicitly added one
launch and then authorized autonomous technical retries without a launch-count stop, while
retaining CPU/memory/time bounds. Both approvals are appended to recovery-ledger.json.
A completed scientific result stops recovery retries.

## Validation, artifacts and next step

- Each recovery invocation passed the three actual launcher contract tests (pins, unchanged protocol,
  success/failure classification) before its live preflight; only attempt 06 proves standing.
- Local static/classification/asset tests passed without Torch/MuJoCo. The corrected report call
  parses and preserves all Markdown. Real local IPC tests included 265-byte cwd, prestarted forkserver,
  and a 128 MiB parent allocation; the manager remained about 17 MiB.
- A server stdlib-only IPC probe then passed with a 48-byte abstract socket address, a roughly
  27 MiB manager, and no model imports. The successful full run uses that mechanism.
- Independently re-read all 300 raw control rows and 1200 physics samples: six seconds, zero applied
  force, no failure, and recomputed minimum height match the summary. No saved result was used to
  replace execution.
- All 283 tracked shared-source hashes, shared HEAD/branch/worktree registration, original assets,
  old failure logs, consumed queue state, persistent SDK thread and local Bridge config are unchanged.
  No owned recovery process remains. No push, GPU, hardware, shared checkout edit or daemon start.

Server run root: `/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue19-recovery-20261028T045520Z`. Each attempt contains launcher.py, stdout.log, stderr.log,
telemetry.jsonl and supervisor-result.json; successful `attempt-06/outputs/` also contains
result.json, physics_samples.json, control_rows.json. CPU/model/IPC caches are under owned outputs.
Local forensic/report-call/diff/supervisor copies: `~/Projects/robot-research-bridge-v2/.bridge/issue19-recovery/`.

Reproduction requires a fresh owned output directory as cwd, BFM_MODEL_FOLDER set to the personal
model-view, BFM_RECOVERY_OUTPUT set to that output, and the same resource supervisor. The payload
uses the project .venv/bin/python -I -B with this launcher. --unit-only is only static contract
validation. Do not rerun the consumed #19 queue key. The Bridge engine and daily policy were not
expanded by this manual recovery, so a future queue continuation needs its own supported task
and corresponding runtime asset configuration.

Next: use this verified original frozen-BFM pipeline as the no-force control for a separately
specified paired contact experiment. Do not rebuild standing control from the earlier fixed-PD fall.
