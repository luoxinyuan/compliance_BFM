# Active whole-body compliance (experimental)

**Current whole-body extension:** see [PARALLEL_TRAINING.md](PARALLEL_TRAINING.md)
for ten random single-contact points and synchronous multi-GPU sampling. The
single-right-hand prototype and its initial validation are documented below.

## Repository inspection, before implementation

- `agents/load_utils.py:load_model_from_checkpoint_dir(outer/checkpoint, device)` loads the original model. `model.act(observation, z, mean=True)` is the frozen low-level interface.
- `agents/fb/model.py:project_z` uses sqrt(z_dim) times unit normalization when `cfg.archi.norm_z` is true. `act` does not project its input. Dimension and convention must come from the checkpoint.
- `reward_inference.py` infers `move-ego-0-0` and stores it in `reward_inference/reward_locomotion.pkl`. This is the standing *candidate*, whose stability must be evaluated.
- `agents/envs/humanoidverse_isaac.py:HumanoidVerseIsaacConfig.build(1)` builds the existing observation adapter; reset supports `reset_to_default_pose=True`; step supports `to_numpy=False`. Actor input keys come from the checkpoint's actor input filter, including history when configured.
- `envs/legged_base_task/legged_robot_base.py:_physics_step` executes torque application and `simulator.simulate_at_each_physics_step` per substep (MuJoCo default 200 Hz, controller 50 Hz).
- `simulator/mujoco/mujoco.py` exposes actual `MjModel`, `MjData`. Its old external force helper ignores application-point torque. This experiment independently uses world-frame `xfrc_applied` with COM torque, confined to an instance hook; original source is unchanged.
- `train.py` trains FB/CPR rather than PPO. A separate small PPO trainer will optimize only the high-level actor/critic.
- MuJoCo backend is single-environment. Initial implementation explicitly rejects other backends and does not claim vectorized training.

No motion-to-force supervision or tracking reward is used. The existing environment may load its motion library for observation infrastructure; its original task reward is discarded.

## Implemented equations and timing

World point and velocity: `x_r = x_body + R_body p_local`, `v_r = J_point(q) qdot` (includes angular velocity at offset points). The Jacobian is only used for physical point velocity, never for motion-to-force supervision or action generation.

For each physics substep, `F_raw = K_h (x_h-x_r) + D_h (v_h-v_r)` and `F = F_raw min(1, F_max / max(||F_raw||, eps))`. Force and `tau_COM = (x_r-x_COM) × F` are added to MuJoCo's world-frame `xfrc_applied`, and removed after the step. Existing unrelated wrenches are preserved. The force in observations and reward is this same applied, capped force. Reward averages the squared force over every executed substep, including action-dependent force changes within the held latent interval. Forces are not re-sampled after an autoreset.

Anchor: `x_h = x_start + s(u) delta`, `u = clip((t-t_activate-anchor_delay)/T,0,1)`, `s(u)=3u²-2u³` (or linear). Velocity is its analytic derivative, zero before/after the ramp. Direction is uniform on the sphere or explicitly configured. Attachment starts at time zero by default, independently of the one-second anchor-motion delay. Anchor origin/displacement are fixed for the episode; default attachment is maintained until termination. Turning off interaction or configuring release is an experiment setting, never a policy action.

Let `s = project_z(z_stand)`, `v = tanh(a_high)`, and `v_tan = v - s (v·s)/(s·s)`. Then `z = project_z(s + residual_scale * clip(||F||/gate_force,0,1) * v_tan)`. A nonzero tangent perturbation cannot cancel the standing vector. The actual checkpoint's dimension and `project_z` determine the sphere; non-normalized models are explicitly unsupported. The Gaussian high-level action is stored before this deterministic transformation, so PPO likelihood ratios remain valid. Zero force produces the standing latent exactly.

Observation: normalized existing BFM actor inputs (including its proprioceptive history), recent actual force vectors, active flag, local contact point/body encoding, and previous latent. No anchor, future trajectory, reference motion or target point enters actor or critic.

Per low-level step:

`r = -w_F mean_substeps(||F||²/F_scale²) - w_angular ||omega_base||² - w_limits ||joint_limit_violation||² - w_ground undesirable_ground_contacts - w_torque ||joint_torque||² - w_acceleration ||joint_acceleration||² - w_action mean((a-a_prev)²) - w_latent mean((z-z_prev)²)`.

Physical penalties are averaged over substeps; latent change is charged once per new high-level command. Falling is detected from actual pre-reset root height/tilt/nonfinite state, new MuJoCo numerical warnings or simulation-time rollback, plus environment termination. Failure incurs a separate penalty and the discounted maximum remaining force cost, so ending the episode cannot simply erase future force costs. No root-displacement or stepping penalty is used. This is a finite-horizon episodic objective; time limits have zero bootstrap value, independently of failure.

A command is held for `round(1/(high_level_hz*low_level_dt))` low-level steps. PPO receives `R=sum_i gamma^i r_i`, transition discount `gamma^k`, terminal mask and GAE trace decay per high-level decision. The critic estimates value at decision boundaries. BFM parameters have gradients disabled and are absent from the optimizer. No low-level fine-tuning is implemented.

## Commands

Run from `/home/xl521/BFM-Zero`, using the existing `.venv`:

```bash
MODEL=pretrained/official/new_model_for_training_code_inference
.venv/bin/python -m unittest discover -s tests -v

# No-force frozen standing validation
.venv/bin/python -m humanoidverse.force_compliance.run baseline \
  --model-folder "$MODEL" --config humanoidverse/force_compliance/no_force.json \
  --output artifacts/force_compliance/no_force --device cuda --video

# Frozen standing with maintained moving tether
.venv/bin/python -m humanoidverse.force_compliance.run baseline \
  --model-folder "$MODEL" --output artifacts/force_compliance/pulling --device cuda --video

# Small smoke run; automatically checks no-force and pulling baseline before PPO
.venv/bin/python -m humanoidverse.force_compliance.run train \
  --model-folder "$MODEL" --output artifacts/force_compliance/train \
  --iterations 2 --rollout 32 --device cuda

# Matched absolute anchor trajectories for baseline and learned actor
.venv/bin/python -m humanoidverse.force_compliance.run evaluate \
  --model-folder "$MODEL" --policy artifacts/force_compliance/train/policy.pt \
  --output artifacts/force_compliance/comparison --episodes 3 --device cuda --video
```

`default.json` exposes all interaction, reward, initial-posture and timing parameters. Use `direction=[1,0,0]`, `[-1,0,0]`, `[0,1,0]`, `[0,0,1]`, `[0,0,-1]` in separate JSON configs for directional tests; change displacement and stiffness for magnitude/generalization tests. Only one contact body is active per experiment. `initial_joint_positions` maps existing DOF names to radians; an empty mapping uses the original robot default pose.

`--standing-file` optionally selects an existing `reward_locomotion.pkl` **from the same model**. Without one, the model's goal inference encodes its initial default standing observation once. This is only a candidate latent, not an assertion that it balances. The training preflight rejects a falling candidate. Latent inference here is permitted standing initialization; there is no ongoing motion tracking objective.

Artifacts: high-level-only `policy.pt`, `config.json`, `runtime.json`, `standing_latent.pt`; preflight and training logs; paired JSON force/position/latent traces, `metrics.json`, force plots, optional videos with anchor, tether, force arrow and magnitude. Failure and horizon completion are distinct. Settling time is null unless the force stays below threshold for the remaining episode, for at least the configured window, and the robot did not fall. Support transitions are reported as **support_changes**, not claimed as a reliable footstep detector. Root/joint/point movement metrics are displacement from reset, not traveled path length.

## Scope and limitations

One MuJoCo G1 only. No Isaac external-force adapter, vectorized PPO, unilateral push contact, admittance baseline, or low-level fine-tuning. Training is intentionally a short runnable experiment, not proof of learned compliance. Force reduction in the free-body physics test validates the tether, not active humanoid behavior. Matched nonzero anchor trajectories and stable robot rollouts are necessary for any compliance conclusion. The original framework's motion library is loaded to retain its observation/reset infrastructure, but none of its tracking rewards are used.

The pretrained assets come from the authors' [official release](https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/new_model_for_training_code_inference). They are stored under `pretrained/official/`, outside tracked source and outside high-level checkpoints. The original data/video modifications present before this task are left intact.

## Completed validation

See [REPORT.md](REPORT.md) for the downloaded model provenance, 46 passing regression tests, actual frozen G1 baselines, two-update PPO smoke run, and three matched evaluation episodes. The smoke-trained actor shows no meaningful force improvement; this is not a trained compliance result.
