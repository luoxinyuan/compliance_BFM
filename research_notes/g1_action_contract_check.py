"""Issue #19: original frozen-BFM no-force baseline; log-only CPU launcher.

Run with the registered g1-action-contract-unit check. Exit zero means that
launcher contract checks ran, NOT that a blocked/negative rollout succeeded.
No stored-result replay, controller replacement, rendering, or asset downloads.
"""
import os
os.environ["MUJOCO_GL"] = "disable"  # Before numpy, torch, mujoco, or GLFW.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import ast
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = Path("/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference")
MODEL = Path(os.environ.get("BFM_MODEL_FOLDER", str(SOURCE_MODEL)))
OUTPUT = Path(os.environ["BFM_RECOVERY_OUTPUT"]) if "BFM_RECOVERY_OUTPUT" in os.environ else None
LAFAN_SOURCE = Path("/mnt/data-oss/users/yi.gao/compliance_BFM/data/lafan_29dof.pkl")
WEIGHT_SHA = "33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361"
WEIGHT_BYTES = 3384968272
PINS = {
    "humanoidverse/force_compliance/environment.py": "424edfd5caeeefdd299284875674698a6904cf06161f1297e8a6f791a7eafdf5",
    "humanoidverse/force_compliance/no_force.json": "4e084a6ed18cc4cb270c119a9754c4f02387725dd371686924979a9490be866c",
    "humanoidverse/force_compliance/run.py": "ce4cf68f1659bb15646cfdfe55dd5ee83a9f91444a295cac312d0802669f8530",
    "humanoidverse/agents/base_model.py": "daa48871eb9bd14c10f88bcc578bd5e42721bc0f4babaef2128e9f41f97754ad",
    "humanoidverse/agents/envs/humanoidverse_isaac.py": "e7f467a1c57bdb347ed10e556c0e1bff63028702443309a228cbeb8e9c7d1985",
    "humanoidverse/envs/legged_base_task/legged_robot_base.py": "5de7f691e32e9798975722e7c4a6c7e28d5ccdb4044c28d9101ff21102dbdd09",
    "humanoidverse/simulator/mujoco/mujoco.py": "b2c6ee5419d62cfc2aae26b1ecb14c2223a5f6a520bf6c37ae6f63432e4eb990",
    "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml": "b3c8b7d25faca3908d874bef4471f1362385428612cce845d2df5467cd71f53e"
}
CFG_PATH = "humanoidverse/force_compliance/no_force.json"
LAFAN = ROOT / "humanoidverse/data/lafan_29dof.pkl"


def configure_local_ipc():
    """Use Linux abstract AF_UNIX sockets: CPFS does not implement socket inodes.

    Runs in the parent and forkserver preload. This only changes IPC addresses,
    retaining multiprocessing's real authenticated Manager/Queue machinery.
    """
    if sys.platform != "linux" or OUTPUT is None:
        return
    import multiprocessing.connection as connection
    if getattr(connection.arbitrary_address, "_bfm19_abstract", False):
        return
    original = connection.arbitrary_address
    def address(family):
        if family == "AF_UNIX":
            return b"\x00compliance_BFM-issue19-" + os.urandom(12).hex().encode("ascii")
        return original(family)
    address._bfm19_abstract = True
    connection.arbitrary_address = address


configure_local_ipc()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def outcome(duration, failed, force_peak):
    if force_peak != 0:
        return "invalid_no_force_protocol"
    return "survived_6s" if duration >= 6.0 - 1e-9 and not failed else "negative"


def inspect_asset(label, path):
    # Read permission, not exists(), is the gate. Never fall back after denial.
    try:
        with path.open("rb") as stream:
            first = stream.read(1)
        return dict(label=label, path=str(path), readable=bool(first),
                    bytes=path.stat().st_size)
    except (PermissionError, FileNotFoundError, OSError) as exc:
        return dict(label=label, path=str(path), readable=False,
                    error_type=type(exc).__name__, error=str(exc))


def preflight():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    result = dict(technical_contract="pass", runtime_status="preflight",
                  model_loaded=False, original_wrapper_used=False,
                  simulated_seconds=0.0, scientific_outcome="not_run",
                  address_space_soft_bytes=soft, address_space_hard_bytes=hard,
                  checkpoint_expected_bytes=WEIGHT_BYTES, assets=[])
    for label, path in (
        ("checkpoint", MODEL / "checkpoint/model/model.safetensors"),
        ("release_config", MODEL / "config.json"),
        ("model_config", MODEL / "checkpoint/model/config.json"),
        ("model_init", MODEL / "checkpoint/model/init_kwargs.json"),
    ):
        asset = inspect_asset(label, path)
        result["assets"].append(asset)
        if not asset["readable"]:
            result.update(runtime_status="blocked_asset", first_boundary=asset)
            return result
        if label == "checkpoint" and asset["bytes"] != WEIGHT_BYTES:
            result.update(runtime_status="blocked_changed_checkpoint",
                          first_boundary="Checkpoint size differs from pinned REPORT.md")
            return result
    original_config = json.loads((SOURCE_MODEL / "config.json").read_text())
    mapped_config = json.loads((MODEL / "config.json").read_text())
    expected_config = json.loads(json.dumps(original_config))
    expected_config["env"]["lafan_tail_path"] = str(LAFAN_SOURCE)
    if mapped_config != expected_config or (MODEL / "checkpoint").resolve() != (SOURCE_MODEL / "checkpoint").resolve():
        raise ValueError("Recovery mapping must only relocate existing LaFAN; checkpoint unchanged")
    result["asset_mapping"] = {"source_model": str(SOURCE_MODEL), "model_folder": str(MODEL),
                               "only_config_delta": "env.lafan_tail_path", "lafan_path": str(LAFAN_SOURCE)}
    # Match ComplianceEnv's exact motion-path selection; no substitute motion data.
    env_config = json.loads((MODEL / "config.json").read_text())["env"]
    configured_motion = Path(env_config["lafan_tail_path"])
    selected_motion = configured_motion if configured_motion.is_file() else LAFAN
    asset = inspect_asset("lafan_original_wrapper_selected", selected_motion)
    result["assets"].append(asset)
    if not asset["readable"]:
        result.update(runtime_status="blocked_asset", first_boundary=asset)
        return result
    # The original loader maps the full checkpoint in addition to model storage.
    # This is only a necessary lower bound, not a guarantee of sufficient VMS.
    if soft != resource.RLIM_INFINITY and soft < 2 * WEIGHT_BYTES:
        result.update(runtime_status="blocked_address_space",
                      first_boundary="RLIMIT_AS below model + one checkpoint mapping; no limit raised")
        return result
    result["checkpoint_sha256"] = digest(MODEL / "checkpoint/model/model.safetensors")
    if result["checkpoint_sha256"] != WEIGHT_SHA:
        result.update(runtime_status="blocked_changed_checkpoint",
                      first_boundary="Checkpoint SHA256 differs from pinned REPORT.md")
        return result
    result["lafan_sha256"] = digest(selected_motion)
    result["runtime_status"] = "ready"
    return result


class LauncherContract(unittest.TestCase):
    def test_pinned_original_sources(self):
        for path, expected in PINS.items():
            with self.subTest(path=path):
                self.assertEqual(digest(ROOT / path), expected)

    def test_unchanged_no_force_protocol(self):
        cfg = json.loads((ROOT / CFG_PATH).read_text())
        self.assertEqual(
            [cfg[k] for k in ("enabled", "episode_duration", "seed",
                             "initial_root_height", "initial_joint_positions",
                             "min_root_height", "min_upright")],
            [False, 6.0, 0, None, {}, 0.4, 0.4])
        ast.parse(Path(__file__).read_text())

    def test_survival_requires_full_horizon_and_zero_force(self):
        self.assertEqual(outcome(6.0, False, 0), "survived_6s")
        self.assertEqual(outcome(6.0, True, 0), "negative")
        self.assertEqual(outcome(0.935, False, 0), "negative")
        self.assertEqual(outcome(6.0, False, 1), "invalid_no_force_protocol")


def run_baseline(result):
    # -I intentionally removes cwd from sys.path. Add only this pinned worktree.
    sys.path.insert(0, str(ROOT))
    # Start a light forkserver before importing Torch or loading any weights.
    # Later original MotionLib Managers inherit the light server, not the model.
    import multiprocessing as mp
    mp.set_start_method("forkserver", force=True)
    if OUTPUT is None or Path.cwd().resolve() != OUTPUT.resolve():
        raise ValueError("Recovery requires its owned output as cwd")
    (OUTPUT / "ipc").mkdir(exist_ok=False)
    mp.current_process()._config["tempdir"] = "ipc"
    with mp.Manager() as manager:
        probe = manager.Queue()
        probe.put("ready")
        if probe.get() != "ready":
            raise RuntimeError("Original Manager/Queue IPC warmup failed")
    result["multiprocessing"] = {"start_method": "forkserver", "ipc_directory": "ipc", "socket_namespace": "linux-abstract-AF_UNIX",
                                  "started_before_model_import": True}
    result["stage"] = "import_original_path"
    import numpy as np
    import torch
    import mujoco
    from humanoidverse.force_compliance.config import ComplianceConfig
    from humanoidverse.force_compliance.environment import ComplianceEnv
    torch.set_num_threads(1)
    cfg = ComplianceConfig(**json.loads((ROOT / CFG_PATH).read_text()))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    result["stage"] = "original_ComplianceEnv_constructor"
    result["model_loaded"] = None  # Constructor may load weights before a later setup error.
    # Observe the original loader's return even if later environment setup fails.
    from humanoidverse.agents import load_utils
    original_loader = load_utils.load_model_from_checkpoint_dir
    def observed_loader(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        result["model_loaded"] = True
        result["stage"] = "model_loaded_building_original_wrapper"
        print("MILESTONE " + json.dumps({"model_loaded": True}), flush=True)
        return model
    load_utils.load_model_from_checkpoint_dir = observed_loader
    try:
        env = ComplianceEnv(MODEL, cfg, device="cpu")
    finally:
        load_utils.load_model_from_checkpoint_dir = original_loader
    result.update(model_loaded=True, original_wrapper_used=True,
                  model_class=type(env.model).__name__,
                  wrapper_class=type(env.wrapped).__name__,
                  standing_source=env.standing_source, latent_norm=float(env.standing.norm()),
                  model_eval=not env.model.training,
                  all_parameters_frozen=not any(p.requires_grad for p in env.model.parameters()),
                  actor_keys=env.keys, low_level_dt=env.dt, high_level_hz=env.actual_hz,
                  torch_version=torch.__version__, mujoco_version=mujoco.__version__)
    try:
        env.reset(seed=cfg.seed)
        m, d = env.sim.model, env.sim.data
        pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        result["initial_root_xyz_wxyz"] = d.qpos[:7].tolist()
        result["initial_joint_angles"] = d.qpos[7:].tolist()
        result["observation_shapes"] = {k: list(env.obs[k].shape) for k in env.keys}
        result["joint_order"] = list(env.sim.dof_names)
        result["yaml_effort"] = env.base.torque_limits.cpu().reshape(-1).tolist()
        result["xml_motor_ctrlrange"] = m.actuator_ctrlrange[6:].tolist()
        result["stage"] = "actual_no_force_rollout"
        measured = []
        action_clipped, action_total = 0, 0
        foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
                    for side in ("left", "right")]
        feet0 = d.xpos[foot_ids, :2].copy()
        original_physics = env.sim.simulate_at_each_physics_step
        original_act = env.model.act

        def observed_act(*args, **kwargs):
            nonlocal action_clipped, action_total
            action = original_act(*args, **kwargs)
            action_clipped += int((action.abs() >= 1).sum())
            action_total += action.numel()
            return action

        def observed_physics():
            # Observe the original substep, before wrapper auto-reset can mask a fall.
            original_physics()
            # Original ComplianceEnv._physics_step already refreshes mj_forward.
            support = [False, False]
            for contact in d.contact[:d.ncon]:
                bodies = [int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])]
                if 0 in bodies:
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, max(bodies)) or ""
                    if "ankle" in name or "foot" in name:
                        support[0 if "left" in name else 1] = True
            limit = np.asarray(result["yaml_effort"])
            torque = d.ctrl[6:].copy()
            xml_range = m.actuator_ctrlrange[6:]
            measured.append(dict(time=float(env.tether.time), height=float(d.qpos[2]),
                upright=float(d.xmat[pelvis].reshape(3, 3)[2, 2]),
                support=any(support), double_support=all(support),
                slip=float(np.linalg.norm(d.xpos[foot_ids, :2] - feet0, axis=1).max()),
                yaml_saturation=float(np.mean(np.abs(torque) >= limit - 1e-6)),
                xml_clipping=float(np.mean((torque < xml_range[:, 0]) | (torque > xml_range[:, 1])))))
        env.sim.simulate_at_each_physics_step = observed_physics
        env.model.act = observed_act
        raw = torch.zeros_like(env.standing)  # Ignored by the original baseline=True branch.
        rows = []
        while not env.done:
            _, _, _, info = env.step(raw, baseline=True)
            rows.extend(info["rows"])
        force_peak = max(r["peak_force"] for r in rows)
        duration = rows[-1]["time"]
        result.update(runtime_status="executed", simulated_seconds=duration,
                      scientific_outcome=outcome(duration, info["failed"], force_peak),
                      failure=bool(info["failed"]), physics_samples=len(measured),
                      actor_calls=action_total // 29,
                      simulated_physics_seconds=len(measured) * float(m.opt.timestep),
                      root_height_final_m=measured[-1]["height"],
                      wrapper_file=sys.modules[type(env.wrapped).__module__].__file__,
                      environment_file=sys.modules[type(env).__module__].__file__,
                      simulator_file=sys.modules[type(env.sim).__module__].__file__,
                      model_device=env.model.device,
                      cuda_initialized=torch.cuda.is_initialized(),
                      all_parameters_cpu=all(p.device.type == "cpu" for p in env.model.parameters()),
                      no_parameter_gradients=all(p.grad is None for p in env.model.parameters()),
                      root_height_min_m=min(r["height"] for r in measured),
                      upright_min=min(r["upright"] for r in measured),
                      tilt_max_deg=math.degrees(math.acos(np.clip(min(r["upright"] for r in measured), -1, 1))),
                      support_fraction=float(np.mean([r["support"] for r in measured])),
                      double_support_fraction=float(np.mean([r["double_support"] for r in measured])),
                      foot_xy_displacement_max_m=max(r["slip"] for r in measured),
                      action_saturation_fraction=action_clipped / action_total,
                      yaml_torque_saturation_fraction=float(np.mean([r["yaml_saturation"] for r in measured])),
                      xml_ctrl_clipping_fraction=float(np.mean([r["xml_clipping"] for r in measured])),
                      force_rms_n=float(np.sqrt(np.mean([r["force_squared"] for r in rows]))),
                      force_peak_n=force_peak)
        if OUTPUT:
            (OUTPUT / "physics_samples.json").write_text(json.dumps(measured, allow_nan=False))
            (OUTPUT / "control_rows.json").write_text(json.dumps(rows, allow_nan=False))
        if result["scientific_outcome"] != "survived_6s":
            result["first_boundary"] = "Original baseline did not meet the unchanged 6s no-force survival protocol"
    finally:
        if "original_physics" in locals():
            env.sim.simulate_at_each_physics_step = original_physics
            env.model.act = original_act
        env.close()


def emit_result(result):
    # The Bridge returns only the last 2000 stderr characters; preserve full stdout.
    if OUTPUT:
        (OUTPUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    compact = {k: v for k, v in result.items()
               if k not in ("assets", "initial_joint_angles", "joint_order",
                            "xml_motor_ctrlrange", "yaml_effort", "first_boundary")}
    boundary = result.get("first_boundary")
    compact["first_boundary"] = (boundary.get("error", boundary) if isinstance(boundary, dict) else boundary)
    compact["asset_reads"] = {a["label"]: a["readable"] for a in result["assets"]}
    print("SUMMARY " + json.dumps(compact, sort_keys=True), file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true", help="No model or physics imports")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LauncherContract)
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        return 1
    if args.unit_only:
        return 0
    result = preflight()
    print("PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    if result["runtime_status"] == "ready":
        try:
            run_baseline(result)
        except Exception as exc:
            # Do not swallow execution errors as a successful technical test.
            result.update(runtime_status="technical_error", scientific_outcome="not_established",
                          error_type=type(exc).__name__, first_boundary=f"{type(exc).__name__}: {str(exc) or repr(exc)}")
            emit_result(result)
            raise
    emit_result(result)
    return 0 if result.get("scientific_outcome") == "survived_6s" else 2


if __name__ == "__main__":
    raise SystemExit(main())
