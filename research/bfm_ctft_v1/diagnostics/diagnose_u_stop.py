"""Replay the exact U stop once and isolate weights from behavior noise.

Read production artifacts through the original runtime. Perform no learning,
checkpoint writes, replay extensions, plan edits or physical-gate changes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ["CTFT_PINNED_RUNTIME"]).resolve()
STAGE = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "research_notes"), str(ROOT / "research/bfm_ctft_v1")]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import m11_ipc


def write(path, value):
    from cas_checkpoint import durable
    durable(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def study(env, original, contract, output):
    import mujoco
    import numpy as np
    import torch
    from cas_checkpoint import restore_learner
    from f1_mjdata_contract import canonical, metadata
    from learning_checkpoint import digest, rng, set_rng
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common
    from m11_runtime_speed import install as install_speed
    from packed_checkpoint import PackedTensorStore
    from rolling_backend import install as install_backend
    from training_data import episode
    from training_live import TrainingLive

    install_speed(env)
    (output / "backend-prefix").mkdir()
    install_backend(env, original, output / "backend-prefix")
    reset_common(env, original, original.act)
    run = STAGE / "runs/formal-U-01"
    store = PackedTensorStore(run / "store")
    assert json.loads((run / "stopped.json").read_text())["last_committed"] == "round-17408"
    assert json.loads((run / "store/latest.json").read_text())["manifest_sha256"] == (
        "df27aad380f2b84504fb9369161ead8a8a8c2f9242a56978ba0affd5e012bcdd"
    )
    package = store.load("round-17408")
    failure_file, = sorted(run.glob("failure-*.pt"))
    saved_failure = torch.load(failure_file, map_location="cpu", weights_only=False)
    expected_task = saved_failure["Task"]
    expected_spec = expected_task["live_spec"]
    pending = saved_failure["partial_actual_rows"]
    assert len(pending) == 605 and expected_task["live_control"] == 182
    assert package["scheduler"]["collected"] == 17408 and package["learner"]["step"] == 136
    with torch.enable_grad():
        agent, buffers = restore_learner(package["learner"])
    del buffers
    reference_meta = store.metadata("round-0")["learner"]["model"]
    actor_prefix = "_actor."
    b0_state = {
        k.removeprefix(actor_prefix): store.unpack(v)
        for k, v in reference_meta.items() if k.startswith(actor_prefix)
    }
    actor = agent._model._actor
    u_state = {
        k.removeprefix(actor_prefix): v for k, v in package["learner"]["model"].items()
        if k.startswith(actor_prefix)
    }
    assert digest(b0_state) == package["initial_actor_sha"]
    assert digest(u_state) != digest(b0_state)
    for key, value in reference_meta.items():
        if key.startswith("_obs_normalizer."):
            assert torch.equal(store.unpack(value), package["learner"]["model"][key]), key
    model = env.sim.model
    limited = np.flatnonzero(model.jnt_limited)
    ankle_num = contract["dof_names"].index("left_ankle_roll_joint")
    ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_roll_joint")
    ankle_qind = int(model.jnt_qposadr[ankle])
    ankle_vind = int(model.jnt_dofadr[ankle])
    lo = float(model.jnt_range[ankle, 0])
    scale = float(contract["q_PD_scale"][0][ankle_num])
    offset = float(contract["q_PD_offset"][0][ankle_num])
    ratio = float(contract["normalize_ratio"])
    clip = float(contract["clip"])
    stage_records = json.loads((STAGE / "runs/s0-source-admission-01/result.json").read_text())["train"]
    coeff_file = STAGE / "runs/s0-online-probe-01/agent-identity.json"
    coeffs = json.loads(coeff_file.read_text())["original_training_config"]["aux_rewards_scaling"]
    live = TrainingLive(env, STAGE, coeffs, stage_records)
    plan = json.loads((STAGE / "locks/exogenous-training-plan-v2.json").read_text())
    reset = json.loads((STAGE / "locks/source-reset-domain-v1.json").read_text())
    live.restore(package["task"])
    set_rng(package["learner"]["rng"])
    layout = metadata(model, env.sim.data)
    assert digest(canonical(live.snapshot(), layout)) == digest(canonical(package["task"], layout))

    # Replay the final 423 controls of the previous episode, before the
    # real scheduler's handoff into the failing native continuation.
    prefix = []
    while not live.done:
        obs = live.observation()
        z = live.command()
        with torch.no_grad():
            action = agent.act(obs, z, mean=False)
        prefix.append(live.step(action, obs, z))
    assert len(prefix) == 423
    assert digest(prefix) == digest(pending[:len(prefix)]), "Recovered production prefix differs"
    spec = episode(plan, reset, package["scheduler"]["source_index"], "U")
    assert spec == expected_spec
    live.start(spec)
    start_task = live.snapshot()
    start_rng = rng()
    start_canon = digest(canonical(start_task, metadata(model, env.sim.data)))
    initial_q = float(env.sim.data.qpos[ankle_qind])
    results = []

    def run_branch(label, weights, mean, limit):
        actor.load_state_dict(weights, strict=True)
        live.restore(start_task)
        set_rng(start_rng)
        assert digest(canonical(live.snapshot(), metadata(model, env.sim.data))) == start_canon
        rows = []
        details = []
        failure = None
        while not live.done and len(rows) < limit:
            assert time.monotonic() < float(os.environ["CCL_DEADLINE_MONOTONIC"])
            obs = live.observation()
            z = live.command()
            with torch.no_grad():
                nominal = agent.act(obs, z, mean=True)
                action = nominal if mean else agent.act(obs, z, mean=False)
            record = {
                "attempt": len(rows) + 1,
                "time_before": float(env.tether.time),
                "left_ankle_q_before": float(env.sim.data.qpos[ankle_qind]),
                "left_ankle_dq_before": float(env.sim.data.qvel[ankle_vind]),
                "left_ankle_action": float(action[0, ankle_num]),
                "left_ankle_action_mean": float(nominal[0, ankle_num]),
                "left_ankle_q_PD": offset + float(np.clip(float(action[0, ankle_num]) * ratio, -clip, clip)) * scale / ratio,
            }
            details.append(record)
            try:
                rows.append(live.step(action, obs, z))
            except UnsafePhysics as exc:
                failure = exc.row
                break
        trace = env.trace
        if failure is not None and failure is not trace[-1] and failure.get("time") != trace[-1].get("time"):
            trace = [*trace, failure]
        q_values = [r["qpos"][ankle_qind] for r in trace]
        result = {
            "label": label, "mean_only": mean, "controls_completed": len(rows),
            "full_window_completed": live.done, "physical_ticks_recorded": len(trace),
            "max_actual_joint_excess_rad": max(r["joint_limit_excess_max_rad"] for r in trace),
            "max_left_ankle_below_limit_rad": max(0.0, lo - min(q_values)),
            "max_external_force_n": max(float(np.linalg.norm(r["force"])) for r in trace),
            "no_support_ticks_max": live.guard.maximum_no_support_ticks,
            "failure": {k: failure.get(k) for k in (
                "time", "joint_limit_excess_max_rad", "height", "tilt_deg", "support",
                "ctft_stop_reason", "undesired_ground_contacts",
            )} if failure else None,
            "last_input": details[-1],
            "last_actual_q": float(env.sim.data.qpos[ankle_qind]),
            "last_actual_dq": float(env.sim.data.qvel[ankle_vind]),
        }
        write(output / (label + ".json"), {"result": result, "left_ankle_inputs": details, "physics_trace": trace})
        print("CTFT_ANKLE_BRANCH", json.dumps(result, allow_nan=False), flush=True)
        return result, rows

    baseline, recreated = run_branch("U_sampled", u_state, False, 250)
    assert baseline["failure"] is not None and len(recreated) == 182
    assert digest(recreated) == digest(pending[423:]), "Recreated native rollout rows differ"
    for key in ("qpos", "qvel", "ctrl"):
        assert np.array_equal(getattr(env.sim.data, key), expected_task["mjdata_arrays"][key]), key
    assert baseline["failure"]["time"] == expected_task["tether"]["time"]
    results.append(baseline)
    for label, weights, mean in [
        ("U_mean", u_state, True),
        ("B0_sampled_matched_rng", b0_state, False),
        ("B0_mean", b0_state, True),
    ]:
        result, _ = run_branch(label, weights, mean, 250)
        results.append(result)
    summary = {
        "status": "EXACT_U_STOP_AND_FROZEN_CROSSOVER_COMPLETE",
        "production_prefix_controls_exact": len(prefix),
        "U_sampled_completed_rows_exact": len(recreated),
        "U_sampled_failure_qpos_qvel_ctrl_exact": True,
        "same_full_handoff_state": True,
        "same_seed_and_noise_calls_for_both_sampled_branches": True,
        "initial_left_ankle_q_rad": initial_q,
        "source": spec, "weights": {"U": digest(u_state), "B0": digest(b0_state)},
        "branches": results, "updates": 0, "formal_transitions": 0,
    }
    write(output / "crossover.json", summary)
    return summary


def main():
    import argparse
    import mujoco
    import m10_physics
    import m10_stage
    from resource_guard import check

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ["CCL_DEADLINE_MONOTONIC"] = str(started + 600)
    os.environ["BFM_RECOVERY_OUTPUT"] = str(output)
    write(output / "launch.json", {
        "pid": os.getpid(), "runtime": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(), "resource": check(STAGE), "max_physics_ticks": 7000,
    })
    actual_step = mujoco.mj_step
    count = {"actual": 0, "reserved": 0}

    def counted(*args, **kwargs):
        added = int(kwargs.get("nstep", 1))
        count["reserved"] += added
        assert count["reserved"] <= 7000
        result = actual_step(*args, **kwargs)
        count["actual"] += added
        return result

    mujoco.mj_step = counted
    m10_stage.run_study = study
    m10_physics.OUT = output
    try:
        m10_physics.main()
    finally:
        write(output / "cost.json", {"wall_s": time.monotonic() - started, "physics": count, "formal_updates": 0})


if __name__ == "__main__":
    main()
