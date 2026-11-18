"""Train-only physical screen for the frozen B0 mean collection branch."""
import hashlib
import json
import os
from pathlib import Path
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


def train_only_manifest(reset, previous):
    """Keep every exposed g0 handoff, plus one midpoint per known ancestor."""
    selectors = {}
    for state in reset["uniform_states"]:
        selectors.setdefault(state["case"], []).append(state["control"])
    sources = {
        (q["spec"]["source"]["case"], q["spec"]["source"]["control"]): "actual_U_g0"
        for q in previous[:-1] if q["spec"]["mode"] == 0
    }
    sources[(10, 171)] = "actual_U_hard_stop"
    for case, controls in selectors.items():
        ctr = sorted(controls)[len(controls) // 2]
        sources.setdefault((case, ctr), "middle_legal_state_of_each_train_ancestor")
    return [
        {"case": case, "control": ctr, "selection": reason}
        for (case, ctr), reason in sorted(sources.items())
    ]


def study(env, original, contract, out):
    import numpy as np
    import torch
    from agent import build
    from learning_checkpoint import digest
    from locomotion_rollout import UnsafePhysics
    from m8_stage import reset_common
    from m11_runtime_speed import install as install_speed
    from rolling_backend import install as install_backend
    from training_live import TrainingLive

    install_speed(env)
    (out / "backend-prefix").mkdir()
    install_backend(env, original, out / "backend-prefix")
    reset_common(env, original, original.act)
    home = STAGE / "runs/formal-U-01"
    failure_file, = list(home.glob("failure-*.pt"))
    prior = torch.load(failure_file, map_location="cpu", weights_only=False)
    reset = json.loads((STAGE / "locks/source-reset-domain-v1.json").read_text())
    cases = train_only_manifest(reset, prior["scheduler"]["episodes"])
    assert len(set(q["case"] for q in cases)) == 12
    write(out / "fixed-train-only-manifest.json", cases)
    records = json.loads((STAGE / "runs/s0-source-admission-01/result.json").read_text())["train"]
    coeffs = json.loads((STAGE / "runs/s0-online-probe-01/agent-identity.json").read_text())["original_training_config"]["aux_rewards_scaling"]
    scale = json.loads((STAGE / "locks/common-aux-scale-v1.json").read_text())["value"]
    with torch.enable_grad():
        frozen, _ = build(aux_scale=scale, compile_execution=False)
    expected = json.loads((STAGE / "runs/u-hardstop-diagnostic-01/crossover.json").read_text())["weights"]["B0"]
    assert digest(frozen._model._actor.state_dict()) == expected
    live = TrainingLive(env, STAGE, coeffs, records)
    results = []
    for item in cases:
        spec = dict(mode=0, force=False, parent="existing-train-native-screen",
                    branch="native_continuation",
                    source={k: item[k] for k in ("case", "control")})
        live.start(spec)
        stopped = None
        while not live.done:
            assert time.monotonic() < float(os.environ["CCL_DEADLINE_MONOTONIC"])
            obs = live.observation()
            z = live.command()
            with torch.no_grad():
                action = frozen.act(obs, z, mean=True)
            try:
                live.step(action, obs, z)
            except UnsafePhysics as exc:
                stopped = exc.row
                break
        if stopped is not None:
            assert stopped["time"] == env.trace[-1]["time"]
        item = dict(item, actual_controls=live.i, intended_controls=live.controls,
                    max_actual_joint_excess_rad=max(r["joint_limit_excess_max_rad"] for r in env.trace),
                    maximum_no_support_ticks=live.guard.maximum_no_support_ticks,
                    maximum_wrist_force_n=max(float(np.linalg.norm(r["force"])) for r in env.trace),
                    hardstop={k: stopped.get(k) for k in (
                        "time", "joint_limit_excess_max_rad", "tilt_deg",
                        "ctft_stop_reason", "undesired_ground_contacts")}
                    if stopped is not None else None)
        assert item["maximum_wrist_force_n"] == 0
        write(out / ("case-%02d-%03d.json" % (item["case"], item["control"])),
              dict(result=item, trace=env.trace))
        results.append(item)
        print("FROZEN_NATIVE_SCREEN", json.dumps(item), flush=True)
    n_safe = sum(x["hardstop"] is None for x in results)
    answer = dict(status="FROZEN_G0_SCREEN_COMPLETE", safe=n_safe, total=len(results),
                  source_ancestors=12, rows=results, updates=0, formal_transitions=0,
                  authority="screen only, does not install behavior-policy amendment")
    write(out / "screen.json", answer)
    return answer


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
    os.environ["CCL_DEADLINE_MONOTONIC"] = str(started + 900)
    os.environ["BFM_RECOVERY_OUTPUT"] = str(output)
    write(output / "launch.json", dict(pid=os.getpid(), resource=check(STAGE),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_physics_ticks=22500))
    step = mujoco.mj_step
    ticks = {"actual": 0, "reserved": 0}

    def counted(*args, **kwargs):
        n = int(kwargs.get("nstep", 1))
        ticks["reserved"] += n
        assert ticks["reserved"] <= 22500
        result = step(*args, **kwargs)
        ticks["actual"] += n
        return result

    mujoco.mj_step = counted
    m10_stage.run_study = study
    m10_physics.OUT = output
    try:
        m10_physics.main()
    finally:
        write(output / "cost.json", dict(wall_s=time.monotonic() - started, physics=ticks, formal_updates=0))


if __name__ == "__main__":
    main()
