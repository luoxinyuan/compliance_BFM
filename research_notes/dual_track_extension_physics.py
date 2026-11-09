"""Short step0 physical parity test, not task qualification or training.

Three real native source prefixes; same complete solver/history state per branch.
The original BFM and migrated zero-input model receive the same legal commands.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research_notes")]
import m11_ipc


def study(e, model, contract, out):
    import numpy as np
    import torch
    from native_env import Task, install, spec
    from dual_track_runtime import TrainingBranch
    from dual_track_fb_extension import CONTACT_KEY, load_warm_started_agent, normalize_contact
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend

    install(); speed(e)
    torch.set_num_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (out / "backend-prefix").mkdir()
    backend_info = backend(e, model, out / "backend-prefix")
    agent, identity = load_warm_started_agent(ARGS.model, device="cuda")
    agent._model.train(False)
    task = Task(e, model, contract, 5603)

    class ModelFacade:
        def __getattr__(self, name):
            return getattr(model, name)

        def act(self, obs, z, mean=True):
            assert mean, "Parity test uses the same deterministic native actor"
            R = e.sim.data.xmat[1].reshape(3, 3)
            point, velocity = task.read_point_state()
            active = bool(e.tether.active)
            hist = np.zeros((25, 3))
            rows = list(task.h.rows)[-25:]
            if rows:
                hist[-len(rows):] = [r["force_world"] if r["measurement_valid"] else [0, 0, 0] for r in rows]
            features = normalize_contact(mount=[1, 1, 0, 0, 0, 0],
                force=e.tether.force, error=e.tether.anchor - point if active else np.zeros(3),
                relative_velocity=e.tether.partner_velocity - velocity if active else np.zeros(3),
                point_relative_root=point - e.sim.data.qpos[:3],
                partner_velocity=e.tether.partner_velocity if active else np.zeros(3),
                active=active, force_history=hist, world_to_body=R.T)
            extended = dict(obs, **{CONTACT_KEY: torch.tensor(features, device=z.device)[None]})
            original = model.act(obs, z, mean=True)
            migrated = agent._model.act(extended, z, mean=True)
            assert torch.equal(original, migrated), "Step0 native action changed"
            return migrated

    facade = ModelFacade()
    results = []
    for source, until, onset in [("standing", 4., 4.9), ("goal_arm5", 5.6, 6.2),
                                 ("motion_walk_task", 5.9, 6.5)]:
        task.m = model
        task.reset(spec(.08, 1.6, onset, source, until), record=True, label=source)
        while e.tether.time < onset + .2 - 1e-7 and not task.done:
            task.step(task.native(source, float(e.tether.time)) / 16)
        assert not task.done, (source, task.boundary)
        snapshot = TrainingBranch(task)

        def execute():
            start = len(task.controls)
            for _ in range(5):
                _, _, done, _ = task.step(task.native(source, float(e.tether.time)) / 16)
                assert not done, (source, task.boundary)
            digest = TrainingBranch(task).fingerprint
            return digest, np.array([r["action"] for r in task.controls[start:]]), np.array(e.sim.data.qpos)

        original_digest, original_action, original_qpos = execute()
        snapshot.restore()
        task.m = facade
        migrated_digest, migrated_action, migrated_qpos = execute()
        assert original_digest == migrated_digest
        assert np.array_equal(original_action, migrated_action)
        assert np.array_equal(original_qpos, migrated_qpos)
        results.append(dict(source=source, controls_per_branch=len(original_action),
            complete_physics_history_equal=True, low_action_equal=True,
            actual_qpos_equal=True, actual_active=bool(e.tether.active)))
        task.m = model
    result = dict(status="passed", backend=backend_info, migration=identity,
                  branches=results, formal_training=False, optimizer_steps=0,
                  task_quality_qualified=False, sampled_source_groups_for_science=0)
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    return dict(status="completed", engineering_verification=result, student_optimizer_updates=0)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    ARGS = parser.parse_args()
    # The bounded supervisor owns this directory; outputs must still be new.
    assert ARGS.output.is_dir() and not (ARGS.output / "verification.json").exists()
    os.environ.update(CLEAN_KIND="direct", P4000_CANONICAL_ROW_ACTOR="1",
                      BFM_RECORD_NOMINAL_DIAGNOSTIC="0", BFM_RECOVERY_OUTPUT=str(ARGS.output))
    os.chdir(ARGS.output)
    import m10_stage, m10_physics
    m10_physics.OUT = ARGS.output
    m10_stage.run_study = study
    m10_physics.main()


if __name__ == "__main__":
    main()
