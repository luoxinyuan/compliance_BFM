"""Fail closed on scientific decisions, aggregate budgets and sealed data leakage.

No GPU imports or jobs. An approval record identifies the exact resolved value
and its actual user authorization; a suggested/default value is not approval.
"""
import hashlib
import json
from pathlib import Path

REQUIRED = {
    "common": ("discount_reward", "curriculum_principles", "task_ranges",
               "source_generation_and_parent_split", "selection_and_safety"),
    "frozen": ("high_architecture_and_optimization", "matched_pair_schedule"),
    "fb": ("contact_schema", "fb_reward_and_retention", "fb_online_z_schedule",
           "fb_replay_and_checkpoint", "fb_update_recipe"),
}


class UnapprovedContract(RuntimeError):
    pass


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_launchable(contract, track):
    assert track in ("frozen", "fb", "student")
    assert contract["version"] == "DUAL_TRACK_CONTRACT_V1_APPROVED_SCOPE"
    missing = []
    groups = ("common", "frozen") if track == "frozen" else ("common", "fb") if track == "fb" else ("common",)
    for group in groups:
        for name in REQUIRED[group]:
            decision = contract["decisions"].get(name, {})
            if decision.get("status") != "approved" or not decision.get("approval_source") or decision.get("value") is None:
                missing.append(name)
    if track == "student":
        missing.append("teacher_selection_and_student_initialization_not_yet_defined")
    if missing:
        raise UnapprovedContract("Formal launch blocked by unresolved scientific decisions: " + ", ".join(missing))
    limits = contract["approved_limits"]
    assert limits["experiment_job_seconds"] == 240 * 3600
    assert limits["fb_true_native_transitions"] == 1_000_000
    assert limits["expert_high_decisions_total"] == 196_608
    return canonical_sha(contract)


def validate_transition(*, observation, next_observation, action, z, target_z,
                        terminal_observation, terminated, truncated,
                        episode_id, successor_episode_id, origin):
    """Prevent the most consequential replay identity errors before insertion.

    Values are already-evaluated dictionaries/tensors; this is not a simulator.
    The collector must capture the physical successor BEFORE any auto-reset.
    """
    assert origin == "actual_online_execution", "Predictions/Teacher labels are not online transitions"
    assert observation is not None and action is not None and z is not None and target_z is not None
    assert not (terminated and truncated), "True terminal wins over a simultaneous time limit"
    assert episode_id == successor_episode_id, "Successor crosses reset"
    assert next_observation is not None
    if terminated or truncated:
        assert terminal_observation is not None, "Missing pre-reset successor"
    return {"bootstrap": not terminated, "continue_gae": not (terminated or truncated)}


def assert_parent_separation(manifests):
    owners = {}
    for split, rows in manifests.items():
        for row in rows:
            for key in ("source_parent_id", "common_prefix_parent_id", "partner_parent_id"):
                value = row[key]
                owner = owners.setdefault((key, value), split)
                assert owner == split, (key, value, owner, split)
    return {split: len({r["parent_id"] for r in rows}) for split, rows in manifests.items()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("track", choices=("frozen", "fb", "student"))
    args = parser.parse_args()
    print(require_launchable(json.loads(args.contract.read_text()), args.track))
