"""Meaningful migration/admission/replay regressions, no policy optimization."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dual_track_admission import (UnapprovedContract, assert_parent_separation,
                                 require_launchable, validate_transition)
from dual_track_fb_extension import (CONTACT_KEY, ContactFirstBlock, normalize_contact,
                                     load_warm_started_agent)


def expect_rejected(fn):
    try:
        fn()
    except (AssertionError, UnapprovedContract):
        return
    raise AssertionError("Invalid case unexpectedly accepted")


def contract_and_replay_tests(contract):
    pending=copy.deepcopy(contract)
    pending['decisions']['discount_reward']['status']='pending_Yi'
    expect_rejected(lambda: require_launchable(pending, "frozen"))
    expect_rejected(lambda: require_launchable(pending, "fb"))
    if contract['decisions']['discount_reward']['status']=='approved':
        assert require_launchable(contract,'frozen') and require_launchable(contract,'fb')
    row = dict(observation={"x": 1}, next_observation={"x": 2}, action=[0.], z=[1.],
               target_z=[1.], terminal_observation={"x": 2}, terminated=False,
               truncated=True, episode_id="e1", successor_episode_id="e1",
               origin="actual_online_execution")
    assert validate_transition(**row) == {"bootstrap": True, "continue_gae": False}
    assert validate_transition(**dict(row, terminated=True, truncated=False)) == {
        "bootstrap": False, "continue_gae": False}
    expect_rejected(lambda: validate_transition(**dict(row, successor_episode_id="e2")))
    expect_rejected(lambda: validate_transition(**dict(row, terminal_observation=None)))
    expect_rejected(lambda: validate_transition(**dict(row, origin="prediction_branch")))
    group = dict(parent_id="p1", source_parent_id="s1", common_prefix_parent_id="c1", partner_parent_id="a1")
    assert assert_parent_separation({"dev": [group], "sealed": []})["dev"] == 1
    expect_rejected(lambda: assert_parent_separation({"dev": [group], "sealed": [dict(group, parent_id="p2")]}))
    return dict(pending_contract_blocks_both_tracks=True, terminal_and_truncation_distinct=True,
                pre_reset_successor_required=True, predicted_rows_rejected=True, shared_parents_rejected=True)


def layernorm_tests(device):
    from humanoidverse.agents.nn_models import Block
    torch.manual_seed(5601)
    x = torch.tensor([[.1, .2, -.5, .8]], device=device)
    # Independent minimal counterexample: zero padding changes the normalization
    # even before the new weights can influence the output.
    old_norm = torch.nn.functional.layer_norm(x, (4,))
    padded_norm = torch.nn.functional.layer_norm(torch.cat([x, torch.zeros_like(x)], -1), (8,))[..., :4]
    naive_change = float((old_norm - padded_norm).abs().max())
    assert naive_change > .01
    gradients = {}
    for parallel in (1, 2):
        block = Block(10, 16, True, parallel).to(device)
        ext = ContactFirstBlock(copy.deepcopy(block), old_obs_dim=6, extra_dim=7).to(device)
        xx = torch.randn((3, 10), device=device)
        extra = torch.randn((3, 7), device=device)
        if parallel == 2:
            xx = xx.expand(2, -1, -1)
            extra = extra.expand(2, -1, -1)
        full = torch.cat([xx[..., :6], extra, xx[..., 6:]], -1)
        assert torch.equal(block(xx), ext(full)), parallel
        ext(full).square().sum().backward()
        norm = float(ext.contact_weight.grad.norm())
        assert norm > 0 and torch.count_nonzero(ext.contact_weight) == 0
        gradients[str(parallel)] = norm
    features = normalize_contact(mount=[1, 1, 0, 0, 0, 0], force=np.array([5., 0, 0]),
        error=np.array([.25, 0, 0]), relative_velocity=np.array([.5, 0, 0]),
        point_relative_root=np.array([.2, .1, .5]), partner_velocity=np.array([.15, 0, 0]),
        active=True, force_history=np.zeros((25, 3)), world_to_body=np.eye(3))
    assert features.shape == (97,) and np.all(features[[6, 9, 12]] == 1)
    return dict(naive_zero_padding_max_normalization_change=naive_change,
                migrated_block_bitwise_parity=True, contact_gradient_norm=gradients,
                contact_feature_unit_checks=True)


def real_model_tests(model_dir, device):
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModel
    start = time.monotonic()
    agent, manifest = load_warm_started_agent(model_dir, device=device)
    old = FBcprAuxModel.load(str(model_dir), device=device, strict=True)
    new = agent._model
    old.train(False); new.train(False)
    torch.manual_seed(5602)
    obs = {k: torch.randn((2, *space.shape), device=device) * .1 for k, space in old.obs_space.spaces.items()}
    extended = dict(obs, **{CONTACT_KEY: torch.randn((2, 97), device=device)})
    z = old.sample_z(2, device=device)
    action = torch.zeros((2, 29), device=device)
    with torch.no_grad():
        comparisons = {
            "actor": (old.act(obs, z), new.act(extended, z)),
            "backward": (old.backward_map(obs), new.backward_map(extended)),
            "forward": (old.forward_map(obs, z, action), new.forward_map(extended, z, action)),
            "critic": (old.critic(obs, z, action), new.critic(extended, z, action)),
            "aux_critic": (old.aux_critic(obs, z, action), new.aux_critic(extended, z, action)),
            "discriminator": (old.discriminator(obs, z), new.discriminator(extended, z)),
        }
    parity = {k: dict(exact=torch.equal(a, b), max_error=float((a - b).abs().max())) for k, (a, b) in comparisons.items()}
    assert all(v["exact"] for v in parity.values()), parity
    # Prove every introduced parameter actually belongs to the correct fresh
    # optimizer. Freezing an unchanged parameter is not evidence of adaptation.
    params = {id(p) for opt in [agent.actor_optimizer, agent.forward_optimizer,
              agent.backward_optimizer, agent.critic_optimizer, agent.aux_critic_optimizer]
              for group in opt.param_groups for p in group["params"]}
    names = [k for k, v in new.named_parameters() if k.endswith("contact_weight") and not k.startswith("_target_")]
    assert names and all(id(dict(new.named_parameters())[k]) in params for k in names)
    return dict(status="passed", manifest=manifest, parity=parity, added_online_matrices=len(names),
                fresh_optimizer_membership=True, model_test_wall_s=time.monotonic() - start,
                formal_updates=0, real_physical_transitions=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert not args.output.exists(), "Immutable result"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    contract = json.loads(args.contract.read_text())
    result = {"admission": contract_and_replay_tests(contract), "layernorm": layernorm_tests(args.device)}
    result["real_model"] = real_model_tests(args.model, args.device)
    result.update(wall_s=time.monotonic() - start, formal_training=False, optimizer_steps=0,
        model_sha256=hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
