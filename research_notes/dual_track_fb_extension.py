"""Isolated contact extension; no formal learning or shared-model mutation on import.

Keep the released residual embedding's input LayerNorm domain unchanged. Adding
zero coordinates to that LayerNorm is NOT a function-preserving migration.
The new fixed-scaled input enters through zero weights after that normalization
and before the existing activation. Original parameter/state names stay intact.
"""
import copy

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONTACT_KEY = "contact_privileged_v1"
CONTACT_DIM = 97
EXTENSION_VERSION = "contact97-preserve-input-layernorm-v1"


def split_input(x, old_obs_dim, extra_dim):
    extra = x[..., old_obs_dim:old_obs_dim + extra_dim]
    old = torch.cat((x[..., :old_obs_dim], x[..., old_obs_dim + extra_dim:]), -1)
    return old, extra


class ContactFirstBlock(nn.Module):
    def __init__(self, original, old_obs_dim, extra_dim):
        super().__init__()
        self.mlp = original.mlp
        self.old_obs_dim, self.extra_dim = old_obs_dim, extra_dim
        linear = self.mlp[1]
        shape = ((linear.weight.shape[0], extra_dim, linear.weight.shape[-1])
                 if linear.weight.ndim == 3 else (linear.weight.shape[0], extra_dim))
        self.contact_weight = nn.Parameter(linear.weight.new_zeros(shape))

    def forward(self, x):
        old, extra = split_input(x, self.old_obs_dim, self.extra_dim)
        y = self.mlp[1](self.mlp[0](old))
        delta = (torch.bmm(extra, self.contact_weight) if self.contact_weight.ndim == 3
                 else F.linear(extra, self.contact_weight))
        y = y + delta
        for layer in self.mlp[2:]:
            y = layer(y)
        return y


class ContactFirstLinear(nn.Module):
    def __init__(self, original, old_obs_dim, extra_dim):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.old_obs_dim, self.extra_dim = old_obs_dim, extra_dim
        self.contact_weight = nn.Parameter(self.weight.new_zeros((self.weight.shape[0], extra_dim)))

    def forward(self, x):
        old, extra = split_input(x, self.old_obs_dim, self.extra_dim)
        return F.linear(old, self.weight, self.bias) + F.linear(extra, self.contact_weight)


def extend_model_in_place(model, extra_dim=CONTACT_DIM):
    """Call before final optimizer construction; original source files are read only."""
    from humanoidverse.agents.nn_filters import DictInputFilterConfig
    from humanoidverse.agents.normalizers import IdentityNormalizerConfig

    assert CONTACT_KEY not in model.obs_space.spaces, "Already extended"
    old_space = model.obs_space
    new_space = gym.spaces.Dict(dict(old_space.spaces, **{
        CONTACT_KEY: gym.spaces.Box(-10., 10., (extra_dim,), dtype=np.float32)}))
    changed = []
    # The discriminator deliberately remains robot-only: native expert snippets
    # are free of physical contact, which must not become an adversarial shortcut.
    for name in ["_actor", "_forward_map", "_backward_map", "_critic", "_aux_critic",
                 "_target_forward_map", "_target_backward_map", "_target_critic", "_target_aux_critic"]:
        if not hasattr(model, name):
            continue
        module = getattr(model, name)
        old_dim = module.input_filter.output_space.shape[0]
        keys = module.input_filter.cfg.key
        assert isinstance(keys, (tuple, list)), (name, keys)
        module.input_filter = DictInputFilterConfig(key=[*keys, CONTACT_KEY]).build(new_space)
        if name.endswith("backward_map"):
            assert isinstance(module.net[0], nn.Linear)
            module.net[0] = ContactFirstLinear(module.net[0], old_dim, extra_dim)
        else:
            embeddings = [k for k in ("embed_s", "embed_sa", "embed_z") if hasattr(module, k)]
            assert embeddings, name
            for key in embeddings:
                seq = getattr(module, key)
                assert hasattr(seq[0], "mlp"), (name, key)
                seq[0] = ContactFirstBlock(seq[0], old_dim, extra_dim)
        changed.append(dict(module=name, old_input=old_dim, added_input=extra_dim))
    norm = model._obs_normalizer
    norm._normalizers[CONTACT_KEY] = nn.Identity()
    norm.cfg = norm.cfg.model_copy(update={"normalizers": {
        **norm.cfg.normalizers, CONTACT_KEY: IdentityNormalizerConfig()}})
    model.obs_space = new_space
    freeze_robot_normalizers(model)
    return changed


def freeze_robot_normalizers(model):
    model._obs_normalizer.train(False)


def normalize_contact(*, mount, force, error, relative_velocity, point_relative_root,
                      partner_velocity, active, force_history, world_to_body):
    """Current causal values, in SI units; no task/clock/future/search state.

    mount=[present,wrist,torso,local_x,local_y,local_z] is installation knowledge.
    active is privileged physical state, never a deployable sensing assertion.
    Past world forces are expressed in the CURRENT pelvis frame, as in O_P/O_S.
    History must contain exactly 25 true 50Hz samples, oldest first; missing
    pre-reset samples are explicitly padded with zero by the collector.
    """
    R = np.asarray(world_to_body, dtype=np.float64)
    history = np.asarray(force_history, dtype=np.float64)
    assert R.shape == (3, 3) and history.shape == (25, 3)
    assert np.asarray(mount).shape == (6,)
    current = np.r_[mount, R @ force / 5., R @ error / .25,
                    R @ relative_velocity / .5, R @ point_relative_root,
                    R @ partner_velocity / .5, float(active)]
    features = np.r_[current, (history @ R.T / 5.).reshape(-1)]
    assert features.shape == (CONTACT_DIM,) and np.isfinite(features).all()
    return np.clip(features, -10, 10).astype(np.float32)


def load_warm_started_agent(model_dir, *, device="cpu"):
    """Fresh optimizer/replay/counters; explicit targets synced to online weights.

    This builder alone does not choose discount/reward/replay/z scheduling.
    Calling update for a formal run requires a separate approved admission file.
    """
    import json
    from pathlib import Path
    import safetensors.torch
    from humanoidverse.agents.base_model import migrate_legacy_aux_critic_config
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig

    model_dir = Path(model_dir)
    raw_cfg = migrate_legacy_aux_critic_config(json.loads((model_dir / "config.json").read_text()))
    raw_cfg["device"] = device
    kw = json.loads((model_dir / "init_kwargs.json").read_text())
    cfg = FBcprAuxAgentConfig(model=FBcprAuxModelConfig(**raw_cfg), compile=False, cudagraphs=False)
    agent = cfg.build(json_to_space(kw["obs_space"]), kw["action_dim"])
    safetensors.torch.load_model(agent._model, model_dir / "model.safetensors", strict=True, device=device)
    manifest = extend_model_in_place(agent._model)
    agent.obs_space = agent._model.obs_space
    # Rebuild optimizer parameter lists, including the new contact matrices.
    # Existing setup applies random initialization, so LOAD PRETRAINED WEIGHTS
    # AFTERWARDS. Never call setup_training again after this final load.
    agent.setup_training()
    missing, unexpected = safetensors.torch.load_model(
        agent._model, model_dir / "model.safetensors", strict=False, device=device)
    expected = {k for k in agent._model.state_dict() if k.endswith("contact_weight")}
    assert set(missing) == expected and not unexpected, (missing, unexpected)
    assert all(torch.count_nonzero(agent._model.state_dict()[k]) == 0 for k in expected)
    for name in ("forward_map", "backward_map", "critic", "aux_critic"):
        getattr(agent._model, "_target_" + name).load_state_dict(
            getattr(agent._model, "_" + name).state_dict(), strict=True)
    freeze_robot_normalizers(agent._model)
    assert all(not x["state"] for x in agent.optimizer_dict.values())
    return agent, dict(version=EXTENSION_VERSION, modules=manifest,
                       fresh_optimizers=True, targets_synced_to_online=True,
                       model_source=str(model_dir), contact_dim=CONTACT_DIM,
                       formal_training=False)

