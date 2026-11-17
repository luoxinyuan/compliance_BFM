"""Issue67 S0: source/config identity and exact original auxiliary-unit audit.

CPU-only. Executes AST-extracted original arithmetic with scalar fixtures, not
physics, optimization, synthetic training, or a replacement reward definition.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace as NS

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
SOURCE = P / 'repo'
ASSET = Path('/mnt/data-oss/users/yi.gao/compliance_BFM/pretrained/official/new_model_for_training_code_inference')
OUT = P / 'experiments/bfm_ctft_v1/runs/s0-asset-reward-01'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    return node


def run():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg_path = ASSET / 'config.json'
    model_dir = ASSET / 'checkpoint/model'
    weights = model_dir / 'model.safetensors'
    cfg = json.loads(cfg_path.read_text())
    mcfg = json.loads((model_dir / 'config.json').read_text())
    init = json.loads((model_dir / 'init_kwargs.json').read_text())
    with weights.open('rb') as f:
        header = json.loads(f.read(struct.unpack('<Q', f.read(8))[0]))
    weight_sha = sha(weights)
    assert weight_sha == '33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
    modules = {}
    for name in ['_actor', '_forward_map', '_backward_map', '_critic', '_aux_critic', '_discriminator',
                 '_target_forward_map', '_target_backward_map', '_target_critic', '_target_aux_critic',
                 '_obs_normalizer', '_aux_reward_normalizer']:
        keys = [k for k in header if k.startswith(name + '.')]
        shapes = {k: header[k]['shape'] for k in keys}
        modules[name] = dict(tensors=len(keys), shapes=shapes)
        assert keys, name
    files = [cfg_path, model_dir / 'config.json', model_dir / 'init_kwargs.json',
             SOURCE / 'humanoidverse/envs/legged_base_task/legged_robot_base.py',
             SOURCE / 'humanoidverse/agents/envs/humanoidverse_isaac.py',
             SOURCE / 'humanoidverse/train.py',
             SOURCE / 'humanoidverse/agents/fb_cpr_aux/agent.py',
             SOURCE / 'humanoidverse/agents/nn_models.py']

    env_path = files[3]
    node = function(env_path, '_compute_reward')
    env_ns = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(env_path), 'exec'), env_ns)
    make_path = SOURCE / 'humanoidverse/train.py'
    node = function(make_path, 'make_transition')
    transition_ns = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(make_path), 'exec'), transition_ns)

    agent_path = SOURCE / 'humanoidverse/agents/fb_cpr_aux/agent.py'
    node = function(agent_path, 'update')
    first = next(i for i,n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'aux_reward' for t in n.targets))
    # Original zero-allocation + weighted-sum loop, stopping before normalizer.
    block = node.body[first:first + 2]
    assert isinstance(block[1], ast.For)
    aux_code = compile(ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])), str(agent_path), 'exec')
    assert not any(isinstance(n, ast.Name) and n.id == 'dt' for n in ast.walk(ast.Module(body=block, type_ignores=[])))

    fixtures = []
    for dt in [.01, .02, .04]:
        raw = torch.tensor([.01], dtype=torch.float32)  # .1rad squared; arithmetic only.
        env = NS(rew_buf=torch.zeros(1), reward_functions=[lambda: raw.clone()],
                 reward_names=['penalty_ankle_roll'], reward_scales={'penalty_ankle_roll': -4. * dt},
                 num_envs=1, config=NS(rewards=NS(reward_penalty_reward_names=[],
                     reward_penalty_curriculum=False, only_positive_rewards=False)),
                 extras={'aux_rewards': {}}, episode_sums={'penalty_ankle_roll': torch.zeros(1)},
                 use_reward_penalty_curriculum=False, use_reward_limits_dof_pos_curriculum=False,
                 use_reward_limits_dof_vel_curriculum=False, use_reward_limits_torque_curriculum=False,
                 add_noise_currculum=False)
        env_ns['_compute_reward'](env)
        obs = {'state': torch.zeros(1, 1)}
        row = transition_ns['make_transition'](obs, torch.zeros(1,29), obs,
            env.rew_buf, torch.zeros(1,dtype=torch.bool), torch.zeros(1,dtype=torch.bool),
            {}, env.extras, torch.zeros(1,dtype=torch.long))
        agent = NS(device='cpu',cfg=NS(train=NS(batch_size=1),
            aux_rewards=['penalty_ankle_roll'],aux_rewards_scaling={'penalty_ankle_roll':-4.}))
        scope = dict(torch=torch,self=agent,train_batch=row,metrics={})
        exec(aux_code,scope)
        actual = float(scope['aux_reward'].item())
        expected = -4. * .01
        assert abs(actual-expected)<1e-8
        fixtures.append(dict(dt=dt, raw_ankle_square=float(raw.item()),
            main_env_integrated_reward=float(env.rew_buf.item()),
            replay_aux_raw=float(row['aux_rewards']['penalty_ankle_roll'].item()),
            actual_original_aux_sum_before_normalizer=actual,
            hypothetical_dt_scaled_aux_sum=actual*dt,
            ratio_to_dt_scaled=1/dt))

    original_train = cfg['agent']['train']
    assert cfg['agent']['model']['actor_std'] == mcfg['actor_std']
    source_model_diff = {k:[cfg['agent']['model'].get(k),mcfg.get(k)]
                        for k in set(cfg['agent']['model'])|set(mcfg)
                        if cfg['agent']['model'].get(k)!=mcfg.get(k)}
    result = dict(status='BLOCKED_ORIGINAL_AUX_NOT_DT_INTEGRATED_EXECUTION_LOCK_A4',
        scope='CPU exact original arithmetic and file/header audit; no model forward, physics, or update',
        weights_sha256=weight_sha, weights_bytes=weights.stat().st_size,
        config_sha256=sha(cfg_path), source_training_model_vs_model_config=source_model_diff,
        training_config=original_train, original_aux_names=cfg['agent']['aux_rewards'],
        original_aux_scaling=cfg['agent']['aux_rewards_scaling'],
        model_actor_std=mcfg['actor_std'],model_amp=mcfg['amp'],
        init_observation_keys=list(init['obs_space']['spaces']), action_dim=init['action_dim'],
        modules=modules, files=[dict(path=str(p),sha256=sha(p)) for p in files],
        fixtures=fixtures, normalizer='scale-only EMA follows raw weighted sum; it is not a dt integrator',
        mathematical_restatement='original aux = sum(c_i * raw_i), contact = -dt * c_contact',
        blocked_reason='Execution lock A4 requires stop/report if original reward unit is not dt-integrated. '
            'No relative-scale or normalizer remedy is silently chosen.',
        real_physics_ticks=0,real_transitions=0,agent_updates=0,formal_N_locked=False,
        wall_s=time.monotonic()-t0)
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','weights_sha256','config_sha256','source_training_model_vs_model_config','fixtures','wall_s']}),flush=True)


if __name__=='__main__':
    run()
