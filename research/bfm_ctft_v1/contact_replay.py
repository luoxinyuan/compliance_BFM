"""Actual S0 evidence to disposable joint-qualification replay, not formal data.

Contact rows contain executed actions and causal observations, not searches or
counterfactual labels. CPR sequences come only from actual native g0 executions.
"""
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from agent import ACTOR_KEY, CRITIC_KEY


def cat(rows):
    def combine(xs):
        if isinstance(xs[0], dict):
            return {k: combine([v[k] for v in xs]) for k in xs[0]}
        return torch.cat(xs)
    return combine(rows)


class Sequences:
    def __init__(self, groups):
        self.groups = groups

    def sample(self, n):
        assert n % 8 == 0
        rows = []
        for _ in range(n//8):
            g = self.groups[int(torch.randint(len(self.groups), ()).item())]
            start = int(torch.randint(len(g)-7, ()).item())
            rows.extend(g[start:start+8])
        c = cat(rows)
        return dict(observation=c['observation'], next=dict(observation=c['next']['observation']))


def build_buffers(root, native_coefficients):
    import mujoco
    from features import ForceHistory, actor_fields
    from humanoidverse.agents.buffers.transition import DictBuffer
    root = Path(root)
    stage = root.parent
    m = mujoco.MjModel.from_xml_path(str(root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    d = mujoco.MjData(m)
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    rows, groups, provenance = [], [], []
    for case in [1, 2]:
        path = stage/'runs/s0-native-sources-03'/f'source-{case:02d}.pkl.gz'
        with gzip.open(path, 'rb') as f:
            x = pickle.load(f)
        assert x['boundary'] is None
        cache = {}

        def observation(i):
            if i not in cache:
                state = x['states'][i]['mjdata_arrays']
                d.qpos[:] = state['qpos']
                d.qvel[:] = state['qvel']
                mujoco.mj_forward(m, d)
                point = d.xpos[wrist].copy()
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, point, wrist)
                prior = x['physics'][(i-10)*4:i*4:4]
                assert len(prior) == 10 and all(np.linalg.norm(v['force']) == 0 for v in prior)
                history = ForceHistory(np.array([np.array(v['pelvis_matrix']).T@np.array(v['force']) for v in prior]))
                extra = actor_fields(force_history=history, known_mount=[0., 0., 0.],
                    wrist_world=point, root_world=d.qpos[:3], wrist_velocity_world=jac@d.qvel,
                    root_velocity_world=d.qvel[:3], world_from_body=d.xmat[pelvis].reshape(3,3), public_mode=0)
                obs = {k:v.detach().cpu().clone() for k,v in x['controls'][i]['bfm_observation'].items()}
                obs[ACTOR_KEY] = torch.tensor(extra[None])
                obs[CRITIC_KEY] = torch.zeros(1, 9)
                cache[i] = obs
            return cache[i]

        group = []
        for i in range(25, 625):
            c = x['controls'][i]
            raw = x['states'][i+1]['base']['extras']['aux_rewards']
            reward = sum(float(raw[k].detach().cpu().reshape(-1)[0])*v for k,v in native_coefficients.items())
            row = dict(observation=observation(i), action=c['action'].detach().cpu().clone(), z=c['z'].detach().cpu().clone(),
                next=dict(observation=observation(i+1), terminated=torch.zeros(1,1,dtype=torch.bool), truncated=torch.zeros(1,1,dtype=torch.bool)),
                aux_rewards=dict(ctft_total_native=torch.tensor([[reward]],dtype=torch.float32)))
            group.append(row)
        groups.append(group)
        rows.extend(group)
        provenance.append(dict(kind='actual_native_g0', ancestor=x['case']['ancestor'], rows=len(group),
            file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del x, cache
    for name in ['force', 'no_force']:
        path = stage/'runs/s0-contact-interface-01'/(name+'.pkl.gz')
        with gzip.open(path, 'rb') as f:
            x = pickle.load(f)
        assert x['boundary'] is None
        controls = x['controls']

        def obs(c):
            o = {k:v.detach().cpu().clone() for k,v in c['observation'].items()}
            o[ACTOR_KEY] = torch.tensor(c['actor46'][None])
            o[CRITIC_KEY] = torch.tensor(c['privileged9'][None])
            return o

        for i, c in enumerate(controls[:-1]):
            assert not c['terminated'] and abs(controls[i+1]['time_s']-c['time_s']-.02)<1e-7
            cost = c['contact_reward']
            assert cost['unit_contract'] == 'A1_native_aux_per_step_and_separate_physical_integral'
            reward = sum(c['native_aux_raw'][k]*v for k,v in native_coefficients.items()) + cost['contact_native_reward']
            rows.append(dict(observation=obs(c), action=c['action'].detach().cpu().clone(), z=c['z'].detach().cpu().clone(),
                next=dict(observation=obs(controls[i+1]), terminated=torch.tensor([[c['terminated']]]), truncated=torch.tensor([[c['truncated']]])),
                aux_rewards=dict(ctft_total_native=torch.tensor([[reward]],dtype=torch.float32))))
        provenance.append(dict(kind='actual_contact_g1_'+name, independent_physical_parent='same_standing_prefix_one_parent',
            file=path.name, rows=len(controls)-1, final_row_omitted_no_next_observation=True,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del x
    replay = DictBuffer(capacity=4096, device='cpu')
    replay.extend(cat(rows))
    return dict(train=replay, expert_slicer=Sequences(groups)), groups, provenance
