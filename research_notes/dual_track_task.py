"""Approved native-source task and common actual-native-tick physical cost.

No learned motion selector, actor task-ID or future partner input. Shared model
and physics source remain read only. New terminal handling preserves the first
unsafe physical state and exposes the real successor before any manual reset.
"""
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from native_env import Task as NativeTask, spec as native_spec
from locomotion_rollout import spherical, UnsafePhysics
from dual_track_fb_extension import normalize_contact, CONTACT_KEY

CFG = json.loads(Path(__file__).with_name('dual_track_formal_run_v1.json').read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def qtime(t):
    return round(float(t) * 10) / 10


def sample_spec(rng, source=None, family=None):
    source = source or str(rng.choice(list(CFG['source_probs']), p=list(CFG['source_probs'].values())))
    family = family or str(rng.choice(list(CFG['request_probs']), p=list(CFG['request_probs'].values())))
    until = qtime(rng.uniform(*CFG['source_until'][source]))
    profile = dict(arm_fraction=float(rng.uniform(.85, 1.)), pulse_fraction=float(rng.uniform(.02, .08)),
                   pulse_duration=float(rng.uniform(.3, .8)), phase_offset=float(rng.uniform(-.1, .1)),
                   phase_rate=float(rng.uniform(.9, 1.1)))
    # Standing profiles finish the bounded native pulse before handoff.
    profile['pulse_duration'] = min(profile['pulse_duration'], until - 3.05 - .3)
    onset = qtime(until + rng.uniform(*CFG['onset_delay']))
    family_range = 'long' if family == 'stop' else family
    if family in ('zero', 'none'):
        distance, ramp = 0., 4.
    else:
        distance = float(rng.uniform(*CFG['distance_ranges'][family_range]))
        velocity = (float(rng.choice(CFG['long_speeds'])) if family_range == 'long'
                    else float(rng.uniform(*CFG['speed_ranges'][family_range])))
        ramp = distance / velocity
    stop = qtime(onset + rng.uniform(.35, .75) * ramp) if family == 'stop' else None
    s = native_spec(distance, ramp, onset, source, until, family != 'none', stop, 10.)
    s['deceleration'] = float(rng.uniform(*CFG['deceleration'])) if stop is not None else 0.
    end_move = stop + s['deceleration'] if stop is not None else onset + ramp
    s.update(release=end_move + float(rng.uniform(*CFG['hold'])), demand_family=family,
             source_profile=profile)
    s['end'] = s['release'] + CFG['recovery']
    source_id = digest(dict(source=source, until=until, profile=profile))
    partner_id = digest({k: s[k] for k in ('distance', 'ramp', 'onset', 'release', 'end', 'deceleration')})
    if stop is not None:
        partner_id = digest(dict(partner=partner_id, stop=stop))
    parent = digest(dict(source=source_id, partner=partner_id))
    s.update(parent_id=parent, source_parent_id=source_id, partner_parent_id=partner_id,
             common_prefix_parent_id=parent, requested_average_speed=distance / ramp)
    return s


def parent_manifests():
    result = {}
    for split, seed, repeats in [('dev', CFG['dev_parent_seed'], 1), ('sealed', CFG['sealed_parent_seed'], 2)]:
        rng = np.random.default_rng(seed)
        rows = []
        for source in CFG['source_probs']:
            for family in ['near', 'near', 'mid', 'mid', 'long', 'long', 'stop', 'stop', 'zero', 'none'] * repeats:
                s = sample_spec(rng, source, family)
                row = {k: s[k] for k in ('parent_id', 'source_parent_id', 'partner_parent_id', 'common_prefix_parent_id')}
                row.update(source=source, family=family, spec=s)
                if family == 'stop':
                    continued = copy.deepcopy(s)
                    del continued['stop_time']
                    continued['deceleration'] = 0.
                    continued['release'] = s['onset'] + s['ramp'] + (s['release'] - s['stop_time'] - s['deceleration'])
                    continued['end'] = continued['release'] + CFG['recovery']
                    continued['demand_family'] = 'continue_pair'
                    row['continued_spec'] = continued
                rows.append(row)
        result[split] = rows
    from dual_track_admission import assert_parent_separation
    assert_parent_separation(result)
    return result


def native_terms(rows, qpd, previous_qpd, spec):
    """Original cost coefficients/scales, resolved at each executed native control.

    Native-time resolution is shared by both tracks; original archived100ms
    aggregation is preserved separately for diagnostics and is not rewritten.
    """
    assert rows
    force = np.mean([len(r['contacts']) * sum(np.dot(c['force'], c['force']) for c in r['contacts']) / 25 for r in rows])
    err2 = np.mean([np.mean([np.sum((np.array(c['anchor']) - c['point'])**2) for c in r['contacts']]) if r['active'] else 0 for r in rows])
    uncapped = np.mean([np.mean([min(np.log1p(np.dot(c['raw_force'], c['raw_force']) / 25), 8) for c in r['contacts']]) for r in rows])
    rv = np.mean([np.mean([min(np.sum((np.array(c['anchor_velocity']) - c['velocity'])**2) / .25, 16) for c in r['contacts']]) if r['active'] else 0 for r in rows])
    angles = []
    for r in rows:
        R = np.asarray(r['pelvis_matrix']); yaw = math.atan2(R[1, 0], R[0, 0])
        a = math.atan2(math.sin(r['target_yaw'] - yaw), math.cos(r['target_yaw'] - yaw))
        angles.append(min(a*a / math.radians(20)**2, 9) * .35 * min(spec['distance'] / .8, 1) if r['active'] else 0.)
    vroot = np.asarray(rows[-1]['qvel'][:3])
    return dict(force=1.5*force, tracking=.15*min(err2/.04, 16), uncapped=.15*uncapped,
        relative_velocity=.1*rv, heading=float(np.mean(angles)),
        slip=.05*min((max(r['contact_tangent_speed_max_mps'] for r in rows)/.15)**2, 16),
        tilt=.1*min((max(0, rows[-1]['tilt_deg']-15)/15)**2, 16),
        continuity=.05*min(float(np.mean((np.asarray(qpd)-previous_qpd)**2))/.01, 16), soft_prior=0.,
        idle=.2*min(float(np.dot(vroot,vroot))/.04,16) if not rows[-1]['active'] else 0.)


class Task(NativeTask):
    def __init__(self, *args):
        self.stage_boundary = None
        self.allow_native_reset = True
        self.native_info = None
        self.inject_terminal_tick = None
        super().__init__(*args)
        base = self.e.base
        original_reset, original_check = base.reset_envs_idx, base._check_termination

        def reset_selected(ids, *a, **kw):
            if self.allow_native_reset or len(ids) == 0:
                return original_reset(ids, *a, **kw)
            # Real terminal observation/history is captured by the existing base
            # pipeline. Delay reset until this task explicitly starts a new episode.
            self.native_reset_deferred = True

        def check():
            original_check()
            if self.stage_boundary is not None:
                base.reset_buf[:] = 1
                base.time_out_buf[:] = False

        base.reset_envs_idx, base._check_termination = reset_selected, check
        import clean_env

        def advance(env, action):
            obs, _, terminated, truncated, info = env.wrapped.step(action, to_numpy=False)
            terminal = bool(torch.as_tensor(terminated).any())
            limit = bool(torch.as_tensor(truncated).any()) and not terminal
            if terminal or limit:
                assert 'final_observation' in info, 'Missing actual pre-reset terminal observation'
                obs = info['final_observation']
            env.obs = obs
            env.steps += 1
            env.previous_action = action.clone()
            env.history = np.roll(env.history, -1, axis=0); env.history[-1] = env.tether.force
            self.native_info = dict(terminated=terminal, truncated=limit,
                final_observation={k:v.detach().clone() for k,v in obs.items()} if terminal or limit else None)
            return terminal or limit or env.failed
        clean_env.advance = advance

    def _wrap_physics(self):
        fn = self.e.sim.simulate_at_each_physics_step
        if getattr(fn, '_dual_track_terminal', False):
            return
        def callback():
            if self.stage_boundary is not None:
                return
            try:
                fn()
                if self.inject_terminal_tick is not None and self.e.tether.time >= self.inject_terminal_tick:
                    self.stage_boundary = dict(kind='injected_terminal_test', time=float(self.e.tether.time))
            except UnsafePhysics as exc:
                # The original first-unsafe physical row and solver state are
                # retained. Remaining ticks of this control are NOT simulated.
                self.stage_boundary = dict(kind='physics_review', time=float(self.e.tether.time), physics=exc.row)
        callback._dual_track_terminal = True
        self.e.sim.simulate_at_each_physics_step = callback

    def prepare(self):
        out = super().prepare()
        if getattr(self, 'geometry_enabled', False):
            self._wrap_physics()
        return out

    def sample_spec(self):
        return sample_spec(self.rng)

    def native(self, name, t):
        profile = self.spec.get('source_profile')
        if profile is None:
            return super().native(name, t)
        if name == 'standing':
            u = (t - 3.05) / profile['pulse_duration']
            a = profile['pulse_fraction'] * math.sin(math.pi*u)**2 if 0 < u < 1 else 0.
            return spherical(self.e.standing, self.native_z['goal_arm5'], a)
        if name == 'goal_arm5':
            u = float(np.clip((t-3.)/1.5, 0, 1)); u = u*u*(3-2*u)
            return spherical(self.e.standing, self.native_z[name], profile['arm_fraction']*u)
        from m9_cyclic import sample
        u = float(np.clip((t-3.)/1.5, 0, 1)); u = u*u*(3-2*u)
        phase = max(0.,t-4.5)/self.refs['straight']['period']*profile['phase_rate']+profile['phase_offset']
        return spherical(self.e.standing, sample(self.refs['straight'],phase),u)

    def reset(self, *a, **kw):
        # The archived physical reset legitimately seeds its native environment0.
        # Do not let that overwrite the NEW learner/replay/exploration RNG streams.
        external_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        self.stage_boundary = None; self.native_info = None; self.native_reset_deferred = False
        self.allow_native_reset = True
        try:
            return super().reset(*a, **kw)
        finally:
            self.allow_native_reset = False
            random.setstate(external_rng[0]);np.random.set_state(external_rng[1])
            torch.set_rng_state(external_rng[2]);torch.cuda.set_rng_state_all(external_rng[3])

    def contact_vector(self):
        R = self.e.sim.data.xmat[1].reshape(3,3)
        point, velocity = self.read_point_state(); t = self.e.tether
        history = np.zeros((25,3)); rows=list(self.h.rows)[-25:]
        if rows: history[-len(rows):]=[r['force_world'] if r['measurement_valid'] else [0,0,0] for r in rows]
        return normalize_contact(mount=[1,1,0,*t.cfg.local_point],force=t.force,
            error=t.anchor-point if t.active else np.zeros(3),
            relative_velocity=t.partner_velocity-velocity if t.active else np.zeros(3),
            point_relative_root=point-self.e.sim.data.qpos[:3],
            partner_velocity=t.partner_velocity if t.active else np.zeros(3),active=t.active,
            force_history=history,world_to_body=R.T)

    def extended_observation(self):
        self.prepare()
        return dict(self.e.obs, **{CONTACT_KEY:torch.as_tensor(self.contact_vector(),device=self.e.standing.device)[None]})

    def step(self, *a, **kw):
        self._wrap_physics()
        start = len(self.physics); controls = len(self.controls); old_q = self.lastq[0].cpu().numpy().copy()
        x, old_reward, done, st = super().step(*a, **kw)
        rows=self.physics[start:]; cs=self.controls[controls:]
        assert rows and cs
        rewards=[]; chunks=[]
        for i,c in enumerate(cs):
            end = cs[i+1]['time'] if i+1<len(cs) else float(self.e.tether.time)+1e-8
            rr=[r for r in rows if c['time']-1e-8<=r['force_time']<end-1e-8]
            assert rr, (c['time'], end, [r['force_time'] for r in rows])
            terms=native_terms(rr,c['q_PD'],old_q,self.spec); dt=len(rr)*.005
            reward=-dt*sum(terms.values()); old_q=np.asarray(c['q_PD'])
            c.update(reward_terms=terms,reward_integral=reward,actual_duration_s=dt)
            rewards.append(reward);chunks.append(rr)
        assert sum(map(len,chunks))==len(rows)
        terminated=bool(self.boundary)
        truncated=bool(done and not terminated)
        if terminated:
            remaining=max(0,math.ceil((self.spec['end']-self.e.tether.time)/.02))
            tail=10+.2*(1-CFG['gamma_native']**remaining)/(1-CFG['gamma_native'])
            rewards[-1]-=tail; cs[-1]['absorbing_failure_penalty']=tail
        reward=sum(rewards);self.episode_return+=reward-old_reward
        if self.stage_boundary is not None:self.boundary=copy.deepcopy(self.stage_boundary)
        st.update(reward=reward,legacy_reward_not_optimized=old_reward,terminated=terminated,truncated=truncated,
                  native_rewards=rewards,actual_duration_s=len(rows)*.005,boundary=self.boundary,
                  source=self.spec['source'],family=self.spec.get('demand_family','unknown'),
                  native_terms=[c['reward_terms'] for c in cs])
        self.stats[-1]=st
        return x,reward,done,st
