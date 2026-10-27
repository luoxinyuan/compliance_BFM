"""CPU tests use actual MuJoCo dynamics, not a fake BFM checkpoint."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from humanoidverse.force_compliance.config import ComplianceConfig
from humanoidverse.force_compliance.environment import ComplianceEnv
from humanoidverse.force_compliance.interaction import Tether, spring_force
from humanoidverse.force_compliance.policy import ForceLatentPolicy, latent_command
from humanoidverse.force_compliance.ppo import advantages, train


class PhysicsTests(unittest.TestCase):
    def test_force_reduces_when_following(self):
        anchor, zero = np.array([.1, 0, 0]), np.zeros(3)
        rigid = spring_force(anchor, zero, zero, zero, 80, 8, 60)
        following = spring_force(anchor, anchor*.8, zero, zero, 80, 8, 60)
        self.assertLess(np.linalg.norm(following), np.linalg.norm(rigid))
        self.assertAlmostEqual(np.linalg.norm(following)/np.linalg.norm(rigid), .2)
        self.assertAlmostEqual(np.linalg.norm(spring_force(anchor*100, zero, zero, zero, 80, 8, 60)), 60)

    def model(self):
        return mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0" timestep="0.002"/><worldbody><body name="hand"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>')

    def test_nominal_force_levels_and_unloading(self):
        m = self.model(); d = mujoco.MjData(m)
        for level in [10., 20., 30., 40., 60.]:
            cfg = ComplianceConfig(body='hand', stiffness=200., nominal_force_levels=[level], direction=[1,0,0])
            tether = Tether(m, d, cfg)
            tether.time = cfg.anchor_delay + cfg.ramp_duration
            np.testing.assert_allclose(tether.measure()[1], [level, 0, 0], atol=1e-8)
            d.qpos[0] += level / cfg.stiffness
            np.testing.assert_allclose(tether.measure()[1], 0, atol=1e-8)
            d.qpos[0] = 0
        for kwargs in [dict(stiffness=0, nominal_force_levels=[10]), dict(nominal_force_levels=[float('nan')]), dict(nominal_force_levels=[61])]:
            with self.assertRaises(ValueError):
                ComplianceConfig(**kwargs)

    def test_offset_point_wrench_equals_mj_applyft(self):
        m = self.model(); d = mujoco.MjData(m)
        d.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        c = ComplianceConfig(body='hand', local_point=[.1, 0, 0], activation_time=0, anchor_delay=0, direction=[1,0,0])
        tether = Tether(m, d, c)
        np.testing.assert_allclose(tether.point_state()[0], [0, .1, 0], atol=1e-8)
        tether.time = 1.
        point, force = tether.measure()
        expected = np.zeros(m.nv)
        mujoco.mj_applyFT(m, d, force, np.zeros(3), point, tether.body, expected)
        wrench = tether.apply()
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.qfrc_applied + d.qfrc_smooth - d.qfrc_bias, expected, atol=1e-7)
        np.testing.assert_allclose(wrench[3:], np.cross(point-d.xipos[tether.body], force))
        self.assertGreater(np.linalg.norm(wrench[3:]), 0)

    def test_physical_force_moves_body_and_anchor_is_independent(self):
        m = self.model(); d = mujoco.MjData(m)
        c = ComplianceConfig(body='hand', activation_time=0, anchor_delay=0, ramp_duration=.2, displacement_min=.1, displacement_max=.1, direction=[1,0,0])
        tether = Tether(m, d, c)
        origin = tether.origin.copy()
        forces = []
        for _ in range(750):
            wrench = tether.apply()
            forces.append(np.linalg.norm(tether.force))
            mujoco.mj_step(m, d)
            d.xfrc_applied[tether.body] -= wrench
            tether.time += m.opt.timestep
        self.assertGreater(d.qpos[0], .08)
        self.assertLess(forces[-1], max(forces)*.05)
        np.testing.assert_array_equal(tether.origin, origin)
        self.assertTrue(tether.active)
        c.enabled = False
        np.testing.assert_array_equal(tether.measure()[1], np.zeros(3))

    def test_real_g1_right_hand_force_enters_physics(self):
        path = Path(__file__).parents[1] / 'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        tether = Tether(m, d, ComplianceConfig(activation_time=0, anchor_delay=0, local_point=[.02, 0, 0]))
        tether.time = 1.
        point, force = tether.measure()
        expected = np.zeros(m.nv)
        mujoco.mj_applyFT(m, d, force, np.zeros(3), point, tether.body, expected)
        mujoco.mj_forward(m, d)
        before = d.qfrc_smooth.copy()
        tether.apply()
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.qfrc_smooth-before, expected, atol=1e-7)
        self.assertGreater(np.linalg.norm(expected), 0)
        mujoco.mj_step(m, d)
        self.assertTrue(np.isfinite(d.qpos).all())

    def test_simulator_warning_is_failure_even_with_finite_state(self):
        path = Path(__file__).parents[1] / 'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'
        m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
        env = ComplianceEnv.__new__(ComplianceEnv)
        env.cfg = ComplianceConfig()
        env.sim = SimpleNamespace(model=m, data=d)
        env.tether = Tether(m, d, env.cfg)
        env.failed = False
        env.sub_rewards, env.sub_force_squared, env.sub_force_peak = [], [], []
        def bad_step():
            mujoco.mj_step(m, d)
            d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
        env._original_step = bad_step
        env._physics_step()
        self.assertTrue(np.isfinite(d.qpos).all())
        self.assertTrue(env.failed)
        np.testing.assert_array_equal(d.xfrc_applied, np.zeros_like(d.xfrc_applied))

    def test_attachment_starts_before_anchor_motion(self):
        m = self.model(); d = mujoco.MjData(m)
        tether = Tether(m, d, ComplianceConfig(body='hand'))
        point, force = tether.measure()
        self.assertTrue(tether.active)
        np.testing.assert_array_equal(force, np.zeros(3))
        tether.time = .5
        tether.measure()
        np.testing.assert_array_equal(tether.anchor, point)
        tether.time = 1.5
        tether.measure()
        self.assertGreater(np.linalg.norm(tether.force), 0)
        tether.cfg.release_time = 2.
        tether.time = 2.
        np.testing.assert_array_equal(tether.measure()[1], np.zeros(3))

    def test_point_velocity_includes_rotation(self):
        m = self.model(); d = mujoco.MjData(m)
        d.qvel[5] = 2
        tether = Tether(m, d, ComplianceConfig(body='hand', local_point=[.1,0,0]))
        np.testing.assert_allclose(tether.point_state()[1], [0,.2,0], atol=1e-8)


class LearningTests(unittest.TestCase):
    def test_latent_norm_zero_force_and_gradients(self):
        dim = 16
        model = SimpleNamespace(cfg=SimpleNamespace(archi=SimpleNamespace(norm_z=True)),
                                project_z=lambda z: dim**.5 * torch.nn.functional.normalize(z, dim=-1))
        standing = model.project_z(torch.randn(1, dim))
        policy = ForceLatentPolicy(12, dim)
        obs = torch.randn(8,12)
        raw = policy.distribution(obs).sample()
        z = latent_command(model, standing, raw, 20., ComplianceConfig())
        torch.testing.assert_close(z.norm(dim=-1), torch.full((8,), dim**.5))
        torch.testing.assert_close(latent_command(model, standing, raw[:1], 0., ComplianceConfig()), standing)
        (-policy.distribution(obs).log_prob(raw).mean() + policy.value(obs).square().mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()))

    def test_duration_discount_and_terminal_mask(self):
        rewards = torch.tensor([1., 2., 3.])
        adv, ret = advantages(rewards, torch.zeros(3), torch.tensor([0., 100., 4.]),
                              torch.tensor([.9**5, .9**2, .9**3]), torch.tensor([False, True, False]), 1.)
        torch.testing.assert_close(ret, torch.tensor([1+(.9**5)*2, 2., 3+(.9**3)*4]))

    def test_ppo_updates_only_high_level(self):
        class Env:
            cfg = ComplianceConfig()
            def observation(self): return torch.ones(1, 7)
            def reset(self): return self.observation()
            def step(self, action):
                return self.observation(), -float(action.square().sum()), False, {'discount': .99**5, 'failed': False}
        policy = ForceLatentPolicy(7, 4)
        old = {k:v.clone() for k,v in policy.state_dict().items()}
        logs = train(Env(), policy, torch.optim.Adam(policy.parameters(), lr=1e-3), iterations=1, rollout=8, epochs=2)
        self.assertTrue(np.isfinite(logs[0]['loss']))
        self.assertTrue(any(not torch.equal(v, old[k]) for k,v in policy.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
