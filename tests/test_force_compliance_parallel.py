import unittest
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
from humanoidverse.force_compliance.config import ComplianceConfig
from humanoidverse.force_compliance.contacts import CONTACTS, select_contact
from humanoidverse.force_compliance.interaction import Tether
from humanoidverse.force_compliance.environment import ComplianceEnv
from humanoidverse.force_compliance.parallel import merge_rollouts, update, force_level_seeds
from humanoidverse.force_compliance.policy import ForceLatentPolicy


class WholeBodyTests(unittest.TestCase):
    def test_audit_covers_all_force_levels(self):
        for direction in [None, [1., 0., 0.]]:
            cfg = ComplianceConfig(nominal_force_levels=[10.,20.,30.,40.,60.], direction=direction)
            seeds = force_level_seeds(cfg)
            self.assertEqual(seeds, force_level_seeds(cfg))
            levels = []
            for seed in seeds:
                rng = np.random.default_rng(seed)
                if direction is None:
                    rng.normal(size=3)
                levels.append(float(rng.choice(cfg.nominal_force_levels)))
            self.assertEqual(levels, cfg.nominal_force_levels)

    def test_random_contacts_reproducible_uniform_and_config_not_mutated(self):
        cfg = ComplianceConfig(contact_points=CONTACTS)
        rng = np.random.default_rng(17)
        indices = [select_contact(cfg, rng)[0] for _ in range(10000)]
        self.assertTrue(np.all(np.abs(np.bincount(indices)-1000) < 150))
        rng = np.random.default_rng(17)
        self.assertEqual(indices[:20], [select_contact(cfg,rng)[0] for _ in range(20)])
        self.assertEqual(cfg.body, 'right_wrist_yaw_link')
        self.assertEqual(cfg.local_point, [0.,0.,0.])
        with self.assertRaises(ValueError):
            select_contact(cfg,rng,10)

    def test_all_ten_points_apply_correct_generalized_force(self):
        m = mujoco.MjModel.from_xml_path(str(Path(__file__).parents[1]/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
        d = mujoco.MjData(m)
        cfg = ComplianceConfig(contact_points=CONTACTS)
        for i, contact in enumerate(CONTACTS):
            with self.subTest(contact=contact['name']):
                _, _, c = select_contact(cfg,np.random.default_rng(0),i)
                tether = Tether(m,d,c)
                tether.time = 2.
                point,force = tether.measure()
                expected = np.zeros(m.nv)
                mujoco.mj_applyFT(m,d,force,np.zeros(3),point,tether.body,expected)
                before = d.qfrc_smooth.copy()
                wrench = tether.apply()
                mujoco.mj_forward(m,d)
                np.testing.assert_allclose(d.qfrc_smooth-before,expected,atol=1e-7)
                self.assertEqual(np.count_nonzero(np.linalg.norm(d.xfrc_applied,axis=1)),1)
                d.xfrc_applied[tether.body] -= wrench
                mujoco.mj_forward(m,d)

    def test_observation_encodes_contact_and_no_anchor(self):
        env = ComplianceEnv.__new__(ComplianceEnv)
        env.cfg = ComplianceConfig(contact_points=CONTACTS)
        env.model = SimpleNamespace(_normalize=lambda x:x)
        env.keys=['state']; env.obs={'state':torch.zeros(1,2)}
        env.device='cpu';env.history=np.zeros((3,3));env.previous_z=torch.ones(1,4);env.standing=env.previous_z
        env.sim=SimpleNamespace(model=SimpleNamespace(nbody=40))
        for index in range(10):
            _,_,c=select_contact(env.cfg,np.random.default_rng(),index)
            env.tether=SimpleNamespace(cfg=c,active=True,body=5,anchor=np.ones(3))
            env.contact_index=index
            before=env.observation()
            torch.testing.assert_close(before[0,-14:-4],torch.eye(10)[index])
            env.tether.anchor *= 100
            torch.testing.assert_close(before,env.observation())

    def test_gae_does_not_mix_workers_or_cross_reset(self):
        results=[]
        for scale in (1.,10.):
            results.append({'batch':dict(reward=np.array([scale,2*scale],np.float32), value=np.zeros(2,np.float32),
                next_value=np.zeros(2,np.float32), discount=np.ones(2,np.float32),done=np.array([False,True]))})
        batch=merge_rollouts(results,1.)
        torch.testing.assert_close(batch['returns'],torch.tensor([3.,30.,2.,20.]))
        results[0]['batch']['done'][0]=True
        batch=merge_rollouts(results,1.)
        torch.testing.assert_close(batch['returns'],torch.tensor([1.,30.,2.,20.]))

    def test_parallel_ppo_update_finite_and_changes_actor(self):
        torch.manual_seed(0)
        p=ForceLatentPolicy(8,4)
        obs=torch.randn(64,8)
        with torch.no_grad():
            dist=p.distribution(obs);action=dist.sample();logp=dist.log_prob(action).sum(-1)
        batch=dict(obs=obs,action=action,logp=logp,reward=torch.ones(64),returns=torch.randn(64),advantages=torch.randn(64))
        before=p.actor[-1].weight.detach().clone()
        metrics=update(p,torch.optim.Adam(p.parameters(),lr=3e-4),batch)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertGreater(metrics['gradient_steps'],0)
        self.assertFalse(torch.equal(before,p.actor[-1].weight))


if __name__ == '__main__':
    unittest.main()
