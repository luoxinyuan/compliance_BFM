import unittest,torch,copy
from research.interaction_steering.contracts import CausalObservation,Guidance,Tier
from research.interaction_steering.executor import direct_command,action_to_pd
from research.interaction_steering.source_adapter import ProgressToLatent,ProgressMemory
from research.interaction_steering.guidance import ForwardAdmittance,AdmittanceParameters
from research.interaction_steering.backends import pbfm_velocity_command,UnqualifiedBackend
from research.interaction_steering.tasks import require_shared_prefix,StateReference

class Contracts(unittest.TestCase):
    def observation(self):
        return dict(q=(0.,)*29,dq=(0.,)*29,gravity=(0.,0.,-1.),angular_velocity=(0.,)*3,action_history=((0.,)*29,)*4,force_history=((1.,0.,0.),)*25,last_z=(1.,)*256,contact_link='right_wrist_yaw_link',contact_local_point=(0.,)*3,known_mode='forward_guidance')
    def test_no_hidden_inputs(self):
        for key in ('anchor','partner_velocity','time','active','task_id','future','search_phase'):
            x=self.observation();x[key]=0
            with self.assertRaises(TypeError):CausalObservation.from_mapping(x)
    def test_prefix_and_causal_guidance(self):
        x=CausalObservation.from_mapping(self.observation());g=ForwardAdmittance(AdmittanceParameters(10.,20.,.22))
        a,mem=g.step(x,0.);b,_=g.step(copy.deepcopy(x),0.)
        self.assertEqual(a,b);self.assertEqual(a.tier,Tier.CS)
        require_shared_prefix([self.observation()],[self.observation()],[a],[b])
        with self.assertRaises(ValueError):require_shared_prefix([1],[2],[a],[b])
    def test_zero_and_nonfinite_latent(self):
        s=torch.ones(1,256);self.assertTrue(torch.equal(direct_command(torch.zeros_like(s),s),s))
        with self.assertRaises(ValueError):direct_command(s*float('nan'),s)
    def test_progress_is_own_memory(self):
        s=torch.ones(1,256);b=s.repeat(4,1);b[:,1]=torch.arange(4.)
        t=ProgressToLatent(b,s,1.2,.214,'a'*64);m=ProgressMemory(.4)
        z,n=t.map(Guidance(.15,0.,0.,True,Tier.CS),m)
        self.assertEqual(m.phase,.4);self.assertGreater(n.phase,.4);self.assertAlmostEqual(float(z.norm()),16.,places=4)
        z0,n0=t.map(Guidance(0.,0.,0.,True,Tier.CS),m);self.assertTrue(torch.equal(z0,s));self.assertEqual(n0.phase,m.phase)
        with self.assertRaises(ValueError):t.map(Guidance(.1,.1,0.,True,Tier.CS),m)
    def test_zero_velocity_not_invalid(self):
        h=torch.tensor([[1.,0.]])
        yes=pbfm_velocity_command(torch.zeros(1,2),h,torch.ones(1,1))
        no=pbfm_velocity_command(torch.zeros(1,2),h,torch.zeros(1,1))
        self.assertEqual(yes.tolist(),[[0.,0.,1.]]);self.assertEqual(no.tolist(),[[0.,0.,0.]])
    def test_yaw_coordinates(self):
        y=pbfm_velocity_command(torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]]),torch.ones(1,1))
        self.assertEqual(y.tolist(),[[0.,-1.,1.]])
    def test_action_coordinates(self):
        a=torch.tensor([[2.,-.1]]);q=action_to_pd(a,torch.zeros(1,2),torch.ones(1,2)*2)
        self.assertTrue(torch.equal(q,torch.tensor([[2.,-.2]])))
    def test_backend_and_state_fail_closed(self):
        with self.assertRaises(RuntimeError):UnqualifiedBackend('P-BFM','weights/physics not qualified').command()
        with self.assertRaises(ValueError):StateReference('p','s','a'*64,'x',False,'old',True)

if __name__=='__main__':unittest.main()
