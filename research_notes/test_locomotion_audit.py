import ast
import unittest
from pathlib import Path
from locomotion_saved_audit import magnitude_summary


class AuditRegression(unittest.TestCase):
    def test_deployment_target_clips_position_without_mutating_action(self):
        import torch
        from locomotion_rollout import deployed_target
        a=torch.tensor([[2.,-2.,.2]]);copy=a.clone()
        q=deployed_target(a,torch.tensor([[.5,.5,.5]]),torch.tensor([[5.,5.,5.]]),torch.tensor([[-1.,-1.,-1.]]),torch.tensor([[1.,1.,1.]]))
        torch.testing.assert_close(q,torch.tensor([[1.,-1.,.7]]))
        torch.testing.assert_close(a,copy)

    def test_list_and_negative_array_statistics(self):
        self.assertEqual(magnitude_summary([[-2.,2.],[-2.,2.]]),[2.,2.])

    def test_new_output_root_prevents_historical_ledger_write(self):
        tree=ast.parse(Path(__file__).with_name('locomotion_supervisor.py').read_text())
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='RUN' for t in n.targets))
        value=assignment.value.right.value
        self.assertEqual(value,'bridge-v2-owned/runs/issue21-locomotion-audit-20261030A')
        self.assertNotIn('reward-stage',value)


if __name__=='__main__':unittest.main()
