import ast
import unittest
from pathlib import Path
from locomotion_saved_audit import magnitude_summary


class AuditRegression(unittest.TestCase):
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
