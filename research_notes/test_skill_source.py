"""Teacher contract and isolation regressions; no model execution."""
import ast,json,unittest
from pathlib import Path
from skill_source_rollout import progress

class ContractTests(unittest.TestCase):
    def test_schedule_boundaries_and_return(self):
        for name in ['goal_free','goal_contact','standing_contact']:
            for t in [i/200 for i in range(1201)]:self.assertTrue(0<=progress(name,t)<=1)
            self.assertEqual(progress(name,0),0);self.assertEqual(progress(name,6),0)
        self.assertEqual(progress('goal_free',.75),1);self.assertEqual(progress('goal_free',4),0)
        self.assertEqual(progress('goal_contact',4),1);self.assertEqual(progress('goal_contact',5),1)
        self.assertEqual(progress('goal_contact',5.75),0)

    def test_supervisor_has_new_root_and_launcher(self):
        text=Path(__file__).with_name('skill_source_supervisor.py').read_text();tree=ast.parse(text)
        assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUN' for t in n.targets))
        self.assertEqual(assignment.value.right.value,'bridge-v2-owned/runs/issue21-skill-source-20261030A')
        self.assertIn("launcher='skill_source_physics.py'",text)

    def test_protocol_preserves_original_model_and_no_training(self):
        p=json.loads(Path(__file__).with_name('skill_source_protocol.json').read_text())
        self.assertEqual(p['model_sha256'],'33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361')
        self.assertEqual(p['branches'],['goal_free','standing_contact','goal_contact'])
        self.assertTrue(p['no_training']);self.assertEqual(p['physical']['max_force_n'],5)

if __name__=='__main__':unittest.main()
