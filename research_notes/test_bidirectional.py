import ast,json,unittest
from pathlib import Path
import torch
from bidirectional_projector import balanced_ids
from contact_projector import ContactGate,Projector
HERE=Path(__file__).parent
class Tests(unittest.TestCase):
    def test_batches_cannot_drop_or_overweight_a_direction(self):
        torch.manual_seed(0)
        for _ in range(100):
            ids=balanced_ids('cpu');self.assertEqual(len(ids),12)
            self.assertEqual(int((ids<60).sum()),6);self.assertEqual(int((ids>=60).sum()),6)
            self.assertEqual(len(set(ids.tolist())),12);self.assertLess(int(ids.max()),120)
    def test_teacher_gate_and_projector_algorithms_unchanged(self):
        old=HERE.parent/'issue21-contact-projector'
        if not old.exists():old=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/trees/issue-21-contact-projector/repo/research_notes')
        for filename,names in [('contact_projector.py',['ContactGate','Projector','deploy_input','constrained']),('ccl02a_teacher.py',['arm_ik','ReleaseFade','audit_episode'])]:
            def funcs(root):return {n.name:ast.dump(n) for n in ast.parse((root/filename).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            a,b=funcs(old),funcs(HERE)
            for name in names:self.assertEqual(a[name],b[name],name)
        a=(old/'ccl02a_teacher.py').read_text();b=(HERE/'ccl02a_teacher.py').read_text()
        start="                    if label=='teacher' and active:"
        end='                    correction=fade.update'
        self.assertEqual(a[a.index(start):a.index(end)],b[b.index(start):b.index(end)])
    def test_protocol_and_schedule_no_eval_leakage(self):
        p=json.loads((HERE/'bidirectional_protocol.json').read_text());tasks=p['tasks']
        self.assertEqual([t['onset_s'] for t in tasks],[1.1,1.1,1.25,1.25])
        self.assertEqual([t['direction'] for t in tasks],[1,-1,1,-1])
        s=(HERE/'ccl02a_teacher.py').read_text()
        self.assertIn("[pairs[('dev-2','teacher')],pairs[('dev-3','teacher')]]",s)
        self.assertNotIn('fit(',s)
        self.assertLess(s.index("(TASKS[1],['student-old'])"),s.index("(TASKS[0],['student'])"))
    def test_force_feature_ablation_does_not_disable_contact_gate(self):
        g=ContactGate();g.update(1.2,[-1.,0.,0.]);h=torch.ones(1,474);fed=h.clone();fed[:,-9:]=0
        self.assertEqual(g.value(1.22),1.);self.assertTrue(torch.equal(h[:,:-9],fed[:,:-9]));self.assertEqual(float(h[:,-9:].sum()),9.)
        self.assertEqual(float(fed[:,-9:].sum()),0.)
if __name__=='__main__':unittest.main()
