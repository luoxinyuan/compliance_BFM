import inspect,json,unittest
from pathlib import Path
from alpha_memory import ForceMemory

class MemoryTests(unittest.TestCase):
    def test_only_force_input(self):
        self.assertEqual(list(inspect.signature(ForceMemory.append).parameters),['self','measured_force_xyz'])
    def test_no_clock_in_padding(self):
        a,b=ForceMemory(),ForceMemory()
        for _ in range(147):b.append([0,0,0])
        for i in range(40):self.assertEqual(a.append([i/40,0,0]),b.append([i/40,0,0]))
    def test_causality_and_bound(self):
        g=ForceMemory(4);self.assertEqual(g.append([1,2,3]),[[0,0,0]]*3+[[1,2,3]])
        for i in range(5):g.append([i,0,0])
        self.assertEqual(list(g.values),[[i,0,0] for i in range(1,5)])
        with self.assertRaises(ValueError):g.append([float('nan'),0,0])
    def test_sealed_protocol(self):
        p=json.loads(Path(__file__).with_name('alpha_student_protocol.json').read_text())
        self.assertEqual(p['physical']['new_final_test']['onset'],5.4)
        self.assertNotIn('heldout',p['training_conditions']);self.assertNotIn('final_new',p['training_conditions'])
        self.assertEqual(p['optimization']['full_data_updates'],2000)
        self.assertEqual(p['memory']['force_samples'],320)

if __name__=='__main__':unittest.main()
