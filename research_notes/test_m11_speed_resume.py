import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from omegaconf import OmegaConf
from m11_runtime_speed import install
from m11_resume_artifacts import restore
from m11_selection import sha
class Tests(unittest.TestCase):
 def test_static_config(self):
  cfg=OmegaConf.create({'a':3,'b':'${a}','list':['${a}'],'unused':'${absent_resolver:x}'})
  report=install(SimpleNamespace(base=SimpleNamespace(config=cfg)))
  self.assertEqual(cfg.b,3);self.assertEqual(cfg.list[0],3)
  self.assertFalse(OmegaConf.is_interpolation(cfg,'b'))
  self.assertTrue(OmegaConf.is_interpolation(cfg,'unused'))
 def test_container_alias_with_unused_resolver(self):
  cfg=OmegaConf.create({'robot':{'gain':3,'unused':'${absent_resolver:x}'},'env':{'robot':'${robot}'}})
  install(SimpleNamespace(base=SimpleNamespace(config=cfg.env)))
  self.assertEqual(cfg.env.robot.gain,3)
  self.assertFalse(OmegaConf.is_interpolation(cfg.env,'robot'))
  self.assertTrue(OmegaConf.is_interpolation(cfg.env.robot,'unused'))
 def test_restore_is_independent_and_preserves_selection(self):
  with tempfile.TemporaryDirectory() as d:
   old=Path(d)/'old';new=Path(d)/'new';old.mkdir();new.mkdir()
   (old/'launch.json').write_text(json.dumps(dict(plan_sha256='p',source_sha256='s')))
   (old/'training.jsonl').write_text(json.dumps(dict(batch=1,elapsed_s=12))+'\n')
   (old/'baseline').mkdir();(old/'baseline/summary.json').write_text('[]')
   (old/'best.pt').write_bytes(b'weights')
   sel=dict(metadata=dict(batches=0),checkpoint_sha256=sha(old/'best.pt'),rank=[0,1])
   (old/'selection.json').write_text(json.dumps(sel))
   r,t=restore(old,new,1,'s','p');self.assertEqual(r,sel);self.assertEqual(t,12)
   self.assertNotEqual((old/'training.jsonl').stat().st_ino,(new/'training.jsonl').stat().st_ino)
   (new/'training.jsonl').write_text('new');self.assertIn('batch',(old/'training.jsonl').read_text())
   self.assertEqual(sha(old/'best.pt'),sha(new/'best.pt'))
if __name__=='__main__':unittest.main()
