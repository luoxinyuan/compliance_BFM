"""CPU-only exact serialization, exception, and RNG regression."""
import gzip,hashlib,io,json,random,tempfile,unittest
from pathlib import Path
from frl_fast_archive import installed

class ArchiveTests(unittest.TestCase):
 def test_formats_exact(self):
  objects=[{'中文':[1.234567890123456,-0.,None,True,False,'é\n"',float('inf')], 'nested':{'a':1}},[],{'large':list(range(10000))}]
  original=(gzip.open,json.dump)
  with tempfile.TemporaryDirectory() as folder:
   for level in (1,9):
    for i,data in enumerate(objects):
     for options in ({},{'indent':2,'ensure_ascii':False},{'sort_keys':True,'separators':(',',':')}):
      expected=json.dumps(data,**options).encode('utf-8');rng=random.getstate();p=Path(folder)/f'{level}-{i}.gz'
      with installed(level):
       with gzip.open(p,'wt',encoding='utf-8') as f:json.dump(data,f,**options)
       with gzip.open(p,'rt',encoding='utf-8') as f:self.assertEqual(f.read().encode('utf-8'),expected)
      self.assertEqual(random.getstate(),rng);self.assertEqual((gzip.open,json.dump),original)
 def test_plain_and_default(self):
  class V:pass
  buf=io.StringIO()
  with installed():json.dump({'v':V()},buf,default=lambda x:'converted')
  self.assertEqual(buf.getvalue(),'{"v": "converted"}')
 def test_restore_exception(self):
  original=gzip.open,json.dump
  with self.assertRaises(RuntimeError):
   with installed():raise RuntimeError('probe')
  self.assertEqual((gzip.open,json.dump),original)
 def test_nan_guard_preserved(self):
  with tempfile.TemporaryDirectory() as folder:
   with installed():
    with gzip.open(Path(folder)/'nan.gz','wt') as f:
     with self.assertRaises(ValueError):json.dump({'x':float('nan')},f,allow_nan=False)
if __name__=='__main__':unittest.main()
