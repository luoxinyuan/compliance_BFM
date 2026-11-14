"""CPU-only synthetic dispatch regression; no physics or learned parameters."""
import ast,tempfile
from pathlib import Path
from types import SimpleNamespace
source=Path(__file__).with_name('warmup_guard.py')
class Base:
 def collect_call(self):self.calls+=1;return self.calls
 def reset(self,i=0):pass
def reference(env,agent,dest,reference_collect=None):
 for _ in range(2):(reference_collect or env.collect_call)()
 return {'failures':[],'quality':{}}
module=ast.parse(source.read_text());module.body=[n for n in module.body if isinstance(n,ast.ClassDef)]
ns=dict(Environment=Base,Path=Path,same_state_reference=reference,actual_release_index=lambda _:None,install_gate=lambda _:None)
exec(compile(module,str(source),'exec'),ns)
with tempfile.TemporaryDirectory() as directory:
 env=ns['GuardedEnvironment']();env.calls=0;env.guard_reference=None;env.guard_output=directory;env.episodes=[]
 env.task=SimpleNamespace(done=False,physics=[{'active':True}],e=SimpleNamespace(tether=SimpleNamespace(time=2)),spec={'release':1})
 env.proxy=SimpleNamespace(agent=None)
 assert env.collect_call()==3
 assert env.calls==3 and len(list(Path(directory).iterdir()))==1
 assert env.collect_call()==4
print('PASS: reference executes base physics dispatch twice, outer call once; one reference directory; no recursion')
