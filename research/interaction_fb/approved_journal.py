"""R2 lossless core delta + append-only replay. Every round is independently addressable."""
import os,json,copy
from pathlib import Path
from . import delta_tree
from .journal import dump,read,LISTS
from .checkpoint import sha,jsonwrite
from diagnostic_frozen import portable
from f1_mjdata_contract import metadata
from dual_track_task import global_rng

class Journal:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  self.previous=None;self.previous_name=None;self.segments=[];self.lengths={k:0 for k in LISTS}
 def adopt(self,source):
  import shutil
  source=Path(source);state=load_state(source);m=json.loads((source/'manifest.json').read_text())
  cursor=source;names=set()
  while True:
   mm=json.loads((cursor/'manifest.json').read_text());names.add(cursor.name)
   if mm['previous'] is None:break
   cursor=source.parent/mm['previous']
  names.update(Path(seg['path']).parts[0] for seg in m['segments'])
  for name in sorted(names):
   assert not (self.root/name).exists()
   shutil.copytree(source.parent/name,self.root/name,copy_function=os.link)
  self.previous={k:v for k,v in state.items() if k not in LISTS}
  self.previous_name=source.name;self.segments=m['segments'];self.lengths=m['lengths']
  jsonwrite(self.root/'latest-committed.json',dict(package=source.name,manifest_sha256=sha(self.root/source.name/'manifest.json')))
  return m['round_id']
 def save(self,round_id,agent,env,runtime,reason='complete_block'):
  root=self.root;dest=root/f'round-{round_id}';assert not dest.exists()
  tmp=root/f'.pending-{round_id}-{os.getpid()}';tmp.mkdir()
  t=env.task
  core=dict(version='R2-lossless-v1',agent=agent.state(),commands=env.commands,actions=env.proxy.actions,
   task_specs=env.specs,spec_id=env.spec_id,episodes=env.episodes,cursor=env.next_cursor,
   physical=portable(t),mj_metadata=metadata(t.e.sim.model,t.e.sim.data),rng=global_rng(),
   task_rng=copy.deepcopy(t.rng.bit_generator.state),physics_rng=copy.deepcopy(t.physics_rng),
   round_id=round_id,runtime=runtime,source_sha=t.prefix_data['audit']['sha256'])
  cmeta,cpu=delta_tree.dump(tmp/'core.delta',core,self.previous)
  assert delta_tree.h((tmp/'core.delta').read_bytes())==cmeta['sha256']
  delta={k:getattr(env,k)[self.lengths[k]:] for k in LISTS}
  dmeta=dump(tmp/'data.ptz',delta);read(tmp/'data.ptz',dmeta)
  seg=dict(path=dest.name+'/data.ptz',**dmeta,counts={k:len(v) for k,v in delta.items()})
  manifest=dict(format='R2-lossless-v1',complete=True,round_id=round_id,reason=reason,runtime=runtime,arm=agent.arm,
   previous=self.previous_name,files={'core.delta':cmeta,'data.ptz':dmeta},segments=self.segments+[seg],
   lengths={k:len(getattr(env,k)) for k in LISTS},transitions=env.next_cursor,updates=agent.updates,
   base_sha='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361')
  jsonwrite(tmp/'manifest.json',manifest);fd=os.open(tmp,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  os.rename(tmp,dest);pending=root/'latest-committed.pending'
  jsonwrite(pending,dict(package=dest.name,manifest_sha256=sha(dest/'manifest.json')));os.replace(pending,root/'latest-committed.json')
  fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  self.previous=cpu;self.previous_name=dest.name;self.segments=manifest['segments'];self.lengths=manifest['lengths']
  return manifest

def load_state(path):
 path=Path(path);m=json.loads((path/'manifest.json').read_text());assert m['complete'] and m['format']=='R2-lossless-v1'
 chain=[];cursor=path;seen=set()
 while True:
  assert cursor.name not in seen;seen.add(cursor.name)
  mm=json.loads((cursor/'manifest.json').read_text());assert mm['complete'];chain.append((cursor,mm))
  prev=mm['previous']
  if prev is None:break
  assert Path(prev).name==prev and prev.startswith('round-');cursor=path.parent/prev
 state=None
 for pp,mm in reversed(chain):state=delta_tree.load(pp/'core.delta',mm['files']['core.delta'],state)
 for k in LISTS:state[k]=[]
 for seg in m['segments']:
  rel=Path(seg['path']);assert len(rel.parts)==2 and rel.parts[0].startswith('round-') and rel.parts[1]=='data.ptz'
  part=read(path.parent/rel,seg)
  for k in LISTS:
   assert len(part[k])==seg['counts'][k];state[k].extend(part[k])
 for k in LISTS:assert len(state[k])==m['lengths'][k]
 return state
