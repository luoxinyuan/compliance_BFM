"""Complete logical v2 checkpoints, lossless deltas and exact actual-action replay.
No copying qpos alone or substituting action predictions for executed actions.
"""
import os,json,copy
from pathlib import Path
import torch
from . import delta_tree
from .journal import dump,read
from .checkpoint import sha,jsonwrite
from dual_track_task import global_rng,restore_global_rng
from diagnostic_frozen import portable
from diagnostic_fb_resume import tree_digest
from f1_mjdata_contract import metadata,canonical,exact

LISTS=('rows','native','physical_stream')
OWN=('spec_id','reset_cursor','next_cursor','option','option_age','command_slot','offset','macro_start','macro_target','motion_phase','motion_steps','phase','goal_stream','goal_changed','handoff','goal_mode','goal_world','instructions','episodes','source_cache')
class Journal:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(exist_ok=True,parents=True);self.previous=None;self.previous_name=None;self.segments=[];self.lengths={k:0 for k in LISTS}
    def save(self,index,agent,env,runtime,reason='complete_block'):
        dest=self.root/f'block-{index}';assert not dest.exists();tmp=self.root/f'.pending-{index}-{os.getpid()}';tmp.mkdir()
        t=env.task
        own={k:copy.deepcopy(getattr(env,k)) for k in OWN}
        core=dict(format='EI-v2-lossless-v1',agent=agent.state(),own=own,physical=portable(t),mj_metadata=metadata(t.e.sim.model,t.e.sim.data),rng=global_rng(),task_rng=copy.deepcopy(t.rng.bit_generator.state),physics_rng=copy.deepcopy(t.physics_rng),source_sha=t.prefix_data['audit']['sha256'],proxy_previous_z=env.proxy.previous_z,proxy_actions=env.proxy.actions,block=index,runtime=runtime,actual_raw_target=getattr(env,'raw_target',None),stochastic=env.proxy.stochastic)
        cm,cpu=delta_tree.dump(tmp/'core.delta',core,self.previous)
        d={k:getattr(env,k)[self.lengths[k]:] for k in LISTS};dm=dump(tmp/'data.ptz',d);read(tmp/'data.ptz',dm)
        segment=dict(path=dest.name+'/data.ptz',**dm,counts={k:len(v) for k,v in d.items()})
        m=dict(format='EI-v2-lossless-v1',complete=True,block=index,reason=reason,arm=agent.arm,runtime=runtime,previous=self.previous_name,files={'core.delta':cm,'data.ptz':dm},segments=self.segments+[segment],lengths={k:len(getattr(env,k)) for k in LISTS},actual_50Hz=env.next_cursor,base_sha='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361')
        jsonwrite(tmp/'manifest.json',m);fd=os.open(tmp,os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,dest)
        pointer=self.root/'latest.pending';jsonwrite(pointer,dict(package=dest.name,manifest_sha256=sha(dest/'manifest.json')));os.replace(pointer,self.root/'latest.json');fd=os.open(self.root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        self.previous=cpu;self.previous_name=dest.name;self.segments=m['segments'];self.lengths=m['lengths'];return m

def load(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());assert m['format']=='EI-v2-lossless-v1' and m['complete'];chain=[];cursor=path;seen=set()
    while True:
        assert cursor.name not in seen;seen.add(cursor.name);mm=json.loads((cursor/'manifest.json').read_text());assert mm['complete'];chain.append((cursor,mm))
        if mm['previous'] is None:break
        assert Path(mm['previous']).name==mm['previous'];cursor=path.parent/mm['previous']
    state=None
    for p,mm in reversed(chain):state=delta_tree.load(p/'core.delta',mm['files']['core.delta'],state)
    for k in LISTS:state[k]=[]
    for seg in m['segments']:
        rel=Path(seg['path']);assert len(rel.parts)==2 and rel.parts[0].startswith('block-')
        part=read(path.parent/rel,seg)
        for k in LISTS:assert len(part[k])==seg['counts'][k];state[k].extend(part[k])
    for k in LISTS:assert len(state[k])==m['lengths'][k]
    return state

def restore(path,agent,env,out):
    s=load(path);own=s['own'];env.source_cache=copy.deepcopy(own['source_cache']);env.phase=own['phase'];env.reset(own['spec_id']);assert env.task.prefix_data['audit']['sha256']==s['source_sha']
    # Replay only this actual episode using its exact submitted command chunk plan.
    episode=own['reset_cursor'];rows=[r for r in s['rows'] if r['episode']==episode]
    # Instructions are appended per actual collection chunk, retain episode identity.
    ins=[r for r in own['instructions'] if r['episode']==episode]
    assert sum(r['n'] for r in ins)==len(rows)
    env.proxy.replay=iter(rows);env.agent=agent;env.proxy.agent=agent;env.reset_cursor=episode
    for cmd in ins:
        env.replay_option=cmd['option'];env.phase=cmd['phase']
        assert env.offset==cmd['offset']
        env.collect(cmd['n'])
        assert torch.equal(env.instructions[-1]['raw'],cmd['raw'])
    try:next(env.proxy.replay);raise AssertionError('unreplayed actual actions')
    except StopIteration:pass
    env.proxy.replay=None;env.replay_option=None
    now=portable(env.task);old=s['physical'];meta=metadata(env.e.sim.model,env.e.sim.data)
    ca=canonical(now,meta)['mjdata_arrays'];cb=canonical(old,s['mj_metadata'])['mjdata_arrays']
    # Same already-proven replay-only allocation workspaces as original checkpoint.restore.
    padding=('efc_J_colind','efc_J_rowsuper','efc_JT')
    assert exact(meta,s['mj_metadata'])
    assert exact({k:v for k,v in ca.items() if k not in padding},{k:v for k,v in cb.items() if k not in padding}),'replayed dynamics/solver differs'
    assert exact(now['history'],old['history']),'BFM real history differs'
    for key in padding:
        a=getattr(env.e.sim.data,key);b=old['mjdata_arrays'][key];assert a.shape==b.shape and a.dtype==b.dtype;a[...]=b
    from rolling_snapshot import put
    from .restore_fields import restore_value,MISSING
    for obj,key in [(env.task,'task'),(env.e,'env'),(env.e.base,'base'),(env.e.tether,'tether'),(env.task.h,'force_history')]:
        put(obj,{k:restore_value(v,getattr(obj,k,MISSING),key+'/'+k) for k,v in old[key].items()})
    for k,v in own.items():setattr(env,k,copy.deepcopy(v))
    for k in LISTS:setattr(env,k,s[k])
    if s['actual_raw_target'] is not None:env.raw_target=s['actual_raw_target'].to(agent.device)
    for k in ('macro_start','macro_target'):
        if getattr(env,k) is not None:setattr(env,k,getattr(env,k).to(agent.device))
    env.proxy.slot=env.offset;env.proxy.start=env.macro_start;env.proxy.target=env.macro_target;env.proxy.previous_z=s['proxy_previous_z'];env.proxy.actions=s['proxy_actions'];env.proxy.stochastic=s['stochastic']
    env.task.physics_rng=copy.deepcopy(s['physics_rng']);env.task.rng.bit_generator.state=copy.deepcopy(s['task_rng']);restore_global_rng(s['rng']);agent.load(s['agent'])
    actual=canonical(portable(env.task),metadata(env.e.sim.model,env.e.sim.data));assert exact(actual,canonical(old,s['mj_metadata'])),'full effective task/physical/history differs'
    assert tree_digest(agent.state())==tree_digest(s['agent']);assert exact(global_rng(),s['rng'])
    jsonwrite(Path(out)/'restore-exact.json',dict(model_Adam_rng_exact=True,full_effective_physics_history_exact=True,block=s['block'],actual_rows=len(s['rows']),physical_replay_only_not_training=True))
    return s
