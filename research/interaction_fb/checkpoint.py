"""Immutable round transactions; a pointer can only name a hash-verified package."""
import os,json,hashlib,copy,time
from pathlib import Path
import numpy as np
import torch
from dual_track_task import global_rng,restore_global_rng
from diagnostic_frozen import portable
from diagnostic_fb_resume import tree_digest
from f1_mjdata_contract import metadata,canonical,exact

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync(p):
    with open(p,'rb') as f:os.fsync(f.fileno())
def jsonwrite(p,x):
    p.write_text(json.dumps(x,indent=2));sync(p)
def save(root,round_id,agent,env,runtime,reason='completed_round'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/f'round-{round_id}'
    assert not dest.exists();tmp=root/f'.pending-{round_id}-{os.getpid()}';tmp.mkdir()
    t=env.task;meta=metadata(t.e.sim.model,t.e.sim.data)
    state=dict(version='IFB-T0-v1',agent=agent.state(),rows=env.rows,native=env.native,commands=env.commands,
        actions=env.proxy.actions,spec_id=env.spec_id,episodes=env.episodes,cursor=env.next_cursor,
        physical=portable(t),mj_metadata=meta,rng=global_rng(),task_rng=copy.deepcopy(t.rng.bit_generator.state),
        physics_rng=copy.deepcopy(t.physics_rng),round_id=round_id,runtime=runtime,
        schema='robot frozen + deploy87 + independent successor interaction16',
        source_sha=t.prefix_data['audit']['sha256'],physical_stream=env.physical_stream)
    torch.save(state,tmp/'state.pt');sync(tmp/'state.pt')
    test=torch.load(tmp/'state.pt',map_location='cpu',weights_only=False)
    assert test['round_id']==round_id and len(test['rows'])==len(env.rows)
    manifest=dict(round_id=round_id,reason=reason,complete=True,runtime=runtime,files={'state.pt':dict(sha256=sha(tmp/'state.pt'),bytes=(tmp/'state.pt').stat().st_size)},
        transitions=env.next_cursor,updates=agent.updates,base_sha='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361')
    jsonwrite(tmp/'manifest.json',manifest)
    fd=os.open(tmp,os.O_DIRECTORY);os.fsync(fd);os.close(fd);os.rename(tmp,dest)
    pointer=root/'latest-committed.pending';jsonwrite(pointer,dict(package=dest.name,manifest_sha256=sha(dest/'manifest.json')))
    os.replace(pointer,root/'latest-committed.json');fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return manifest

def restore(path,agent,env):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    assert m['complete']
    for f,v in m['files'].items():assert sha(path/f)==v['sha256'] and (path/f).stat().st_size==v['bytes']
    if m.get('format')=='delta-v1':
        from .journal import load_state
        s=load_state(path)
    else:s=torch.load(path/'state.pt',map_location='cpu',weights_only=False)
    assert s.get('task_specs',env.specs)==env.specs,'Task manifest changed on restore'
    env.reset(s['spec_id']);assert env.task.prefix_data['audit']['sha256']==s['source_sha']
    agent.load(s['agent']);p=env.proxy;p.agent=agent;p.replay_actions=iter(s['actions']);p.actions=[]
    for raw in s['commands']:
        p.slot=0;p.start=env.task.last_command.clone();p.target=raw.cuda()*16
        p.take_step(raw.cuda(),5)
    try:next(p.replay_actions);raise AssertionError('unplayed actual action')
    except StopIteration:pass
    p.replay_actions=None;p.slot=0;p.start=env.task.last_command.clone();p.target=p.start.clone()
    from rolling_snapshot import put
    t=env.task;old=s['physical']
    # Compare actual regenerated physical state before restoring counters/RNG.
    now=portable(t);ma=metadata(t.e.sim.model,t.e.sim.data)
    ca=canonical(now,ma)['mjdata_arrays'];cb=canonical(old,s['mj_metadata'])['mjdata_arrays']
    if not exact(ca,cb):
        differences={}
        for key in ca:
            if not exact(ca[key],cb[key]):
                a,b=ca[key],cb[key]
                differences[key]=dict(shape_now=list(a.shape),shape_saved=list(b.shape),
                    max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape and a.size else None)
        evidence=Path(os.environ['BFM_RECOVERY_OUTPUT'])
        jsonwrite(evidence/'restore-difference.json',dict(arrays=differences,metadata_now=ma,metadata_saved=s['mj_metadata']))
        torch.save(dict(now=now,old=old,controls=t.controls,physics=t.physics),evidence/'restore-difference.pt')
    # These archived arrays must be copied back by replay-only
    # recovery. Return pilot found efc_JT was also omitted. Preserve all stored
    # bytes; no unused-state claim, mask, tolerance or success-gate change.
    restored_arrays=('efc_J_colind','efc_J_rowsuper','efc_JT')
    assert exact({k:v for k,v in ca.items() if k not in restored_arrays},
                 {k:v for k,v in cb.items() if k not in restored_arrays}), 'non-restorable physical replay divergence'
    assert exact(ma,s['mj_metadata']), 'MuJoCo allocation/configuration changed'
    for key in restored_arrays:
        destination=getattr(t.e.sim.data,key);saved=old['mjdata_arrays'][key]
        assert destination.shape==saved.shape and destination.dtype==saved.dtype and destination.flags.writeable
        destination[...] = saved
    restored=portable(t)
    assert exact(canonical(restored,metadata(t.e.sim.model,t.e.sim.data))['mjdata_arrays'],cb), 'full physical array restore differs'


    assert exact(now['history'],old['history']),'BFM history differs'
    def devices(value,template):
        from collections import deque
        if torch.is_tensor(value):
            assert torch.is_tensor(template),'Missing tensor device template'
            return value.to(template.device)
        if isinstance(value,dict):return {k:devices(v,template[k]) for k,v in value.items()}
        if isinstance(value,(list,tuple,deque)):
            assert len(value)==len(template),'Physical history length differs'
            vv=[devices(v,w) for v,w in zip(value,template)]
            return deque(vv,maxlen=value.maxlen) if isinstance(value,deque) else tuple(vv) if isinstance(value,tuple) else vv
        return value
    for obj,key in [(t,'task'),(t.e,'env'),(t.e.base,'base'),(t.e.tether,'tether'),(t.h,'force_history')]:put(obj,{k:devices(v,getattr(obj,k)) for k,v in old[key].items()})
    t.physics_rng=copy.deepcopy(s['physics_rng']);restore_global_rng(s['rng']);t.rng.bit_generator.state=copy.deepcopy(s['task_rng'])
    env.physical_stream=s['physical_stream'];env.rows=s['rows'];env.native=s['native'];env.commands=s['commands'];p.actions=s['actions'];env.episodes=s['episodes'];env.next_cursor=s['cursor']
    assert tree_digest(agent.state())==tree_digest(s['agent']),'agent/Adam/target restore differs'
    return s['round_id']

def adopt(root,source):
    """Reference an immutable verified incoming round in the new run's own pointer."""
    import shutil
    root=Path(root);source=Path(source);root.mkdir(parents=True,exist_ok=True)
    dest=root/source.name;assert not dest.exists();shutil.copytree(source,dest,copy_function=os.link)
    temp=root/'latest-committed.pending';jsonwrite(temp,dict(package=dest.name,manifest_sha256=sha(dest/'manifest.json')))
    os.replace(temp,root/'latest-committed.json');fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
