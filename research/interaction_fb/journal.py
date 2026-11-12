"""Append-only replay/physics segments + per-round complete mutable core.
Every committed manifest resolves an immutable complete logical checkpoint.
"""
import io,zlib,json,os,copy,shutil
from pathlib import Path
import torch
from .checkpoint import sha,sync,jsonwrite
from dual_track_task import global_rng
from diagnostic_frozen import portable
from f1_mjdata_contract import metadata
LISTS=('rows','native','physical_stream')
def dump(path,value):
    buf=io.BytesIO();torch.save(value,buf);path.write_bytes(zlib.compress(buf.getvalue(),1));sync(path)
    return dict(sha256=sha(path),bytes=path.stat().st_size)
def read(path,expected):
    assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes']
    return torch.load(io.BytesIO(zlib.decompress(path.read_bytes())),map_location='cpu',weights_only=False)
def load_state(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text());assert m['complete']
    core=read(path/'core.ptz',m['files']['core.ptz']);lists={k:[] for k in LISTS}
    for seg in m['segments']:
        rel=Path(seg['path']);assert len(rel.parts)==2 and rel.parts[0].startswith('round-') and rel.parts[1]=='delta.ptz'
        part=read(path.parent/rel,seg)
        for k in LISTS:
            assert len(part[k])==seg['counts'][k];lists[k].extend(part[k])
    for k in LISTS:assert len(lists[k])==m['lengths'][k]
    return dict(core,**lists)
def save(root,round_id,agent,env,runtime,reason='completed_round'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dest=root/f'round-{round_id}';assert not dest.exists()
    previous=[];lengths={k:0 for k in LISTS}
    pointer=root/'latest-committed.json'
    if pointer.exists():
        pp=json.loads(pointer.read_text());mp=root/pp['package']/'manifest.json';assert sha(mp)==pp['manifest_sha256']
        prev=json.loads(mp.read_text());assert prev['round_id']<round_id
        previous=prev['segments'];lengths=prev['lengths']
    tmp=root/f'.pending-{round_id}-{os.getpid()}';tmp.mkdir()
    t=env.task
    core=dict(version='IFB-delta-v1',agent=agent.state(),commands=env.commands,actions=env.proxy.actions,
        task_specs=env.specs,spec_id=env.spec_id,episodes=env.episodes,cursor=env.next_cursor,
        physical=portable(t),mj_metadata=metadata(t.e.sim.model,t.e.sim.data),rng=global_rng(),
        task_rng=copy.deepcopy(t.rng.bit_generator.state),physics_rng=copy.deepcopy(t.physics_rng),
        round_id=round_id,runtime=runtime,source_sha=t.prefix_data['audit']['sha256'])
    delta={k:getattr(env,k)[lengths[k]:] for k in LISTS}
    assert all(len(getattr(env,k))>=lengths[k] for k in LISTS)
    files={'core.ptz':dump(tmp/'core.ptz',core),'delta.ptz':dump(tmp/'delta.ptz',delta)}
    check=read(tmp/'core.ptz',files['core.ptz']);assert check['round_id']==round_id
    check=read(tmp/'delta.ptz',files['delta.ptz']);assert all(len(check[k])==len(delta[k]) for k in LISTS)
    seg=dict(path=dest.name+'/delta.ptz',**files['delta.ptz'],counts={k:len(v) for k,v in delta.items()})
    manifest=dict(format='delta-v1',round_id=round_id,reason=reason,complete=True,runtime=runtime,arm=agent.arm,
        files=files,segments=previous+[seg],lengths={k:len(getattr(env,k)) for k in LISTS},
        transitions=env.next_cursor,updates=agent.updates,base_sha='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361')
    jsonwrite(tmp/'manifest.json',manifest);fd=os.open(tmp,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    os.rename(tmp,dest);p=root/'latest-committed.pending'
    jsonwrite(p,dict(package=dest.name,manifest_sha256=sha(dest/'manifest.json')));os.replace(p,pointer)
    fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd);return manifest

def adopt(root,source):
    root=Path(root);source=Path(source);root.mkdir(parents=True,exist_ok=True)
    state=load_state(source);m=json.loads((source/'manifest.json').read_text())
    # Import only immutable dependencies, never rewrite source packages.
    for name in sorted(set(Path(s['path']).parts[0] for s in m['segments'])|{source.name}):
        assert not (root/name).exists();shutil.copytree(source.parent/name,root/name,copy_function=os.link)
    p=root/'latest-committed.pending';jsonwrite(p,dict(package=source.name,manifest_sha256=sha(source/'manifest.json')))
    os.replace(p,root/'latest-committed.json');fd=os.open(root,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
