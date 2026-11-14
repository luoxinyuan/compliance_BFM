"""Fixed-replay QA1 metadata/schedule lock. CPU only, no optimizers/physics."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.runtime),str(a.runtime/'research_notes')]
from research.interaction_fb.approved_journal import load_state
from research.interaction_fb.journal import dump,read
from diagnostic_fb_resume import tree_digest
started=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
s=load_state(a.source);rows=s['rows'];ph=s['physical_stream'];assert len(rows)==3500 and len(ph)==14000
assert s['agent']['q_steps']==56 and s['agent']['actor_steps']==0
hist=[r for r in s['agent']['audit'] if r['kind']=='q'];assert len(hist)==56
counts=np.bincount([i for r in hist for i in r['indices']],minlength=3500);assert counts.sum()==3584
meta=[];seen_active={};previous_speed={};prev_spec=None
for i,row in enumerate(rows):
    parent=bytes(row['provenance']['parent_sha256'][0].tolist()).hex()
    si=int(row['provenance']['spec_index'].item());spec=s['task_specs'][si]
    assert parent==spec['parent_id']
    ticks=ph[4*i:4*i+4];t=float(row['provenance']['time']);assert abs(ticks[0]['force_time']-t)<1e-8
    assert abs(float(row['duration'])-.02)<1e-7
    active=[bool(x['active']) for x in ticks];speed=float(np.linalg.norm(ticks[0]['anchor_velocity']))
    if not ticks[0]['connection_enabled']:phase='no_contact_hold'
    elif len(set(active))>1:phase='transition_mixed'
    elif not active[0]:phase='released_recovery' if seen_active.get(si,False) else 'precontact'
    elif speed<=1e-8:phase='holding'
    elif speed < previous_speed.get(si,speed)-1e-8:phase='decelerating'
    else:phase='moving'
    seen_active[si]=seen_active.get(si,False) or any(active);previous_speed[si]=speed
    meta.append(dict(index=i,parent=parent,phase=phase,family=spec['demand_family'],spec_index=si,time=t,speed=speed,
        actual_active_ticks=sum(active),stratum=parent+'|'+phase))
strata=collections.defaultdict(list)
for m in meta:strata[m['stratum']].append(m['index'])
gu=torch.Generator().manual_seed(636413);gc=torch.Generator().manual_seed(636414)
u=torch.randint(3500,(456,64),generator=gu);c=torch.empty_like(u);exposure=counts.copy()
for b in range(456):
    for slot in range(64):
        group=strata[meta[int(u[b,slot])]['stratum']];minimum=min(exposure[j] for j in group)
        ties=[j for j in group if exposure[j]==minimum]
        j=ties[int(torch.randint(len(ties),(1,),generator=gc))];c[b,slot]=j;exposure[j]+=1
        assert meta[j]['stratum']==meta[int(u[b,slot])]['stratum']
    assert collections.Counter(meta[int(j)]['stratum'] for j in u[b])==collections.Counter(meta[int(j)]['stratum'] for j in c[b])
def validate(schedule,source_digest):
    assert source_digest==tree_digest(rows)
    assert schedule.shape==(456,64) and ((schedule>=0)&(schedule<3500)).all()
    for b in range(456):assert [meta[int(j)]['stratum'] for j in schedule[b]]==[meta[int(j)]['stratum'] for j in u[b]]
validate(c,tree_digest(rows))
bad=c.clone();different=next(j for j in range(3500) if meta[j]['stratum']!=meta[int(u[0,0])]['stratum']);bad[0,0]=different
rejections=[]
for name,ss,dd in [('wrong_parent_or_phase',bad,tree_digest(rows)),('wrong_source',c,'wrong')]:
    try:validate(ss,dd)
    except AssertionError:rejections.append(name)
    else:raise AssertionError('Rejection fixture accepted '+name)
assert torch.equal(u,torch.randint(3500,(456,64),generator=torch.Generator().manual_seed(636413)))
parents=json.loads((a.runtime/'research_notes/r2_splits/train.json').read_text())['parents'];seen={m['parent'] for m in meta}
pros=[]
for index,branch,offset in [(9,0,.7),(6,0,.7),(8,0,2.6),(12,1,.7)]:
    pp=parents[index];spec=pp['specs'][branch];assert pp['parent_id'] not in seen if 'parent_id' in pp else spec['parent_id'] not in seen
    onset=spec['onset'];target=round((onset+offset)*10)/10
    pros.append(dict(parent_index=index,parent=spec['parent_id'],spec=spec,boundary_s=target,candidate_seed=636500+index,
        candidate_contract='mean, +/- draw0, +/- draw1; one 20ms action then fixed mean to 1s',prefix_seed=636600+index))
packet=dict(metadata=meta,historical_counts=torch.tensor(counts),U=u,C=c,U_rng=gu.get_state(),C_rng=gc.get_state(),prospective=pros)
saved=dump(a.output/'schedule.ptz',packet);assert tree_digest(read(a.output/'schedule.ptz',saved))==tree_digest(packet)
source_manifest=hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest()
lock=dict(version='G17-QA1-approved-v1',source=str(a.source),source_manifest_sha256=source_manifest,
    source_replay_digest=tree_digest(rows),source_agent_digest=tree_digest(s['agent']),
    source_adams_digest=tree_digest(s['agent']['optimizers']),source_targets_digest=tree_digest(s['agent']['targets']),
    schedule=saved,metadata_digest=tree_digest(meta),U_digest=tree_digest(u),C_digest=tree_digest(c),
    seeds=dict(U=636413,C=636414),nodes=[56,128,256,512],batch64=True,
    total_new_Q_groups_per_branch=456,only_Q=True,actor_updates=0,I_paused=True,
    counts_by_family=dict(collections.Counter(m['family'] for m in meta)),counts_by_phase=dict(collections.Counter(m['phase'] for m in meta)),
    independent_parents=len(seen),strata={k:len(v) for k,v in strata.items()},historical_unique=int((counts>0).sum()),
    schedule_fixtures=dict(exact_slot_stratum=True,least_exposure_every_slot=True,uniform_reproducible=True,rejections=rejections),
    phase_semantics='Actual per-row four 5ms ticks only: connection-disabled/active transition/precontact/released/velocity zero/velocity decreasing/moving. 1e-8 speed metadata tolerance; no actor/reward modification.',
    prospective=pros,wall_s=time.monotonic()-started)
(a.output/'lock.json').write_text(json.dumps(lock,indent=2));print(json.dumps(lock,indent=2))
