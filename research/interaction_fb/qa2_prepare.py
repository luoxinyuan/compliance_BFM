"""Issue65 P0/P1: immutable observed trajectories only, no optimizer or physics."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
P=ROOT.parents[2];STAGE=ROOT.parent
from research.interaction_fb.approved_journal import load_state
from research.interaction_fb.journal import dump,read
from research.interaction_fb.schema import privileged
from diagnostic_fb_resume import tree_digest

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))

def audit_force(ph):
    active=0;saturated=0;rawerr=0.;forceerr=0.;rows=[]
    for p in ph:
        for c in p['contacts']:
            on=bool(c['active']);assert on==bool(p['active'])
            raw=80*(np.asarray(c['anchor'])-c['point'])+8*(np.asarray(c['anchor_velocity'])-c['velocity']) if on else np.zeros(3)
            cap=float(c['cap']);assert cap==10
            f=raw*min(1.,cap/max(float(np.linalg.norm(raw)),1e-30))
            rawerr=max(rawerr,float(np.max(np.abs(raw-c['raw_force']))))
            forceerr=max(forceerr,float(np.max(np.abs(f-c['force']))))
            if on:active+=1;saturated+=int(np.linalg.norm(raw)>=cap)
            rows.append(dict(raw_norm=float(np.linalg.norm(raw)),applied_norm=float(np.linalg.norm(c['force'])),tracking=float(np.linalg.norm(np.asarray(c['anchor'])-c['point'])),relative_speed=float(np.linalg.norm(np.asarray(c['anchor_velocity'])-c['velocity'])),active=on))
    assert rawerr<=1e-12 and forceerr<=1e-12,(rawerr,forceerr)
    frac=saturated/active if active else None
    return dict(active_ticks=active,saturated_ticks=saturated,saturation_fraction=frac,stratum=('inactive' if frac is None else 'low' if frac<=.1 else 'high' if frac>=.9 else 'mixed'),raw_linf=rawerr,applied_linf=forceerr,raw_norm_max=max(r['raw_norm'] for r in rows),tracking_max=max(r['tracking'] for r in rows),relative_speed_max=max(r['relative_speed'] for r in rows))

def main():
    t=time.monotonic();out=STAGE/'prep/p0-p1';out.mkdir(exist_ok=False)
    source=P/'experiments/r2-v11-qentry-20261114/runs/E-warmup3500-10/checkpoints/round-14'
    src=load_state(source);rr=src['rows'];ph=src['physical_stream'];assert len(rr)==3500 and len(ph)==14000
    assert tree_digest(rr)=='befe8ca2574600769f9737c329dab4ac48d118197d4d1516bf187e8dd3541e36'
    sch=P/'experiments/r2-g17-qa1-20261115/prep/fixed-schedule-01'
    oldlock=json.loads((sch/'lock.json').read_text());schedule=read(sch/'schedule.ptz',oldlock['schedule']);meta=schedule['metadata']
    mainforce=audit_force(ph);branches=[];g=P/'experiments/r2-v11-qentry-20261114/runs/paired-Q-ranking-12'
    archived=json.loads((g/'qualification.json').read_text())
    for st in archived['results']:
        for br in st['branches']:
            trace=read(g/f"round-{st['round']}"/(br['id']+'.ptz'),br['archive'])
            af=audit_force(trace['physics'])
            branches.append(dict(round=st['round'],parent=st['parent_id'],family=st['family'],candidate=br['id'],G1s_pi_mean=br['discounted_return'],Q56_infinite=br['scores']['E']['q_min'],**af))
    p0=dict(status='PASS_DATA_FORCE_SEMANTICS',main=mainforce,archived_pi_mean_branches=branches,
        previous_semantics_audit=dict(path='QA1-C/preupdate-target-audit.json',exact_mean_controls=1968,max_action_linf=0,max_z_linf=0),
        conditioning='Q receives current privileged quantities, target/previous z and slot; no future partner spec or task clock. Actual reset/source future exists in simulator, not Q input. No strict same-past/opposite-future counterfactuals in old40: UNMEASURED.',
        policy_identity='3500 stochastic frozen mu vs archived40 first-action interventions followed by deterministic pi_mean; not exchangeable labels.',new_physics=0,optimizer_updates=0)
    write(out/'p0.json',p0)
    # Verify identity/continuity before constructing ANY finite-window label.
    contiguous=np.zeros(3500,dtype=bool);checks=collections.Counter();maxnext=0.
    for i,row in enumerate(rr):
        assert int(row['provenance']['origin_actual_online_execution'])==1
        assert abs(float(row['duration'])-.02)<1e-7
        assert float(row['aux_rewards']['contact_physical'])<=1e-12
        assert abs(float(row['provenance']['time'])-ph[4*i]['force_time'])<1e-8
        assert not bool(row['next']['terminated']), 'Unregistered absorbing-terminal path requires explicit audit'
        if i+1<len(rr):
            nxt=rr[i+1];same=meta[i]['parent']==meta[i+1]['parent'] and meta[i]['spec_index']==meta[i+1]['spec_index'] and torch.equal(row['provenance']['online_episode'],nxt['provenance']['online_episode'])
            if same:
                assert abs(float(nxt['provenance']['time'])-float(row['provenance']['time'])-float(row['duration']))<1e-8
                for key,val in row['next']['observation'].items():
                    other=nxt['observation'][key]
                    d=float((val-other).abs().max());maxnext=max(maxnext,d)
                    assert torch.equal(val,other),('Next observation discontinuity',i,key,d)
                contiguous[i]=True;checks['continuous_edges']+=1
            else:checks['episode_or_parent_boundaries']+=1
    # Four fit parents, two heldout parents: whole stop/continue siblings stay together.
    train_parents={meta[i]['parent'] for i in (0,480,930,1995)}
    held_parents={m['parent'] for m in meta}-train_parents
    assert len(train_parents)==4 and len(held_parents)==2
    X=torch.cat([torch.cat([privileged(r['observation']),r['action'],r['z'],r['zi']],-1) for r in rr]).float()
    rewards=np.asarray([float(r['aux_rewards']['contact_physical']) for r in rr]);discount=np.asarray([.99**(float(r['duration'])/.1) for r in rr])
    labels=np.full((3500,50),np.nan);eligible=np.zeros((3500,50),dtype=bool)
    for i in range(3500):
        total=0.;weight=1.
        for h in range(1,51):
            j=i+h-1
            if j>=3500 or (h>1 and not contiguous[j-1]):break
            total+=weight*rewards[j];weight*=discount[j];labels[i,h-1]=total;eligible[i,h-1]=True
    indices=[];strata=collections.defaultdict(list)
    for i,m in enumerate(meta):
        if m['parent'] in train_parents:
            for h in range(1,51):
                if eligible[i,h-1]:strata[m['stratum']].append((i,h))
    groups=sorted(strata);gen=torch.Generator().manual_seed(650211);samp=[]
    # Same parent/phase slot sequence AND same remaining-horizon draws for V1/all V2.
    for k in range(1024):
        row=[]
        for b in range(64):
            group=groups[int(torch.randint(len(groups),(1,),generator=gen))]
            row.append(strata[group][int(torch.randint(len(strata[group]),(1,),generator=gen))])
        samp.append(row)
    samp=torch.tensor(samp,dtype=torch.long)
    parents=json.loads((ROOT/'research_notes/r2_splits/train.json').read_text())['parents'];seen={m['parent'] for m in meta}
    prospective=[]
    for idx,offset in [(6,None),(8,None),(9,.7),(11,None)]:
        spec=parents[idx]['specs'][0];assert spec['parent_id'] not in seen
        fam=spec['demand_family']
        if idx==6:boundary=round((spec['onset']+spec['ramp']+.5)*10)/10
        elif idx==8:boundary=round((spec['stop_time']+spec['deceleration']+.3)*10)/10
        elif idx==9:boundary=round((spec['onset']+offset)*10)/10
        else:boundary=round((spec['onset']+.7)*10)/10
        assert boundary+1<=spec['end']
        prospective.append(dict(parent_index=idx,parent=spec['parent_id'],spec=spec,boundary_s=boundary,prefix_seed=650500+idx,candidate_seed=650600+idx,crn_seeds=[650701,650702]))
    assert {p['spec']['demand_family'] for p in prospective}=={'mid','stop','near','zero'}
    # Reject crossing even same parent stop/continue and enforce n=50 exact MC.
    assert not contiguous[2519] and meta[2519]['parent']==meta[2520]['parent']
    assert not eligible[2519,1] and np.isclose(labels[0,49],sum(np.prod(discount[:j])*rewards[j] for j in range(50)))
    data=dict(X=X,labels=torch.tensor(labels,dtype=torch.float64),eligible=torch.tensor(eligible),rewards=torch.tensor(rewards,dtype=torch.float64),discounts=torch.tensor(discount,dtype=torch.float64),metadata=meta,schedule=samp,fit_parents=sorted(train_parents),heldout_parents=sorted(held_parents),source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),source_replay_digest=tree_digest(rr),prospective=prospective,source_reference=str(source))
    dm=dump(out/'data.ptz',data);assert tree_digest(read(out/'data.ptz',dm))==tree_digest(data)
    counts=[]
    for h in (5,25,50):
        for split,pp in [('fit',train_parents),('heldout',held_parents)]:
            ix=[i for i,m in enumerate(meta) if m['parent'] in pp and eligible[i,h-1]]
            counts.append(dict(H=h,split=split,windows=len(ix),parents=len({meta[i]['parent'] for i in ix}),high=sum(abs(labels[i,h-1])>=1 for i in ix),low=sum(abs(labels[i,h-1])<=.1 for i in ix),by_phase=dict(collections.Counter(meta[i]['phase'] for i in ix)),censored=sum(m['parent'] in pp for m in meta)-len(ix)))
    p1=dict(status='PASS_LABELS_INDEPENDENCE_LIMITED',source_rows=3500,independent_parents=6,fit_parents=sorted(train_parents),heldout_parents=sorted(held_parents),counts=counts,next_observation_max_linf=maxnext,continuity_checks=dict(checks),optimizer_updates=0,physics_ticks=0,data=dm,data_logical_digest=tree_digest(data),schedule_digest=tree_digest(samp),remaining_horizon_contract='Shared horizon head requires legal remaining h1..50. Only H5/25/50 are primary reports. All variants share the same actual full-window legal targets and parent/phase/horizon indices.',prospective=prospective,wall_s=time.monotonic()-t)
    write(out/'p1.json',p1);print(json.dumps(dict(p0=p0['main'],p1={k:v for k,v in p1.items() if k not in ('prospective','fit_parents','heldout_parents')}),indent=2),flush=True)

if __name__=='__main__':main()
