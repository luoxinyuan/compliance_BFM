"""Deterministic group recipes. No sealed execution or outcome is generated."""
import json,hashlib,copy,random
from pathlib import Path

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def build():
    result={};sources=set();partners=set()
    for split,n,seed in [('train',30,636301),('dev',12,636302),('sealed',24,636303)]:
        rng=random.Random(seed);parents=[]
        for j,family in enumerate(['near','mid','stop']*(n//3)+['zero','zero','none','none']):
            # Executed bounded native pulse creates different source histories;
            # different clock/seed alone is never an independent source.
            until=round(rng.uniform(4.1,4.8),1)
            profile=dict(arm_fraction=1.,pulse_fraction=rng.uniform(.02,.08),pulse_duration=rng.uniform(.3,.7),phase_offset=0.,phase_rate=1.)
            source=dict(source='standing',until=until,profile=profile)
            source_id=digest(source);assert source_id not in sources;sources.add(source_id)
            onset=round(until+rng.uniform(.4,.9),1)
            if family=='near':d=rng.uniform(.04,.12);ramp=rng.uniform(1.6,2.6)
            elif family in ('mid','stop'):d=rng.uniform(.16,.45);ramp=rng.uniform(2.8,4.8)
            else:d=0.;ramp=3.
            ramp=round(ramp,1)
            hold=round(rng.uniform(1.5,2.5),1);stop=round(onset+ramp*rng.uniform(.4,.7),1) if family=='stop' else None
            decel=round(rng.uniform(.5,.8),1) if stop is not None else 0.
            partner=dict(distance=d,ramp=ramp,onset=onset,hold=hold,stop=stop,deceleration=decel,enabled=family!='none')
            pid=digest(dict(source_origin=source_id,request=partner));assert pid not in partners;partners.add(pid)
            group=digest(dict(source=source_id,partner=pid))
            specs=[]
            for branch in (['stop','continue'] if family=='stop' else [family]):
                endmove=stop+decel if branch=='stop' else onset+ramp
                s=dict(geometry='wrist',enabled=family!='none',distance=d,ramp=ramp,onset=onset,
                    release=endmove+hold,end=endmove+hold+4.,rotation_deg=0.,curvature=0.,heading_deg=0.,
                    family='walking',source='standing',source_until=until,total_cap=10.,anchor_mode='onset_actual',
                    demand_family=branch,source_profile=profile,parent_id=group,source_parent_id=source_id,
                    partner_parent_id=pid,common_prefix_parent_id=group,deceleration=decel if branch=='stop' else 0.)
                if branch=='stop':s['stop_time']=stop
                specs.append(s)
            parents.append(dict(parent_id=group,source_parent_id=source_id,partner_parent_id=pid,family=family,specs=specs))
        result[split]=dict(seed=seed,parents=parents)
    return result

def audit(m):
    report={'structural_group_pass':True,'physical_state_independence':'not yet executed; recipe uniqueness does not establish statistical independence','partner_identity':'request plus actual-source-origin recipe; identical zero-demand motion alone is not independent' ,'splits':{}}
    seen={k:set() for k in ('parent_id','source_parent_id','partner_parent_id')}
    for split,v in m.items():
        groups=v['parents'];summary={}
        for k in seen:
            ids=[g[k] for g in groups];assert len(ids)==len(set(ids));assert not seen[k].intersection(ids);seen[k].update(ids)
        for g in groups:
            summary[g['family']]=summary.get(g['family'],0)+1
            if g['family']=='stop':
                a,b=g['specs'];keys=('parent_id','common_prefix_parent_id','source_parent_id','partner_parent_id','source_profile','source_until','distance','ramp','onset')
                assert all(a[k]==b[k] for k in keys);assert .16<=a['distance']<=.45
        report['splits'][split]=dict(parents=len(groups),episodes=sum(len(g['specs']) for g in groups),families=summary,canonical_sha256=digest(v))
    return report
if __name__=='__main__':
    import sys
    dst=Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True);m=build();a=audit(m)
    for k,v in m.items():
        p=dst/(k+'.json');assert not p.exists();p.write_text(json.dumps(v,indent=2));a['splits'][k]['file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    (dst/'audit.json').write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))
