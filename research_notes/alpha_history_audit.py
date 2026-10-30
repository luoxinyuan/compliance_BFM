"""Bounded saved-training-data audit; no BFM loading, simulation or fitting."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from contact_gate import ContactGate

SOURCE=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-contact-gate-20261030A')
NAMES=['dev_early','dev_late','no_contact','pulse']


def audit():
    out=SOURCE/'physics-attempt-01/outputs'
    cfg=json.loads((out/'protocol.json').read_text())['gate']
    manifest=json.loads((SOURCE/'analysis/evidence-audit.json').read_text())['audited_conditions']
    episodes={};features=[];labels=[];locations=[];summary={}
    for name in NAMES:
        path=out/(name+'__contact.json');raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest[name]['trace_sha256']['contact']
        e=json.loads(raw);cs=e['control'];episodes[name]=e
        assert e['boundary'] is None
        full=ContactGate(cfg)
        for c in cs:assert full.step(c['measured_force_xyz'])['alpha']==c['alpha']
        suffix={}
        for n in [3,20,320]:
            diffs=[];worst=None
            for i,c in enumerate(cs):
                g=ContactGate(cfg)
                for j in range(max(0,i-n+1),i+1):g.step(cs[j]['measured_force_xyz'])
                error=abs(g.alpha-c['alpha']);diffs.append(error)
                if worst is None or error>worst['absolute_alpha_error']:
                    worst=dict(index=i,time=c['time'],teacher_alpha=c['alpha'],suffix_alpha=g.alpha,absolute_alpha_error=error)
            suffix[str(n)]=dict(max_abs=max(diffs),rms=float(np.sqrt(np.mean(np.square(diffs)))),worst=worst)
        summary[name]=dict(control_samples=len(cs),source_sha256=manifest[name]['trace_sha256']['contact'],suffix_reconstruction=suffix)
        for i,c in enumerate(cs):
            prop=np.concatenate([np.asarray(c['bfm_obs'][k]).reshape(-1) for k in ['state','last_action','history_actor']])
            history=np.zeros((20,3));past=np.asarray([q['measured_force_xyz'] for q in cs[max(0,i-19):i+1]])
            history[-len(past):]=past
            features.append(np.r_[prop,history.reshape(-1)/5]);labels.append(c['alpha']);locations.append([name,i,c['time']])
    x=np.asarray(features);y=np.asarray(labels);scale=x.std(0).clip(.1);x=(x-x.mean(0))/scale
    dist=(np.sum(x*x,1)[:,None]+np.sum(x*x,1)[None,:]-2*x@x.T).clip(0)/x.shape[1]
    for i,(name,ix,t) in enumerate(locations):
        for j,(name2,jx,t2) in enumerate(locations):
            if name==name2 and abs(ix-jx)<=10:dist[i,j]=np.inf
    nearest=dist.argmin(1);d=np.sqrt(dist[np.arange(len(x)),nearest]);gap=abs(y-y[nearest])
    idx=np.argsort(gap)[-8:][::-1]
    pairs=[dict(a=locations[i],b=locations[nearest[i]],standardized_feature_rms_distance=float(d[i]),alpha_a=float(y[i]),alpha_b=float(y[nearest[i]]),gap=float(gap[i])) for i in idx]
    assert all(v['suffix_reconstruction']['320']['max_abs']==0 for v in summary.values())
    return dict(episodes=summary,short_input_dimension=x.shape[1],nearest_neighbor_distance_quantiles=np.quantile(d,[0,.5,.9,1]).tolist(),
        nearest_neighbor_alpha_gap_quantiles=np.quantile(gap,[0,.5,.9,1]).tolist(),largest_nearest_neighbor_gaps=pairs,
        interpretation='Short force suffix alone loses entry/progress memory; this does not prove full proprio is ambiguous. A320-sample6.4s raw-force FIFO exactly reconstructs teacher alpha for all training records when replayed from resting state, so contains sufficient information on this finite protocol. No teacher variables are input to Student; this replay is audit only.',
        selected_memory=dict(force_samples=320,seconds=6.4,proprio_action_history='original BFM deployment-whitelisted inputs',zero_padding='sensor absence before start, no validity mask, length, clock, event counter, or teacher state'),
        heldout_used=False,model_loaded=False,policy_updates=0)

if __name__=='__main__':
    result=audit();Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
