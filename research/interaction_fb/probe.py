"""Grouped directional probe first read; never a policy-selection or sealed gate.
Equal 288-slot linear ridge design, training-fold-only scaling, fixed lambda.
Effective feature rank differs and is reported; not a causal capacity proof.
"""
import json,gzip,hashlib
from pathlib import Path
import numpy as np

def auc(y,s):
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):return None
    d=s[pos,None]-s[neg];return float(((d>0)+.5*(d==0)).mean())
def features(rows,name):
    raw=np.asarray([r['B_R'] if name=='R' else r['B_I'] if name=='I' else r['B_R']+r['B_I'] if name=='RI' else r['current_contact'] for r in rows],dtype=np.float64)
    assert raw.shape[1]<=288
    return np.pad(raw,((0,0),(0,288-raw.shape[1])))
def evaluate(rows):
    groups=sorted(set(r['parent'] for r in rows));fold={p:int(hashlib.sha256(('probe-v1:'+p).encode()).hexdigest(),16)%3 for p in groups}
    result=dict(groups=len(groups),fold=fold,formal_gate=False,reason='dev-only first read; no sealed claims; effective ranks differ',labels={})
    for horizon in ('0.5','1.0'):
        for label in ('active','quality'):
            subset=[r for r in rows if r['labels'][horizon][label] is not None]
            # Also stratify on actual present contact, not future task metadata.
            key=horizon+'-'+label;result['labels'][key]={}
            for active in (None,0,1):
                rr=[r for r in subset if active is None or r['current_active']==active]
                per={};yy=np.asarray([r['labels'][horizon][label] for r in rr]);gg=np.asarray([r['parent'] for r in rr]);ff=np.asarray([fold[r['parent']] for r in rr])
                for name in ('R','I','RI','current_contact'):
                    pred=np.full(len(rr),np.nan);ranks=[];valid=[]
                    if rr:
                        x=features(rr,name)
                        for f in range(3):
                            tr=ff!=f;te=ff==f
                            if len(set(gg[tr]))<4 or len(set(gg[te]))<2 or len(set(yy[tr]))<2 or len(set(yy[te]))<2:continue
                            mu=x[tr].mean(0);sd=x[tr].std(0).clip(.05);xt=(x[tr]-mu)/sd;xv=(x[te]-mu)/sd
                            w=np.linalg.solve(xt.T@xt+np.eye(288)*(.01*sum(tr)),xt.T@(yy[tr]-yy[tr].mean()))
                            pred[te]=xv@w+yy[tr].mean();valid.append(f);ranks.append(int(np.linalg.matrix_rank(xt)))
                    ok=np.isfinite(pred)
                    per[name]=dict(AUROC=auc(yy[ok],pred[ok]),valid_folds=valid,heldout_groups=len(set(gg[ok])),rows=int(sum(ok)),effective_ranks=ranks,underpowered=not len(valid))
                result['labels'][key][str(active)]=per
    return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    assert 'sealed' not in str(a.input).lower();assert not a.output.exists()
    with gzip.open(a.input,'rt') as f:r=json.load(f)
    a.output.write_text(json.dumps(evaluate(r),indent=2,allow_nan=False))
