"""CPU-only D1 archived-action gradients and matched finite-window comparisons."""
import json,sys,io,zlib,hashlib,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch
import numpy as np
from torch import nn
from research.interaction_fb.schema import privileged
P=ROOT.parents[2];S=ROOT.parent
torch.set_num_threads(1)
path=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-B-v3/checkpoints/round-19/core.ptz'
ck=torch.load(io.BytesIO(zlib.decompress(path.read_bytes())),map_location='cpu',weights_only=False)['agent']
w={k[2:]:v for k,v in ck['nets'].items() if k.startswith('q.')}
q=nn.Sequential(nn.Linear(w['0.weight'].shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1));q.load_state_dict(w);q.requires_grad_(False)
all_results=[]
for run in ('d1-p2-02','d1-p3-01'):
    d=S/'runs'/run;ev=json.loads((d/'evaluation.json').read_text());rows=[];v=None;grad=None
    for b in ev['branches']:
        trace=torch.load(d/b['candidate']/'trace.pt',map_location='cpu',weights_only=False)
        first=trace['rows'][0];a=first['action'].clone().requires_grad_();o=first['observation'];z=first['z'];zi=first['zi']
        score=q(torch.cat([privileged(o),a,z,zi],-1));g=torch.autograd.grad(score.sum(),a)[0]
        if b['candidate']=='V1':v=a.detach();grad=g.detach()
        support=[]
        for t in trace['physics']:
            if float(np.max(t['stage_loaded_slip_mps']))>.45:support.append(t)
        rows.append(dict(candidate=b['candidate'],gpu_q=b['one_step']['q'],cpu_q=float(score),cpu_gpu_q_abs=abs(float(score)-b['one_step']['q']),
            action_delta_l2=float((a-v).norm()),q_first_order_from_V1=float(((a-v)*grad).sum()),
            q_gradient_l2=float(g.norm()),return3s=b['discounted_return'],metrics=b['metrics'],
            segments=b['discounted_segments'],one_step=b['one_step'],risk_ticks=len(support),
            action=a.detach().tolist(),q_gradient=g.tolist(),trace_sha256=b['trace_sha256']))
    pairs=[]
    for a,b in itertools.combinations(rows,2):
        dq=a['gpu_q']-b['gpu_q'];dr=a['return3s']-b['return3s']
        pairs.append(dict(pair=[a['candidate'],b['candidate']],q_delta=dq,return_delta=dr,rank_agrees=dq*dr>0))
    all_results.append(dict(run=run,parent=ev['case'],rows=rows,pairs=pairs,rank_agree=sum(x['rank_agrees'] for x in pairs),pair_count=len(pairs)))
out=S/'analysis';out.mkdir(exist_ok=True)
(out/'d1-analysis.json').write_text(json.dumps(dict(results=all_results,scope='CPU retrospective gradient; 3s V1 continuation is not old target infinite-horizon truth'),indent=2))
print(json.dumps([{k:v for k,v in x.items() if k not in ('rows','pairs')} for x in all_results],indent=2))
