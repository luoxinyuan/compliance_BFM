"""Explicit remaining-horizon Q_mu heads. No BFM, actor or physical updates."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
from research.interaction_fb.journal import dump,read
from diagnostic_fb_resume import tree_digest
S=ROOT.parent
torch.set_num_threads(1)

class HorizonQ(nn.Module):
    def __init__(self,d,mean,scale):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('scale',scale.clone())
        self.net=nn.Sequential(nn.Linear(d+1,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
    def forward(self,x,h):
        h=h.float().reshape(-1,1)
        v=self.net(torch.cat([(x-self.mean)/self.scale,h/50],-1))
        return -F.softplus(v)*(h/50)

def write(p,d):
    tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));os.replace(tmp,p)
def setup(data,device='cuda'):
    x=data['X'].to(device);fit=torch.tensor([m['parent'] in data['fit_parents'] for m in data['metadata']],device=device)
    avg=x[fit].mean(0);scale=x[fit].std(0).clamp(min=.01)
    torch.manual_seed(650801);torch.cuda.manual_seed_all(650801) if device=='cuda' else None
    return x,HorizonQ(x.shape[1],avg,scale).to(device)

def metrics(pred,labels,meta,ix,baseline=None):
    records=[]
    for p in sorted({meta[i]['parent'] for i in ix}):
        ids=np.array([i for i in ix if meta[i]['parent']==p],dtype=int);y=labels[ids];q=pred[ids]
        hi=np.abs(y)>=1;lo=np.abs(y)<=.1
        records.append(dict(parent=p,n=len(ids),high=int(hi.sum()),low=int(lo.sum()),mae=float(np.mean(abs(q-y))),
            high_nmae=float(np.mean(abs(q[hi]-y[hi])/abs(y[hi]))) if hi.any() else None,
            high_optimism=float(np.mean((q[hi]-y[hi])/abs(y[hi]))) if hi.any() else None,
            low_mae=float(np.mean(abs(q[lo]-y[lo]))) if lo.any() else None))
    high=[r for r in records if r['high']];low=[r for r in records if r['low']]
    macro=float(np.mean([r['high_nmae'] for r in high])) if high else None
    optimism=float(np.mean([r['high_optimism'] for r in high])) if high else None
    numerical=(macro is not None and macro<=.25 and max(r['high_nmae'] for r in high)<=.5 and optimism<=.1 and bool(low) and max(r['low_mae'] for r in low)<=.10)
    enough=len(high)>=2 and len(low)>=2
    return dict(parents=records,high_parents=len(high),low_parents=len(low),high_macro_nmae=macro,high_macro_optimism=optimism,numerical_bounds=bool(numerical),G2=('PENDING_BASELINE' if numerical else 'FAIL') if enough else 'UNDERPOWERED')

def evaluate(model,x,data):
    ans={};N=len(x);meta=data['metadata'];eligible=data['eligible'].numpy();Y=data['labels'].numpy()
    for h in [5,25,50]:
        with torch.no_grad():pred=torch.cat([model(x[i:i+256],torch.full((min(256,N-i),),h,device=x.device)) for i in range(0,N,256)]).cpu().numpy()[:,0]
        hh={}
        for split,pp in [('fit',data['fit_parents']),('heldout',data['heldout_parents'])]:
            ix=[i for i,m in enumerate(meta) if m['parent'] in pp and eligible[i,h-1]]
            hh[split]=metrics(pred,Y[:,h-1],meta,ix)
            hh[split]['overall_rmse']=float(np.sqrt(np.mean((pred[ix]-Y[ix,h-1])**2)))
        ans[str(h)]=hh
    return ans

def baseline(data):
    meta=data['metadata'];y=data['labels'][:,49].numpy();valid=data['eligible'][:,49].numpy()
    fi=[i for i,m in enumerate(meta) if m['parent'] in data['fit_parents'] and valid[i]]
    ho=[i for i,m in enumerate(meta) if m['parent'] in data['heldout_parents'] and valid[i]]
    globalmean=float(np.mean(y[fi]));ph={p:float(np.mean([y[i] for i in fi if meta[i]['phase']==p])) for p in {meta[i]['phase'] for i in fi}}
    return {k:dict(metrics=metrics(v,y,meta,ho),global_mean=globalmean,phase_means=ph) for k,v in [('global',np.full(len(y),globalmean)),('phase',np.array([ph.get(m['phase'],globalmean) for m in meta]))]}

def save(out,name,model,target,opt,cursor,data,schedule_digest,extra=None):
    state=dict(version='QA2-explicit-Q-mu-h1-50-v1',model=copy.deepcopy(model.state_dict()),target=copy.deepcopy(target.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),cursor=cursor,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),source_manifest=data['source_manifest_sha256'],source_replay=data['source_replay_digest'],schedule_digest=schedule_digest,source_complete_physical_reference=data['source_reference'],extra=extra or {})
    state={k:v for k,v in state.items()};p=out/(name+'.ptz');sm=dump(p,state);assert tree_digest(read(p,sm))==tree_digest(state)
    write(p.with_suffix('.manifest.json'),dict(sm,logical_digest=tree_digest(state),cursor=cursor,runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()))
    return sm

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--mode',choices=['dry','V1','V2','restore-check'],required=True);a=pa.parse_args()
    torch.use_deterministic_algorithms(True);t=time.monotonic()
    lock=json.loads((S/'prep/p0-p1/p1.json').read_text());data=read(S/'prep/p0-p1/data.ptz',lock['data']);x,model=setup(data)
    assert tree_digest(data['schedule'])==lock['schedule_digest']
    target=copy.deepcopy(model).requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=.0001)
    Y=data['labels'].to('cuda',dtype=torch.float32);rew=data['rewards'].to('cuda',dtype=torch.float32);disc=data['discounts'].to('cuda',dtype=torch.float32)
    schedule=data['schedule'];initdigest=tree_digest(model.state_dict());base=baseline(data)
    if a.mode=='dry':
        # Frozen forward/backward profiling only. No optimizer.step or target mutation.
        torch.cuda.synchronize();b=time.monotonic()
        for k in range(32):
            ids=schedule[k,:,0].cuda();h=schedule[k,:,1].cuda();opt.zero_grad(set_to_none=True)
            loss=F.mse_loss(model(x[ids],h)[:,0],Y[ids,h-1]);loss.backward()
        torch.cuda.synchronize();elapsed=time.monotonic()-b
        assert tree_digest(model.state_dict())==initdigest and len(opt.state)==0
        out=S/'prep/dry';out.mkdir(exist_ok=False);sm=save(out,'frozen-profile',model,target,opt,0,data,lock['schedule_digest'])
        result=dict(optimizer_updates=0,profile_forward_backward_s_per_batch=elapsed/32,profile_groups=32,parameter_count=sum(p.numel() for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated(),packet_bytes=sm['bytes'],wall_s=time.monotonic()-t,baselines=base)
        write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True);return
    if a.mode=='restore-check':
        results=[]
        for name in ['V1-MC','V2-n1','V2-n5','V2-n25','V2-n50']:
            p=S/'runs'/name/(name+'.ptz');m=json.loads(p.with_suffix('.manifest.json').read_text());state=read(p,m)
            x2,m2=setup(data);t2=copy.deepcopy(m2).requires_grad_(False);o2=torch.optim.Adam(m2.parameters(),lr=.0001)
            m2.load_state_dict(state['model']);t2.load_state_dict(state['target']);o2.load_state_dict(copy.deepcopy(state['optimizer']))
            assert tree_digest(m2.state_dict())==tree_digest(state['model']) and tree_digest(t2.state_dict())==tree_digest(state['target']) and tree_digest(o2.state_dict())==tree_digest(state['optimizer'])
            assert state['source_replay']==data['source_replay_digest'] and state['cursor']==1024
            results.append(dict(model=name,model_target_adam_exact=True,source_reference_exact=True))
        write(S/'prep/new-process-value-restore.json',dict(results=results,wall_s=time.monotonic()-t));print(json.dumps(results));return
    variants=[('V1-MC',50)] if a.mode=='V1' else [('V2-n1',1),('V2-n5',5),('V2-n25',25),('V2-n50',50)]
    for name,n in variants:
        x,model=setup(data);assert tree_digest(model.state_dict())==initdigest
        target=copy.deepcopy(model).requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=.0001)
        out=S/'runs'/name;out.mkdir(exist_ok=False);audit=[];vstart=time.monotonic()
        save(out,name+'-initial',model,target,opt,0,data,lock['schedule_digest'])
        with torch.enable_grad():
            for k,ss in enumerate(schedule):
                ids=ss[:,0].cuda();h=ss[:,1].cuda();actual_n=torch.minimum(h,torch.full_like(h,n));total=torch.zeros(64,device='cuda');w=torch.ones(64,device='cuda')
                with torch.no_grad():
                    for j in range(n):
                        mask=actual_n>j
                        if not bool(mask.any()):break
                        at=ids[mask]+j;total[mask]+=w[mask]*rew[at];w[mask]*=disc[at]
                    rem=h-actual_n;boot=rem>0
                    if bool(boot.any()):total[boot]+=w[boot]*target(x[ids[boot]+actual_n[boot]],rem[boot])[:,0]
                    if n==50:assert torch.allclose(total,Y[ids,h-1],atol=2e-6,rtol=1e-6)
                opt.zero_grad(set_to_none=True);pred=model(x[ids],h)[:,0];loss=F.mse_loss(pred,total);loss.backward()
                grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
                assert np.isfinite(grad) and torch.isfinite(loss);opt.step()
                with torch.no_grad():
                    for pt,po in zip(target.parameters(),model.parameters()):pt.lerp_(po,.01)
                audit.append(dict(group=k+1,loss=float(loss.detach()),gradient_l2=grad,bootstrap_count=int(boot.sum()),parent_phase_draws=dict(collections_count(data['metadata'],ss[:,0].tolist()))))
        sm=save(out,name,model,target,opt,1024,data,lock['schedule_digest'])
        result=dict(name=name,n=n,groups=1024,initial_model_digest=initdigest,model=sm,metrics=evaluate(model,x,data),baselines=base,wall_s=time.monotonic()-vstart)
        write(out/'updates.json',audit);write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    print('TOTAL_WALL',time.monotonic()-t,flush=True)

def collections_count(meta,ix):
    import collections
    return collections.Counter(meta[i]['stratum'] for i in ix)

if __name__=='__main__':main()
