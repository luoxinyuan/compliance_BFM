"""H0 two finite outcome heads; explicit goal-to-real-command selection, no scalar Q."""
import argparse,copy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from research.interaction_fb.h0_common import *
from research.interaction_fb.journal import dump,read
from diagnostic_fb_resume import tree_digest
from dual_track_task import global_rng
torch.set_num_threads(1)
class Models(torch.nn.Module):
 def __init__(self,n):
  super().__init__()
  def mlp(a,b,hidden=64):return torch.nn.Sequential(torch.nn.Linear(a,hidden),torch.nn.SiLU(),torch.nn.Linear(hidden,32),torch.nn.SiLU(),torch.nn.Linear(32,b))
  # E/I have the exact same causal state/command input and first hidden capacity.
  torch.manual_seed(660201);self.E=mlp(n,24)
  torch.manual_seed(660201);self.F=mlp(n,16)
  self.B=mlp(24,16,32);self.D=mlp(16,24,32)
 def forward(self,x):return self.E(x),self.F(x)
def vectors(h,plans):
 return torch.stack([torch.cat([h,plan.reshape(-1)/16]) for plan in plans]).float()
def read_fit():
 support=json.loads((S/'prep/data-support.json').read_text());rr=read(S/'prep/archived-fit.ptz',support['archive'])
 ff=S/'runs/fit-03';f=json.loads((ff/'fit.json').read_text());assert f['status']=='FIT_COMPLETE';rr+=read(ff/'fit-records.ptz',f['archive'])
 assert len(rr)==24 and len({r['parent'] for r in rr})==6 and all(r['y'] is not None for r in rr)
 X=torch.stack([torch.cat([r['h'],r['plan'].reshape(-1)/16]) for r in rr]);Y=torch.stack([r['y'] for r in rr])/torch.tensor(SCALES)
 return rr,X,Y
def load_models():
 out=S/'runs/learn-04';manifest=json.loads((out/'head-manifest.json').read_text());packet=read(out/'heads.ptz',manifest);models=Models(packet['input_dim']);models.load_state_dict(packet['models']);models.eval().requires_grad_(False);packet['manifest']=manifest
 return models,packet
def predict(models,p,h,plans):
 X=vectors(h,plans);X=((X-p['center'])/p['scale']).clamp(-10,10)
 with torch.no_grad():ye,fi=models(X);yi=models.D(fi)
 return dict(E=ye.tolist(),I=yi.tolist()),dict(E=ye,I=fi)
def select(models,p,preds,g,method):
 gg=torch.tensor(g)/torch.tensor(SCALES)
 with torch.no_grad():
  if method=='E':scores=(preds['E']-gg).square().mean(-1)
  else:scores=(preds['I']-models.B(gg[None])).square().mean(-1)
 return int(scores.argmin())
def main():
 p=argparse.ArgumentParser();p.add_argument('--restore-only',action='store_true');a=p.parse_args();t=time.monotonic()
 out=S/'runs/learn-04';out.mkdir(exist_ok=True)
 if a.restore_only:
  models,p=load_models();rr,X,Y=read_fit();xx=((X-p['center'])/p['scale']).clamp(-10,10);outputs=models(xx)
  assert tree_digest(outputs)==p['final_output_digest']
  opt=torch.optim.Adam(models.E.parameters(),lr=.001);opt.load_state_dict(p['optimizer_E']);assert tree_digest(opt.state_dict())==tree_digest(p['optimizer_E'])
  oi=torch.optim.Adam(list(models.F.parameters())+list(models.B.parameters())+list(models.D.parameters()),lr=.001);oi.load_state_dict(p['optimizer_I']);assert tree_digest(oi.state_dict())==tree_digest(p['optimizer_I'])
  from dual_track_task import restore_global_rng
  restore_global_rng(p['rng']);assert tree_digest(global_rng())==tree_digest(p['rng'])
  gen=torch.Generator();gen.set_state(p['sampler_rng']);assert torch.equal(gen.get_state(),p['sampler_rng'])
  write(out/'restore.json',dict(status='FRESH_PROCESS_MODEL_OPT_OUTPUT_EXACT',wall_s=time.monotonic()-t));return
 rr,X,Y=read_fit();center=X.mean(0);scale=X.std(0,unbiased=False).clamp_min(.01);xx=((X-center)/scale).clamp(-10,10)
 models=Models(X.shape[1]);ep=sum(p.numel() for p in models.E.parameters());ip=sum(p.numel() for n,p in models.named_parameters() if not n.startswith('E.'));assert ep<541441 and ip<541441
 assert torch.equal(models.E[0].weight,models.F[0].weight) and torch.equal(models.E[2].weight,models.F[2].weight)
 oe=torch.optim.Adam(models.E.parameters(),lr=.001);oi=torch.optim.Adam(list(models.F.parameters())+list(models.B.parameters())+list(models.D.parameters()),lr=.001)
 parents=sorted({r['parent'] for r in rr});by=[[i for i,r in enumerate(rr) if r['parent']==pa] for pa in parents];gen=torch.Generator().manual_seed(660202)
 history=[];schedule=[]
 with torch.enable_grad():
  for k in range(1024):
   ix=torch.tensor([by[j%6][int(torch.randint(4,(1,),generator=gen))] for j in torch.randperm(24,generator=gen).tolist()]);schedule.append(ix)
   y=Y[ix];xe=xx[ix];ye=models.E(xe);le=(ye-y).square().mean();oe.zero_grad();le.backward();oe.step()
   b=models.B(y);f=models.F(xe);recon=models.D(b);pred=models.D(f)
   std=(b.var(0,unbiased=False)+1e-4).sqrt();var=torch.relu(1-std).mean();bb=b-b.mean(0);cov=bb.T@bb/max(1,len(bb)-1);cv=(cov-torch.diag(torch.diag(cov))).square().sum()/16
   li=(pred-y).square().mean()+(f-b).square().mean()+(recon-y).square().mean()+.1*var+.1*cv
   oi.zero_grad();li.backward();oi.step()
   if k in (0,127,255,511,1023):history.append(dict(group=k+1,E_loss=float(le.detach()),I_loss=float(li.detach()),variance_penalty=float(var.detach()),covariance_penalty=float(cv.detach())))
 with torch.no_grad():
  ye,fi=models(xx);yi=models.D(fi);b=models.B(Y);spec=torch.linalg.svdvals(b-b.mean(0));prob=spec.square()/spec.square().sum().clamp_min(1e-12);rank=float(torch.exp(-(prob*prob.clamp_min(1e-12).log()).sum()))
  # Phase-only/random encoders are offline diagnostics on the exact same real outcome labels.
  rg=torch.Generator().manual_seed(660203);W=torch.randn(24,16,generator=rg)/np.sqrt(24);randomB=Y@W
  retrieval={}
  for name,Z in [('raw',Y),('randomB',randomB),('B_I',b)]:
   hit=[];admissible=0
   for i,r in enumerate(rr):
    eligible=[j for j,s in enumerate(rr) if s['parent']!=r['parent'] and s['phase']==r['phase']]
    if not eligible:continue
    j=min(eligible,key=lambda j:float((Z[i]-Z[j]).square().sum()));true=min(eligible,key=lambda j:float((Y[i]-Y[j]).square().sum()));hit.append(j==true);admissible+=1
   retrieval[name]=dict(same_phase_cross_parent_n=admissible,raw_nearest_agreement=float(np.mean(hit)) if hit else None)
 phasepred=torch.stack([Y[[j for j,s in enumerate(rr) if s['parent']!=r['parent'] and s['phase']==r['phase']]].mean(0) if any(s['parent']!=r['parent'] and s['phase']==r['phase'] for s in rr) else Y[[j for j,s in enumerate(rr) if s['parent']!=r['parent']]].mean(0) for r in rr])
 diagnostics=dict(fit_only=True,E_future_standardized_MSE=float((ye-Y).square().mean()),I_future_standardized_MSE=float((yi-Y).square().mean()),B_centered_variance=b.var(0,unbiased=False).tolist(),B_singular_values=spec.tolist(),B_effective_rank=rank,retrieval=retrieval,phase_only_cross_parent_MSE=float((phasepred-Y).square().mean()),same_state_command_wrist_spread_m=[float(torch.cdist(torch.stack([r['y'][:3] for r in rr if r['parent']==p]),torch.stack([r['y'][:3] for r in rr if r['parent']==p])).max()) for p in parents],weights_fixed_before_evaluation=True)
 packet=dict(models=models.state_dict(),optimizer_E=oe.state_dict(),optimizer_I=oi.state_dict(),rng=global_rng(),sampler_rng=gen.get_state(),sampler_schedule=torch.stack(schedule),center=center,scale=scale,input_dim=X.shape[1],train_parents=parents,group=1024,final_output_digest=tree_digest((ye,fi)),source_fit_archives=[{k:r[k] for k in ('parent','command','K')} for r in rr],params=dict(E=ep,I=ip),diagnostics=diagnostics)
 manifest=dump(out/'heads.ptz',packet);write(out/'head-manifest.json',manifest);write(out/'learning.json',dict(status='FIXED_ENDPOINT_COMPLETE',history=history,diagnostics=diagnostics,parameters=packet['params'],wall_s=time.monotonic()-t,updates_each=1024))
 print('H0_HEADS_COMPLETE',json.dumps(diagnostics),flush=True)
if __name__=='__main__':main()
