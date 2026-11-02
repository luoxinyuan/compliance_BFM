import tempfile,json,math
from pathlib import Path
import torch
from m11_selection import Selection,sha,exploration_scale,continuation_rank
from m11_policy import Expert,update

def test_selection_resume_scores_do_not_alias():
 with tempfile.TemporaryDirectory() as t:
  p=Path(t);old=p/'historical.pt';old.write_bytes(b'historical best');old_sha=sha(old)
  current=p/'current.pt';current.write_bytes(b'weaker current resumed weights');ev=p/'current-eval.json';ev.write_text('{"score":7}')
  s=Selection(p,{'protocol':'new evaluation scope'});assert s.consider(current,[0,7],ev);assert s.record['rank']==[0,7] and s.record['checkpoint_sha256']==sha(current)
  bad=p/'bad.pt';bad.write_bytes(b'even worse');assert not s.consider(bad,[0,8],ev);assert sha(p/'best.pt')==sha(current)
  good=p/'good.pt';good.write_bytes(b'improved');gev=p/'good-eval.json';gev.write_text('{"score":3}');assert s.consider(good,[0,3],gev);assert sha(p/'best.pt')==sha(good);assert s.record['evaluation_sha256']==sha(gev);assert sha(old)==old_sha

def test_external_std_batch_contract_and_resume():
 torch.manual_seed(13);net=Expert(11);opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=3e-5),dict(params=[net.log_std],lr=1e-5),dict(params=net.critic.parameters(),lr=3e-4)]);net.log_std.requires_grad_(False);opt.param_groups[1]['lr']=0
 x=torch.randn(768,11);original=net.log_std.detach().clone()
 for factor in [1.,.25,.125]: # below historical exp(-4) clamp is explicitly handled
  with torch.no_grad():net.log_std.copy_(original+math.log(factor))
  d=net.distribution(x);raw=d.sample();lp=d.log_prob(raw).sum(-1).detach();std=net.log_std.detach().clone();v=net.value(x).detach()
  batch=dict(obs=x,action=raw,logp=lp,value=v,advantages=torch.randn(768),returns=v+torch.randn(768),raw_adv_std=torch.tensor(1.))
  stats=update(net,opt,batch,epochs=1);assert stats['pre_logprob_max_error']<3e-4;assert torch.equal(std,net.log_std);assert all(g['std']==0 for g in stats['gradients'])
  batch['logp']=net.distribution(x).log_prob(raw).sum(-1).detach()
  with torch.no_grad():net.log_std.add_(math.log(.9))
  try:update(net,opt,batch,epochs=1)
  except AssertionError:pass
  else:raise AssertionError('Changing std after sampling must fail old probability verification')
 assert exploration_scale('fixed',159)==1 and abs(exploration_scale('annealed',120)-.25)<1e-12
 assert exploration_scale('annealed',19)==1 and exploration_scale('annealed',60)>exploration_scale('annealed',119)>.25

def test_tail_selection():
 def row(name,cost):return dict(case=name,force_rms=cost,error_cm=0,cap_fraction=0,boundary=None,condition=dict(enabled=True,family='walking'))
 b={n:row(n,2) for n in ['a','b','c']};good,_=continuation_rank([row('a',2),row('b',2),row('c',2)],b);tail,_=continuation_rank([row('a',0),row('b',0),row('c',6)],b);assert tail>good

if __name__=='__main__':
 test_selection_resume_scores_do_not_alias();test_external_std_batch_contract_and_resume();test_tail_selection();print('M11 continuation tests passed: score-weight-SHA binding, frozen std likelihood, clamp, gradients, tail selection')
