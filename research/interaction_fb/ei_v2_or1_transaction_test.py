"""CPU-only synthetic regression; never substitutes for real BFM/GPU/physics gates."""
import copy,json,argparse,time
from pathlib import Path
import torch
from .ei_v2_or1_optimizer import one_proposal,candidate,exact

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.monotonic();rows=[]
 for populated in (False,True):
  torch.manual_seed(11);net=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Tanh());opt=torch.optim.Adam(net.parameters(),lr=1e-5,weight_decay=0)
  x=torch.tensor([[.1,.2,-.3],[.4,-.2,.6]])
  def loss():return (net(x)-.7).square().mean()
  if populated:
   loss().backward();opt.step();opt.zero_grad(set_to_none=True)
  before=copy.deepcopy(net.state_dict());old=copy.deepcopy(opt.state_dict())
  def evaluate(alpha):return {'objective':float(loss()),'KL':{'joint':.01 if 0<alpha<=.25 else .1 if alpha>0 else 0.}}
  r=one_proposal(net,opt,loss,evaluate,commit=False,evaluate_all=True)
  assert r['selected_alpha']==.25 and exact(net.state_dict(),before)and exact(opt.state_dict(),old)
  proposed_net=copy.deepcopy(net);proposed_opt=torch.optim.Adam(proposed_net.parameters(),lr=1e-5,weight_decay=0);proposed_opt.load_state_dict(old);((proposed_net(x)-.7).square().mean()).backward();torch.nn.utils.clip_grad_norm_(proposed_net.parameters(),1);proposed_opt.step()
  proposed=copy.deepcopy(proposed_net.state_dict());new=copy.deepcopy(proposed_opt.state_dict())
  expected=candidate(before,proposed,.25)
  r=one_proposal(net,opt,loss,evaluate,commit=True,evaluate_all=False)
  assert r['committed'] and exact(net.state_dict(),expected) and exact(opt.state_dict(),new)
  state=copy.deepcopy(net.state_dict());os=copy.deepcopy(opt.state_dict())
  def reject(alpha):return {'objective':float(loss()),'KL':{'joint':1.}}
  rr=one_proposal(net,opt,loss,reject,commit=True,evaluate_all=False)
  assert not rr['committed'] and exact(state,net.state_dict()) and exact(os,opt.state_dict())
  rows.append(dict(initial_Adam_nonempty=populated,selected_alpha=.25,rollback_full_exact=True,commit_scaled_parameters_full_proposed_moments_exact=True,step_increment_once=True))
 b={'x':torch.tensor([1e-5])};p={'x':torch.tensor([-1e-6])};assert torch.equal(candidate(b,p,1.)['x'],p['x'])
 result=dict(status='SYNTHETIC_CPU_TRANSACTION_PASS_NOT_BFM_QUALIFICATION',cases=rows,alpha1_endpoint_exact=True,wall_after_import_s=time.monotonic()-start,physics_ticks=0,real_on_policy_rows=0)
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
