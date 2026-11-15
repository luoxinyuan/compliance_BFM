"""No physics/optimizer: exact terminal vs truncation and physical-time GAE fixtures."""
import math,copy
import torch
from .ei_v2_ppo import segments,advantages
class V:
    device=torch.device('cpu')
    def v(self,o):return o['x'][:,0]
    def value_input(self,o):return o['x']
def run():
    rows=[]
    for k in range(5):rows.append(dict(duration=torch.tensor([[.02]],dtype=torch.float64),aux_rewards={'contact_physical':torch.tensor([[-1.]])},observation={'x':torch.tensor([[.4]])},next={'observation':{'x':torch.tensor([[.4]])},'terminated':torch.tensor([[False]]),'truncated':torch.tensor([[False]])},macro_slot=k,episode=0))
    expected_R=-sum((.99**.2)**k for k in range(5));v=float(rows[0]['observation']['x'][0,0]);results={}
    for kind in ('continuing','terminated','truncated'):
        r=copy.deepcopy(rows)
        if kind!='continuing':r[-1]['next'][kind]=torch.tensor([[True]])
        a,t,detail=advantages(V(),r,segments(r));expected=expected_R+(0 if kind=='terminated' else .99*v)
        assert t.item()==torch.tensor(expected,dtype=torch.float32).item(),(kind,t.item(),expected)
        results[kind]=dict(return_actual=t.item(),return_expected_FP32=torch.tensor(expected,dtype=torch.float32).item(),discount=detail['discounts'][0],trace=detail['trace_discounts'][0])
    assert results['terminated']['trace']==0 and results['truncated']['trace']==0
    assert results['continuing']['trace']>0
    return dict(status='EXACT_ANALYTIC_TIME_AND_TERMINAL_FIXTURES_PASS',fixtures=results,optimizer_steps=0,physics_ticks=0,probability_tolerance_changed=False)
if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
