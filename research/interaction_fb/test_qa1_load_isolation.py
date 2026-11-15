"""Small CPU Adam regression. No BFM/physics or scientific optimizer groups."""
import copy,torch

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(636413)
        original=torch.nn.Linear(3,2);o=torch.optim.Adam(original.parameters(),lr=.0001)
        x=torch.randn(8,3);original(x).square().sum().backward();o.step();o.zero_grad(set_to_none=True)
        weights=copy.deepcopy(original.state_dict());pristine=copy.deepcopy(o.state_dict());before=copy.deepcopy(pristine)
        regular=torch.nn.Linear(3,2);regular.load_state_dict(weights);ro=torch.optim.Adam(regular.parameters(),lr=.0001);ro.load_state_dict(copy.deepcopy(pristine))
        fixed=torch.nn.Linear(3,2);fixed.load_state_dict(weights);fo=torch.optim.Adam(fixed.parameters(),lr=.0001);fo.load_state_dict(copy.deepcopy(pristine))
        for net,opt in ((regular,ro),(fixed,fo)):
            opt.zero_grad(set_to_none=True);net(x).square().sum().backward();opt.step()
        assert same(regular.state_dict(),fixed.state_dict()) and same(ro.state_dict(),fo.state_dict())
        assert same(pristine,before),'Source Adam changed after actual isolated update'
        fresh=torch.nn.Linear(3,2);fresh.load_state_dict(weights);new=torch.optim.Adam(fresh.parameters(),lr=.0001);new.load_state_dict(copy.deepcopy(pristine))
        assert same(new.state_dict(),before) and same(fresh.state_dict(),weights)
        for index,record in pristine['state'].items():
            for key,tensor in record.items():assert new.state_dict()['state'][index][key].data_ptr()!=tensor.data_ptr()
    assert not torch.cuda.is_initialized()
    print('PASS CPU Adam isolated update + exact next reload + unchanged source + optimizer math unchanged; CUDA_initialized=false BFM_Q_groups=0 physics=0')
if __name__=='__main__':main()
