"""Signed OR1: one actual Adam proposal, fixed parameter backtracking, full transaction.
This does not change sampling, probability space, Adam moments, reward or physical gates.
"""
import copy,math,random
import numpy as np
import torch

ALPHAS=(1.,.5,.25,.125,.0625)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def candidate(before,proposal,alpha):
    # alpha1 must preserve original bytes; FP32 cancellation is not an identity.
    if alpha==1.:return {k:v.clone() for k,v in proposal.items()}
    out={}
    for k,v in before.items():
        p=proposal[k]
        if torch.is_floating_point(v) or torch.is_complex(v):out[k]=v+alpha*(p-v)
        else:
            assert torch.equal(v,p),('changed nonfloating buffer',k)
            out[k]=v.clone()
    return out

def finite_tree(x):
    if torch.is_tensor(x):return bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
    if isinstance(x,(tuple,list)):return all(finite_tree(v) for v in x)
    if isinstance(x,float):return math.isfinite(x)
    return True

def gradient_by_module(module):
    out={}
    for n,p in module.named_parameters():
        k=n.split('.')[0];s=out.setdefault(k,dict(sum_square=0.,max_abs=0.,nonzero=0,tensors=0))
        if p.grad is not None:
            g=p.grad.detach();s['sum_square']+=float(g.double().square().sum());s['max_abs']=max(s['max_abs'],float(g.abs().max()));s['nonzero']+=int((g!=0).sum());s['tensors']+=1
    for s in out.values():s['norm']=math.sqrt(s.pop('sum_square'))
    return out

def one_proposal(module,opt,loss_function,evaluate,*,commit=False,evaluate_all=False,expected_first_KL=None):
    """loss_function -> scalar; evaluate -> objective,KL,extra on fixed actual data.

    evaluate uses no_grad. No random draw or optimizer call per alpha. Full new Adam
    moments are committed once with selected scaled parameters; all rejected state
    is restored. On exception, caller preserves the pending transaction separately.
    """
    before=copy.deepcopy(module.state_dict());oldopt=copy.deepcopy(opt.state_dict());before_rng=rng_state()
    oldgrads={n:None if p.grad is None else p.grad.detach().clone() for n,p in module.named_parameters()}
    with torch.no_grad():baseline=evaluate(0.)
    opt.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss=loss_function();assert bool(torch.isfinite(loss));loss.backward()
        pre=gradient_by_module(module)
        gn=float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.));assert math.isfinite(gn)
        post=gradient_by_module(module);opt.step()
    proposal=copy.deepcopy(module.state_dict());newopt=copy.deepcopy(opt.state_dict());assert finite_tree(proposal) and finite_tree(newopt)
    trials=[];chosen=None;chosen_state=None
    for alpha in ALPHAS:
        c=candidate(before,proposal,alpha);module.load_state_dict(c,strict=True)
        if alpha==1.:assert exact(module.state_dict(),proposal)
        with torch.no_grad():m=evaluate(alpha)
        if alpha==1. and expected_first_KL is not None:assert exact(m['KL'],expected_first_KL),('archived first proposal KL did not reproduce',m['KL'],expected_first_KL)
        accept=(finite_tree(m) and m['KL']['joint']<=.02 and m['objective']<=baseline['objective'])
        m.update(alpha=alpha,inside_KL=bool(math.isfinite(m['KL']['joint']) and m['KL']['joint']<=.02),objective_nonincrease=bool(m['objective']<=baseline['objective']),acceptable=bool(accept));trials.append(m)
        if accept and chosen is None:chosen=alpha;chosen_state=c
        if accept and not evaluate_all:break
    assert exact(before_rng,rng_state()),'proposal or candidate evaluation consumed RNG'
    module.load_state_dict(before,strict=True);opt.load_state_dict(oldopt)
    assert exact(module.state_dict(),before) and exact(opt.state_dict(),oldopt)
    if commit and chosen is not None:
        module.load_state_dict(chosen_state,strict=True);opt.load_state_dict(newopt)
        assert exact(module.state_dict(),chosen_state) and exact(opt.state_dict(),newopt)
    else:
        for n,p in module.named_parameters():p.grad=oldgrads[n]
    assert exact(before_rng,rng_state())
    return dict(accepted=bool(commit and chosen is not None),selected_alpha=chosen,baseline=baseline,trials=trials,grad_before_clip=pre,grad_after_clip=post,total_grad_before_clip=gn,model_Adam_transaction_exact=True,candidate_RNG_unchanged=True,committed=bool(commit and chosen is not None))
