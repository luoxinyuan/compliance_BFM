"""Approved nominal-anchored Envelope with deterministic inward FP32 rounding.
Only numeric representation is rounded; physical limits and four constants unchanged.
"""
import torch
class ExecutableValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx,analytic,actual):return actual.clone()
    @staticmethod
    def backward(ctx,gradient):return gradient,None
class Envelope:
    def __init__(self,config,device):
        self.config=config;rows=config['mechanical_table']
        self.scale=torch.tensor([[r['scale'] for r in rows]],dtype=torch.float32,device=device)
        self.offset=torch.tensor([[r['offset'] for r in rows]],dtype=torch.float32,device=device)
        self.qmin=torch.tensor([[r['qmin'] for r in rows]],dtype=torch.float32,device=device)
        self.qmax=torch.tensor([[r['qmax'] for r in rows]],dtype=torch.float32,device=device)
        margin=torch.maximum(torch.tensor(.02,device=device),.05*(self.qmax-self.qmin))
        self.lo=self.qmin+margin;self.hi=self.qmax-margin
        assert bool((self.lo<=self.hi).all()) and bool((self.scale>0).all())
    def g(self,a):return self.offset+(a*5).clamp(-5,5)*self.scale/5
    def __call__(self,base,raw):
        b=base.clamp(-1,1);r=raw.clamp(-1,1);q0=self.g(b);qr=self.g(r)
        dm=torch.minimum(torch.full_like(q0,.12),.25*(q0-self.lo).clamp_min(0))
        dp=torch.minimum(torch.full_like(q0,.12),.25*(self.hi-q0).clamp_min(0))
        dq=qr-q0;ds=torch.maximum(torch.minimum(dq,dp),-dm);requested=q0+ds
        analytic=(((requested-self.offset)*5/self.scale)/5).clamp(-1,1)
        # Preserve exact original actions whenever the actual q delta is permitted.
        with torch.no_grad():
            exact_delta=qr.double()-q0.double()
            unchanged=(exact_delta>=-dm.double())&(exact_delta<=dp.double())
            zero=requested==q0
            actual=torch.where(unchanged,r,torch.where(zero,b,analytic)).detach()
            for iteration in range(9):
                qa=self.g(actual);delta=qa.double()-q0.double()
                valid=(delta>=-dm.double())&(delta<=dp.double())
                # Representable output must be toward nominal, never outside ideal projection.
                directional=torch.where(requested>=q0,(qa<=requested)&(qa>=q0),(qa>=requested)&(qa<=q0))
                valid=valid&(unchanged|directional)
                if bool(valid.all()):break
                if iteration==8:
                    # Offset cancellation can place many action ULPs inside one q ULP.
                    # Monotone bit-space bisection returns the nearest permitted float,
                    # not a guessed tolerance or a scientific-envelope contraction.
                    def order(x):
                        u=x.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
                        return torch.where((u&0x80000000)!=0,(~u)&0xffffffff,u^0x80000000)
                    def unorder(k):
                        u=torch.where((k&0x80000000)!=0,k^0x80000000,(~k)&0xffffffff)
                        return u.to(torch.int32).view(torch.float32)
                    good=order(b);bad=order(actual);fixed=valid.clone()
                    for _ in range(32):
                        middle=(good+bad)//2;trial=unorder(middle);tq=self.g(trial);td=tq.double()-q0.double()
                        ok=(td>=-dm.double())&(td<=dp.double())
                        ok=ok&torch.where(requested>=q0,(tq<=requested)&(tq>=q0),(tq>=requested)&(tq<=q0))
                        moving=(good-bad).abs()>1
                        good=torch.where(moving&ok,middle,good);bad=torch.where(moving&~ok,middle,bad)
                    actual=torch.where(fixed,actual,unorder(good));qa=self.g(actual);delta=qa.double()-q0.double()
                    assert bool(((delta>=-dm.double())&(delta<=dp.double())).all())
                    break
                actual=torch.where(valid,actual,torch.nextafter(actual,b))
            error=(qa.double()-requested.double()).abs()
            assert bool((error<=1e-6).all()),'Pure numeric correction exceeds authorized 1e-6rad'
            assert bool(torch.isfinite(actual).all()) and bool((actual.abs()<=1).all())
        # Explicit straight-through derivative of the approved analytic projection;
        # float32 rounding has no useful classical derivative. Forward is exactly actual.
        analytic=torch.where(unchanged,r,torch.where(zero,b,analytic))
        executed=ExecutableValue.apply(analytic,actual)
        audit=dict(base=b.detach(),raw=r.detach(),requested=requested.detach(),qbase=q0.detach(),
                   qexec=self.g(executed).detach(),executed=executed.detach(),
                   dminus=dm.detach(),dplus=dp.detach(),rounding_error=error.float(),
                   triggered=(r!=actual),numeric_steps=torch.full((len(actual),1),iteration,device=actual.device,dtype=torch.int64))
        return executed,audit
