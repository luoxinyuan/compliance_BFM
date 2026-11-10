"""Fixed precontact dual rule; no future/task labels enter the actor."""
import copy
import torch

class Retention:
    def __init__(self,net,enabled):
        self.initial=copy.deepcopy(net).eval().requires_grad_(False)
        self.enabled=enabled;self.coefficient=.1 if enabled else 0.
        self.reference=None

    def seed_reference(self,x,mask):
        if self.reference is None:
            rows=x[mask]
            assert len(rows)>0,'First real batch has no legal precontact reference'
            ix=torch.linspace(0,len(rows)-1,min(512,len(rows)),device=x.device).long().unique()
            self.reference=rows[ix].detach().clone()

    def divergence(self,net,x):
        with torch.no_grad():base=self.initial.distribution(x).mean
        return (net.distribution(x).mean-base).square().sum(-1)/(2*.008**2)

    def loss(self,net,x,mask):
        if not self.enabled:return torch.zeros((),device=x.device)
        values=[self.divergence(net,self.reference).mean()]
        if bool(mask.any()):values.append(self.divergence(net,x[mask]).mean())
        return self.coefficient*torch.stack(values).mean()

    @torch.no_grad()
    def after_batch(self,net,x,mask):
        values=self.divergence(net,self.reference)
        current=self.divergence(net,x[mask]) if bool(mask.any()) else values[:0]
        mean=torch.stack([values.mean(),current.mean()]).mean() if len(current) else values.mean()
        all_values=torch.cat([values,current])
        before=self.coefficient
        if self.enabled:self.coefficient=max(0.,min(100.,before+.1*(float(mean)/.01-1.)))
        return dict(reference_n=len(values),on_policy_precontact_n=len(current),mean_D_pre=float(mean),
            p95_D_pre=float(torch.quantile(all_values,.95)),max_D_pre=float(all_values.max()),
            coefficient_used=before,coefficient_next=self.coefficient,target=.01,
            fixed_dual_rule='clip(lambda+.1*(meanD/.01-1),0,100); once per complete PPO batch')

    def state_dict(self):
        return dict(coefficient=self.coefficient,reference=self.reference,enabled=self.enabled)

    def load_state_dict(self,x):
        assert x['enabled']==self.enabled
        self.coefficient=x['coefficient'];self.reference=x['reference'].to(next(self.initial.parameters()).device)
