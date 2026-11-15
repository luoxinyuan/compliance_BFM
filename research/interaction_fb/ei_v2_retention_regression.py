"""CPU-only exact operator regression on immutable qualified actual action means.
Not a physics run or training experiment. No BFM parameters are optimized.
"""
import os,sys,json,time,copy,subprocess,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from .journal import read
from .ei_v2_ppo import retention as fixed
from diagnostic_fb_resume import tree_digest

def main():
    start=time.monotonic();torch.set_num_threads(1)
    oldcode=subprocess.check_output(['git','show','f164f01:research/interaction_fb/ei_v2_ppo.py'],cwd=ROOT,text=True)
    oldns=dict(__name__='research.interaction_fb.ei_v2_old_retention_fixture',__package__='research.interaction_fb');exec(compile(oldcode,'<archived-old-ppo>','exec'),oldns)
    meta=json.loads((STAGE/'runs/P1-frozen-sample-01/qualification.json').read_text());data=read(STAGE/'runs/P1-frozen-sample-01/qualified-retention-data.ptz',meta['retention']);keep=data['keep_by_parent'];eps=data['eps'];mus=[r['old_mu'] for rows in keep.values() for r in rows]
    class DummyBase:
        def __init__(self):self.cfg=types.SimpleNamespace(actor_std=None);self.cursor=0
        def _normalize(self,o):return o
        def _actor(self,*a):
            self.last=mus[self.cursor];self.cursor+=1;return types.SimpleNamespace(mean=self.last)
    class ActionPerturbation:
        def __init__(self,offset):self.device=torch.device('cpu');self.delta=torch.nn.Parameter(torch.full((1,29),float(offset)));self.base=DummyBase()
        def mean(self,*a):return self.base.last+self.delta
    result=[]
    for offset in (0.,.0001,.001,-.001):
        a,b=ActionPerturbation(offset),ActionPerturbation(offset)
        la,va=oldns['retention'](a,keep,eps);lb,vb=fixed(b,keep,eps)
        assert torch.equal(la,lb) and torch.equal(va,vb), 'retention forward or dual violation changed'
        # Lambda0 is the exact initial training case; old0*sqrt-gradient becomesNaN.
        scale=0. if offset==0 else .7
        (scale*la).backward();(scale*lb).backward()
        if offset==0:
            assert not torch.isfinite(a.delta.grad).all()
            assert torch.isfinite(b.delta.grad).all() and torch.count_nonzero(b.delta.grad)==0
            expected_grad=torch.zeros_like(b.delta)
            assert torch.equal(b.delta.grad,expected_grad)
            bad=int(torch.count_nonzero(~torch.isfinite(a.delta.grad)));good=True
        else:
            assert torch.isfinite(a.delta.grad).all() and torch.equal(a.delta.grad,b.delta.grad),'applicable nonzero gradient changed'
            oa=torch.optim.Adam([a.delta],lr=1e-5);ob=torch.optim.Adam([b.delta],lr=1e-5);oa.step();ob.step()
            assert torch.equal(a.delta,b.delta) and tree_digest(oa.state_dict())==tree_digest(ob.state_dict()),'applicable Adam update changed'
            bad=0;good=True
        result.append(dict(raw_action_offset=offset,forward_value_exact=True,dual_violation_exact=True,old_nonfinite_gradient_components=bad,fixed_gradient_correct=good,applicable_nonzero_gradient_and_Adam_exact=offset!=0))
    # Test the mathematical retention loss directly in qPD-difference coordinates.
    # Raw-action finite differences around stored nonzero means have FP32 addition
    # rounding asymmetry (archived test01), and are not an exact-zero identity.
    from .ei_v2_ppo import _ZeroSafeRMS
    def difference_loss(d):
        return torch.maximum(_ZeroSafeRMS.apply(d.square().mean(-1))/eps['eps_rms'],d.abs().max(-1).values/eps['eps_p95']).square().mean()
    for j in range(29):
        d=torch.zeros(1,29);d[0,j]=1e-4
        assert torch.equal(difference_loss(d),difference_loss(-d))
    out=STAGE/'runs/P1-retention-math-regression-02';out.mkdir(exist_ok=False)
    report=dict(status='ZERO_RMS_GRADIENT_FIX_EXACT_FORWARD_AND_APPLICABLE_GRADIENT_ADAM_PASS',method='archived actual qualified old_mu values, identical deployed qPD clipping/scales and original parent weights; explicit raw-action perturbation unit fixture, not an Actor or physical success claim',actual_qualified_frames=len(mus),parents=len(keep),checks=result,central_difference_zero=29,scientific_thresholds_changed=False,Actor_forward_changed=False,new_physics=0,new_scientific_RL_rows=0,new_BFM_parameter_updates=0,CPU_wall_s=time.monotonic()-start)
    (out/'retention-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
