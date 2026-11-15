"""CPU mathematical fixtures and fixed-endpoint identity checks, no optimizer/physics."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(1)
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'research_notes')];S=R.parent
from research.interaction_fb.journal import read
from research.interaction_fb.qa2_value import metrics
from diagnostic_fb_resume import tree_digest
def load(n):
    p=S/'runs'/n/(n+'.ptz');m=json.loads(p.with_suffix('.manifest.json').read_text());return read(p,m),m
def main():
    start=time.monotonic();a,am=load('V1-MC');b,bm=load('V2-n50')
    for key in ['model','target','optimizer','cursor','source_manifest','source_replay','schedule_digest']:assert tree_digest(a[key])==tree_digest(b[key]),key
    # G2 scalar fixtures and parent-balanced rather than frame-weighted aggregation.
    y=np.array([-6.380,-6.380,-.0325,-.0325]);meta=[dict(parent=str(i)) for i in range(4)]
    good=np.array([-6.,-6.,-.04,-.04]);bad=np.array([-1.535,-1.535,-.8,-.8])
    g=metrics(good,y,meta,list(range(4)));f=metrics(bad,y,meta,list(range(4)))
    assert g['numerical_bounds'] and not f['numerical_bounds']
    assert abs(g['high_macro_nmae']-.38/6.38)<1e-12
    # Fixed 20% incremental-information gate: clear improvement accepts; mean-only rejects.
    def improvement(model,global_error,phase_error):
        baseline=min(global_error,phase_error)
        return (baseline-model)/baseline if baseline>0 else None
    assert improvement(.10,.25,.20)>=.20 and improvement(.20,.25,.20)<.20
    models={}
    for n in ['V1-MC','V2-n1','V2-n5','V2-n25','V2-n50']:
        ss,mm=load(n);assert ss['cursor']==1024;models[n]=dict(sha256=mm['sha256'],logical_digest=mm['logical_digest'],schedule=ss['schedule_digest'],source_manifest=ss['source_manifest'])
    ans=dict(status='PASS',MC_vs_n50_model_target_Adam_exact=True,G2_good_fixture=g,G2_bad_fixture=f,baseline_improvement_fixtures=True,models=models,actual_optimizer_updates=0,physics_ticks=0,wall_s=time.monotonic()-start)
    (S/'prep/target-fixtures.json').write_text(json.dumps(ans,indent=2));print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
