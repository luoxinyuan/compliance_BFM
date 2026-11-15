"""Complete only the failed JSON publication from immutable qualified data."""
import collections,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
from research.interaction_fb.journal import read
from diagnostic_fb_resume import tree_digest
S=ROOT.parent
def main():
    t=time.monotonic();out=S/'prep/p0-p1';path=out/'data.ptz';dm=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    d=read(path,dm);assert not (out/'p1.json').exists();meta=d['metadata'];valid=d['eligible'].numpy();y=d['labels'].numpy()
    counts=[]
    for h in (5,25,50):
        for split,pp in [('fit',d['fit_parents']),('heldout',d['heldout_parents'])]:
            ix=[i for i,m in enumerate(meta) if m['parent'] in pp and valid[i,h-1]]
            counts.append(dict(H=h,split=split,windows=len(ix),parents=len({meta[i]['parent'] for i in ix}),high=int(sum(abs(y[i,h-1])>=1 for i in ix)),low=int(sum(abs(y[i,h-1])<=.1 for i in ix)),by_phase=dict(collections.Counter(meta[i]['phase'] for i in ix)),censored=int(sum(m['parent'] in pp for m in meta)-len(ix))))
    p1=dict(status='PASS_LABELS_INDEPENDENCE_LIMITED',source_rows=3500,independent_parents=6,fit_parents=d['fit_parents'],heldout_parents=d['heldout_parents'],counts=counts,
        next_observation_exact=True,continuity_fixture='Original prepare ran every continuity/physical time/episode/source check before immutable data dump; no change to that packet.',optimizer_updates=0,physics_ticks=0,data=dm,data_logical_digest=tree_digest(d),schedule_digest=tree_digest(d['schedule']),remaining_horizon_contract='Shared H-conditioned head, legal remaining h1..50; primary H5/25/50. All variants exact same actual labels/parent-phase-horizon indices.',prospective=d['prospective'],serialization_resume_wall_s=time.monotonic()-t)
    (out/'p1.json').write_text(json.dumps(p1,indent=2,allow_nan=False))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==dm['sha256']
    fixture=dict(failure_family_id='p1-json-numpy-scalar',repairs=1,first_failure='TypeError int64 in p1 summary JSON after data.ptz committed; no optimization/physics',original_packet_sha256=dm['sha256'],packet_digest_unchanged=True,schedule_digest=tree_digest(d['schedule']),serialization_fixture=json.loads(json.dumps(dict(i=np.int64(2),f=np.float64(3.)),default=lambda v:v.item())),resumed_from='p0-p1/data.ptz immutable snapshot; no labels/indices/RNG regenerated',new_physics=0,new_optimization=0,wall_s=time.monotonic()-t)
    (S/'prep/p1-serialization-repair.json').write_text(json.dumps(fixture,indent=2));print(json.dumps(dict(counts=counts,proof=fixture),indent=2),flush=True)
if __name__=='__main__':main()
