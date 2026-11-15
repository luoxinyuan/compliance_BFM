"""Descriptive current-body/load matched R1a; frozen weights, no new samples/thresholds."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from research.interaction_fb.h0_common import *
from research.interaction_fb.h0_learn import read_fit,load_models
torch.set_num_threads(1)
def main():
 start=time.monotonic();rr,X,Y=read_fit();models,packet=load_models();parents=sorted({r['parent'] for r in rr});first={p:next(r for r in rr if r['parent']==p) for p in parents}
 def force(r):return r['h'][721:1621].reshape(100,3,3)[-1,0]*5
 matching=[];pairs=[]
 with torch.no_grad():B=models.B(Y)
 gen=torch.Generator().manual_seed(660203);randomB=Y@(torch.randn(24,16,generator=gen)/np.sqrt(24));hit={n:[] for n in ['raw','randomB','B_I']}
 for p in parents:
  r=first[p];eligible=[q for q in parents if q!=p and first[q]['phase']==r['phase']]
  if not eligible:
   matching.append(dict(parent=p,phase=r['phase'],status='NO_OTHER_PARENT_IN_PHASE',candidate_n=0));continue
  def distance(q):return float((r['h'][:465]-first[q]['h'][:465]).square().mean()+((force(r)-force(first[q]))/10).square().sum())
  q=min(eligible,key=distance);other=first[q];js=[j for j,s in enumerate(rr) if s['parent']==q]
  matching.append(dict(parent=p,matched_parent=q,phase=r['phase'],distance=distance(q),normalized_proprio_RMS=float((r['h'][:465]-other['h'][:465]).square().mean().sqrt()),body_frame_force_gap_N=float((force(r)-force(other)).norm()),source_force_N=float(force(r).norm()),matched_force_N=float(force(other).norm()),status='NEAREST_AVAILABLE_NOT_ABSOLUTE_QUALIFICATION',candidate_n=len(js)))
  for i,s in enumerate(rr):
   if s['parent']!=p:continue
   true=min(js,key=lambda j:float((Y[i]-Y[j]).square().sum()))
   for name,Z in [('raw',Y),('randomB',randomB),('B_I',B)]:
    chosen=min(js,key=lambda j:float((Z[i]-Z[j]).square().sum()));hit[name].append(chosen==true)
 result=dict(status='DESCRIPTIVE_MATCHED_R1A_COMPLETE',scope='fit-only auxiliary; not preregistered new pass threshold or heldout evidence',matching_rule='same phase; choose nearest different parent by mean-square465D frozen-normalized proprio + squared body-frame measured-force vector difference/10N; all4 commands of that parent remain eligible',matching=matching,nearest_future_raw_agreement={k:dict(n=len(v),fraction=float(np.mean(v))) for k,v in hit.items()},absolute_body_load_proximity_gate='UNMEASURED_NO_PRELOCKED_NUMERIC_THRESHOLD',wall_s=time.monotonic()-start,model_updates=0,physics_ticks=0)
 write(S/'prep/matched-r1a.json',result);pub=json.loads((ROOT/'research_notes/h0_results.json').read_text());pub['matched_r1a']=result;write(ROOT/'research_notes/h0_results.json',pub)
 f=ROOT/'research_notes/H0_REPRESENTATION_AND_CAUSAL_ADDRESS_RESULTS.md'
 with f.open('a') as fd:
  fd.write('\n## R1a补充：相同phase内当前身体/负担最近父组（只读描述性）\n\n没有重训或改变Goal效应判据；按当前可部署本体与实测力选择最接近的另一父组，再比较该组四个实际未来结果。匹配距离和力差完整保留；没有事前绝对邻近阈值，故只称nearest-available，不能把距离大也说成合格近邻。entry仅一fit父组、无跨父组支持；20条来自其余五个父组，仍是fit-only、非20独立父组。\n\n')
  fd.write(json.dumps(result['nearest_future_raw_agreement'],ensure_ascii=False)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
