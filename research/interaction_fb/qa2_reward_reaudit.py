"""P0 supplemental actual reward component and quantization audit, read-only."""
import json,sys,time
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(1)
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');S=P/'experiments/r2-qa2-value-20261115';R=S/'runtime-value'
sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.journal import read
from dual_track_task import native_terms
start=time.monotonic();G=P/'experiments/r2-v11-qentry-20261114/runs/paired-Q-ranking-12';qual=json.loads((G/'qualification.json').read_text());errors={};quantized_exact=0;components=[]
for st in qual['results']:
    for br in st['branches']:
        tr=read(G/f"round-{st['round']}"/(br['id']+'.ptz'),br['archive']);cs=tr['controls'];ph=tr['physics'];rows=tr['rows'];assert len(rows)==len(cs)==50
        # Only heading uses distance; original zero-yaw source all current target yaw0.
        spec=next(p['specs'][0] for p in json.loads((R/'research_notes/r2_splits/train.json').read_text())['parents'] if p['parent_id']==st['parent_id'])
        sums={};rewards=[]
        for j,c in enumerate(cs):
            rr=ph[j*4:j*4+4];dt=len(rr)*.005;prev=cs[j-1]['q_PD'] if j else c['q_PD']
            actual=native_terms(rr,c['q_PD'],prev,spec)
            for k,v in actual.items():
                if j==0 and k=='continuity':continue
                errors[k]=max(errors.get(k,0.),abs(v-c['reward_terms'][k]))
            expected=-dt*sum(c['reward_terms'].values());label=rows[j]['aux_rewards']['contact_physical']
            assert torch.equal(torch.tensor([[expected]],dtype=label.dtype),label),('reward label differs beyond recorded dtype quantization',st['round'],br['id'],j)
            quantized_exact+=1;rewards.append(float(label))
            for k,v in c['reward_terms'].items():sums[k]=sums.get(k,0.)+dt*v
        weight=1.;Gval=0.
        for row,reward in zip(rows,rewards):Gval+=weight*reward;weight*=.99**(float(row['duration'])/.1)
        assert abs(Gval-br['discounted_return'])<1e-12
        components.append(dict(round=st['round'],parent=st['parent_id'],candidate=br['id'],cost_integrals=sums,discounted_return=Gval))
assert max(errors.values())<=1e-12
ans=dict(status='PASS',actual_20ms_labels_exact_after_dtype_quantization=quantized_exact,component_max_difference=errors,first_control_continuity='Archived actual continuity retained; no invented pre-start q_PD. Other49 controls per branch recomputed same previous q_PD.',branches=components,actual_new_physics=0,updates=0,cuda_initialized=torch.cuda.is_initialized(),wall_s=time.monotonic()-start)
(S/'prep/reward-reaudit.json').write_text(json.dumps(ans,indent=2));print(json.dumps({k:v for k,v in ans.items() if k!='branches'},indent=2))
