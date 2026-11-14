"""CPU-only retrospective analysis of actual saved Return evaluation branches."""
import os,time,json,hashlib,csv,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np
from torch import nn
S=Path(__file__).resolve().parent;R=S/'runtime'
import sys
sys.path[:0]=[str(R),str(R/'research_notes')]
from research.interaction_fb.schema import privileged,deploy
from diagnostic_fb_resume import tree_digest
torch.set_num_threads(2);torch.set_grad_enabled(False)
start=time.monotonic()
ck=torch.load(S/'runs/fit-01/return/adapter-64.pt',map_location='cpu',weights_only=False)
sd={k[2:]:v for k,v in ck['agent']['nets'].items() if k.startswith('q.')}
q=nn.Sequential(nn.Linear(sd['0.weight'].shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
q.load_state_dict(sd);q.eval().requires_grad_(False)
rows=[];same=[];qrows=[]
for path in sorted((S/'runs').glob('evaluate-case-*/evaluation.json')):
    ev=json.loads(path.read_text());case=ev['case'];initials={}
    for branch in ev['branches']:
        model=branch['model'];mode='sample' if branch['sample'] else 'mean'
        dest=path.parent/(model+'-'+mode)
        tr=torch.load(dest/'trace.pt',map_location='cpu',weights_only=False)
        first=tr['actions'][0];o=first['observation'];a=first['audit']['executed'];z=first['z']
        score=float(q(torch.cat([privileged(o),a,z,torch.zeros(1,32)],-1)))
        au=first['audit'];raw=au['adapter'];sc=ck['agent']['scale']
        correction=(au['qexec']-au['qbase'])
        initial=dict(observation=tree_digest(o),z=tree_digest(z),base=tree_digest(au['base']),
                     initial_qpos=tree_digest(tr['initial']['mjdata_arrays']['qpos']),
                     initial_qvel=tree_digest(tr['initial']['mjdata_arrays']['qvel']))
        initials[(model,mode)]=initial
        discount=1.;ret=0.
        for row in tr['rows']:
            ret+=discount*float(row['aux_rewards']['contact_physical'])
            discount*=.99**(float(row['duration'])/.1)
        yaw=np.rad2deg(np.unwrap([math.atan2(np.asarray(r['pelvis_matrix'])[1,0],np.asarray(r['pelvis_matrix'])[0,0]) for r in tr['physics']]))
        rec=dict(case_index=int(path.parent.name.split('-')[2]),role=case['role'],parent_id=case['parent_id'],
                 time=case['time'],model=model,mode=mode,status=branch['status'],**branch['metrics'],
                 actual_discounted_3s_return=ret,yaw_change_first_tick_to_last_deg=float(yaw[-1]-yaw[0]),
                 first_q_CPU=score,first_raw_return_proxy=float(((raw*sc)**2).mean()/.01),
                 first_raw_joint_equivalent_rms_rad=float(((raw*sc)**2).mean().sqrt()),
                 first_executed_correction_rms_rad=float(correction.square().mean().sqrt()),
                 first_executed_correction_max_rad=float(correction.abs().max()),
                 envelope_trigger_fraction=float(au['triggered'].float().mean()),trace_sha256=branch['trace_sha256'])
        rows.append(rec)
    for mode in ('mean','sample'):
        entries=[(model,value) for (model,typ),value in initials.items() if typ==mode]
        if len(entries)>1:
            matches=all(value==entries[0][1] for _,value in entries)
            same.append(dict(parent_id=case['parent_id'],mode=mode,models=[m for m,_ in entries],
                             exact_first_obs_z_base_qpos_qvel=matches,first=entries[0][1]))
            assert matches,'First-state/action attribution failed'
    qrows.extend([r for r in rows if r['parent_id']==case['parent_id']])
(S/'data/physical-comparison.json').write_text(json.dumps(dict(rows=rows,first_state_identity=same,
    Q_scope='CPU retrospective frozenQ; not the same target as a finite3s realized return',
    missing_cases=[4,5,6,7,0],new_physics=0,new_updates=0),indent=2))
if rows:
    with (S/'data/physical-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
learning={name:json.loads((S/f'runs/fit-01/{name}/learning.json').read_text()) for name in ('control','return')}
summary={name:dict(first=values[0],last=values[-1]) for name,values in learning.items()}
(S/'data/learning-summary.json').write_text(json.dumps(summary,indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for name,values in learning.items():
        x=[r['adapter_step'] for r in values]
        axes[0].plot(x,[r['raw_return'] for r in values],label=name)
        axes[1].plot(x,[r['q'] for r in values],label=name)
    axes[0].set_yscale('log');axes[0].set_ylabel('Raw Return proxy (not physical quality)')
    axes[1].set_ylabel('Frozen Q local score (not realized return)')
    for ax in axes:ax.set_xlabel('Adapter-only update');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(S/'data/learning.png',dpi=150);plt.close(fig)
except ImportError:
    pass
assert not torch.cuda.is_initialized()
(S/'jobs/analysis-cost.json').write_text(json.dumps(dict(wall_s=time.monotonic()-start,
    new_physics=0,new_updates=0,cuda_initialized=False),indent=2))
print(json.dumps(dict(rows=len(rows),first_state_matches=len(same),wall_s=time.monotonic()-start)))
