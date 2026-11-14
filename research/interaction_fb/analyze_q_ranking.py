"""D11 read-only diagnosis of archived R2 Q ranking; CPU, no update/physics."""
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]

def main():
    started=time.time()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    from research.interaction_fb.approved_journal import load_state
    from research.interaction_fb.journal import read
    from research.interaction_fb.schema import privileged
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    q=args.stage;run=q/'runs/paired-Q-ranking-12'
    result=json.loads((run/'qualification.json').read_text())
    models={a:load_state(q/'runs'/name/'checkpoints/round-14') for a,name in
            [('E','E-warmup3500-10'),('I','I-warmup3500-11')]}
    def mlp(x,weights,prefix,head=False):
        for i in (0,2,4):
            x=F.linear(x,weights[f'{prefix}.{i}.weight'],weights[f'{prefix}.{i}.bias'])
            if i!=4:x=torch.tanh(x)
        return -F.softplus(x) if head else x
    def zi(arm):
        a=models[arm]['agent'];key='bi' if arm=='I' else 'goal_encoder'
        x=mlp(a['prompt'],a['nets'],key)
        return math.sqrt(32)*F.normalize(x,dim=-1)
    rows=[];error=0.;q_error=[];header_checks=[];support=[]
    for state in result['results']:
        dest=run/f"round-{state['round']}"
        mp=json.loads((q/'runs/E-warmup3500-10/checkpoints'/f"round-{state['round']}"/'manifest.json').read_text())
        # Initial physical bundle includes the exact pre-intervention robot observations
        # and high-level interpolation start/target. No simulation is used here.
        data=(dest/'full-start.ptz').read_bytes()
        initial=read(dest/'full-start.ptz',dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        branch_records=[]
        for branch in state['branches']:
            trace=read(dest/(branch['id']+'.ptz'),branch['archive'])
            row=trace['rows'][0];o=row['observation'];a=row['action'];z=row['z']
            header={k:torch.equal(o[k],v) for k,v in initial['physical']['env']['obs'].items()}
            header.update(previous_z=torch.equal(o['_previous_z'],initial['proxy']['start']),
                          target_z=torch.equal(o['_target_z'],initial['proxy']['target']),
                          slot_zero=int(o['_slot'].item())==0)
            header_checks.append(dict(round=state['round'],candidate=branch['id'],fields=header))
            assert all(header.values()),('First executed input differs from scored physical input',header_checks[-1])
            elapsed=0.;actual=0.
            for rr in trace['rows']:
                actual+=.99**(elapsed/.1)*float(rr['aux_rewards']['contact_physical'])
                elapsed+=float(rr['duration'])
            error=max(error,abs(actual-branch['discounted_return']))
            assert abs(actual-branch['discounted_return'])<1e-12
            for arm in ('E','I'):
                ii=zi(arm)
                x=torch.cat([privileged(o),a,z,ii],-1)
                scores=[float(mlp(x,models[arm]['agent']['nets'],k+'.0',True)) for k in ('q1','q2')]
                delta=abs(min(scores)-branch['scores'][arm]['q_min'])
                q_error.append(delta)
            candidates=[rr for rr in models['E']['rows']
                        if bytes(rr['provenance']['parent_sha256'][0].tolist()).hex()==state['parent_id']]
            dt=min(abs(float(rr['provenance']['time'])-state['time']) for rr in candidates)
            nearby=[rr for rr in candidates if abs(abs(float(rr['provenance']['time'])-state['time'])-dt)<1e-7]
            dist=min(float((a-rr['action']).norm()) for rr in nearby)
            support.append(dict(round=state['round'],candidate=branch['id'],
                closest_actual_replay_time_distance_s=dt,nearest_actual_action_l2=dist,
                proposal_linf_from_mean=branch['first_action_linf_from_mean'],
                proposal_noise_l2_sigma=branch['first_action_noise_l2_sigma']))
            branch_records.append(branch)
        per={}
        for arm in ('E','I'):
            pairs=[x for x in result['verdict'][arm]['comparisons'] if x['round']==state['round'] and x['eligible']]
            bt=0
            for aa,bb in itertools.combinations(branch_records,2):
                gd=aa['discounted_return']-bb['discounted_return']
                if abs(gd)<.0001:continue
                boot=aa['scores'][arm]['bootstrapped_window_return']-bb['scores'][arm]['bootstrapped_window_return']
                bt+=gd*boot>0
            per[arm]=dict(correct=sum(x['correct'] for x in pairs),eligible=len(pairs),
                accuracy=sum(x['correct'] for x in pairs)/len(pairs) if pairs else None,
                mean_q=branch_records[0]['scores'][arm]['q_min'],
                q_mean_vs_realized_prefix_gap=branch_records[0]['scores'][arm]['q_min']-branch_records[0]['discounted_return'],
                bootstrapped_vs_finite_same_direction_count=int(bt))
        mean=branch_records[0]
        rows.append(dict(round=state['round'],time=state['time'],family=state['family'],
            parent_id=state['parent_id'],arms=per,mean_finite_return=mean['discounted_return'],
            mean_force_rms_n=mean['active_force_rms_n'],mean_error_rms_m=mean['active_error_rms_m'],
            max_slip=max(v['quality']['slip_peak'] for v in branch_records),
            max_joint_excess=max(v['quality']['max_joint_excess_rad'] for v in branch_records),
            unsupported_ticks=sum(v['quality']['unsupported_ticks'] for v in branch_records)))
    warmup={}
    for arm,name in [('E','E-warmup3500-10'),('I','I-warmup3500-11')]:
        last=json.loads((q/'runs'/name/'blocks.json').read_text())[-1]
        updates=[v for v in last['updates'] if v['kind']=='q']
        warmup[arm]=dict(last_Q_loss=updates[-1]['losses'],last_target_mean=updates[-1]['target_mean'],
            last_Q_mean=updates[-1]['q_mean'],Q_groups=last['q_steps'],actor_groups=last['actor_steps'])
    answer=dict(status='D11_read_only_complete',physics_steps=0,optimizer_updates=0,cuda_initialized=torch.cuda.is_initialized(),
        runtime_Q_gate=result['lock_sha256'],states=rows,support=support,actual_input_header_checks=header_checks,
        return_reconstruction_max_error=error,Q_cpu_reconstruction_max_abs_error=max(q_error),
        Q_cpu_reconstruction_error_scope='CPU versus archived NVIDIA FP32 forward; not bit-exact claim or changed probability tolerance',
        warmup=warmup,wall_s=time.time()-started)
    (args.output/'diagnostic.json').write_text(json.dumps(answer,indent=2))
    print(json.dumps({k:v for k,v in answer.items() if k not in ('support','actual_input_header_checks')},indent=2))
if __name__=='__main__':main()
