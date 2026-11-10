"""Issue56 retrospective CPU-only archive analysis. No optimizer step or physics run."""
import argparse, collections, json, math, os, statistics, sys, time
from pathlib import Path
import numpy as np
import torch
def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False))
def med(x): return float(np.median(x)) if len(x) else None
def nearest(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return np.sqrt(np.maximum(0,(a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T).min(1)/a.shape[1])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.monotonic();a.output.mkdir(exist_ok=False);torch.set_num_threads(1)
    run=a.stage/'runs/expert-kl002';rt=a.stage/'runtime'
    sys.path[:0]=[str(rt/'research_notes'),str(rt)]
    base=json.loads((run/'dev-0/summaries.json').read_text());new=json.loads((run/'dev-48/summaries.json').read_text())
    orig={(x['parent_id'],x['child']):x for x in base};pairs=[]
    for row in new:
        old=orig[row['parent_id'],row['child']]
        assert old['source_state_sha256']==row['source_state_sha256']
        phases={}
        for ph in ['startup','sustained','deceleration','holding','released']:
            if ph not in old['quality'] or ph not in row['quality']:continue
            u,v=old['quality'][ph],row['quality'][ph]
            keys=['error_cm','force_rms','force_xyz_rms','error_xyz_rms_cm','error_xyz_signed_cm','cap_fraction','root_delta_cm','root_vx_mean','slip_peak_mps','slip_p95_mps','yaw_deg_final','qpd_max_step_deg','q_actual_max_step_deg','contact_physics']
            phases[ph]={'initial':{k:u.get(k) for k in keys},'batch48':{k:v.get(k) for k in keys}}
            if ph!='released' and u.get('error_cm') is not None and v.get('error_cm') is not None:
                phases[ph]['delta_integrated_squared_error_cm2_s']=(v['error_cm']**2*v['contact_physics']-u['error_cm']**2*u['contact_physics'])*.005
        pairs.append(dict(parent_id=row['parent_id'],child=row['child'],source=row['source'],family=row['family'],condition=row['condition'],initial_all=old['all'],batch48_all=row['all'],delta_error_cm=None if old['all']['error_cm'] is None else row['all']['error_cm']-old['all']['error_cm'],phases=phases))
    write(a.output/'phase-pairs.json',pairs)
    ledger=[json.loads(x) for x in (run/'source-episodes.jsonl').read_text().splitlines()]
    specs={x['spec']['parent_id']:x['spec'] for x in ledger}
    counts=collections.defaultdict(lambda:dict(decisions=0,seconds=0.,parents=set(),completed_episodes=0,positive_advantage=0,negative_advantage=0))
    # Fixed reservoir for descriptive archived-training-state overlap, never dev-state equivalence.
    rng=np.random.default_rng(5648);reservoir={};seen=collections.Counter();last_batches={}
    from dual_track_returns import advantages
    for it in range(1,49):
        stats=json.loads((run/f'rollout-{it}-stats.json').read_text())
        with np.load(run/f'rollout-{it}.npz',allow_pickle=False) as f: data={k:f[k] for k in f.files}
        ts={k:torch.from_numpy(v) for k,v in data.items()}
        adv,ret=advantages(ts['reward'],ts['value'],ts['next_value'],ts['discount'],ts['terminated'],ts['truncated'],.95)
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
        if it>=45:last_batches[it]=(data,stats,adv)
        for j,st in enumerate(stats):
            spec=specs[st['parent_id']]
            ph='startup' if st['phase']=='moving' and st['decision_time_s']<spec['onset']+1-1e-7 else st['phase']
            key='/'.join([st['source'],st['family'],ph]);c=counts[key]
            c['decisions']+=1;c['seconds']+=st['actual_duration_s'];c['parents'].add(st['parent_id'])
            c['completed_episodes']+=bool(st['terminated'] or st['truncated'])
            c['positive_advantage']+=adv[j].item()>0;c['negative_advantage']+=adv[j].item()<0
            if st['family'] in ('near','long') and ph in ('startup','moving','contact_hold'):
                k=(st['source'],st['family'],ph);seen[k]+=1
                row=(data['obs'][j].copy(),st['parent_id'])
                pool=reservoir.setdefault(k,[])
                if len(pool)<160:pool.append(row)
                else:
                    ix=int(rng.integers(seen[k]))
                    if ix<160:pool[ix]=row
    coverage={k:dict(v,parents=len(v['parents'])) for k,v in counts.items()}
    write(a.output/'coverage-to48.json',coverage)
    bc=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/interactive-contact-pilot-20261108/runs/bc-fit-01/old_data_only-500.pt')
    os.environ['P4000_CANONICAL_ROW_ACTOR']='1'
    from dual_track_runtime import PrivilegedBranch
    net=PrivilegedBranch(torch.load(bc,map_location='cpu',weights_only=True))
    ck=torch.load(run/'candidate-48.pt',map_location='cpu',weights_only=False)
    net.load_state_dict(ck['model'],strict=True);net.requires_grad_(False);net.actor.requires_grad_(True)
    center=net.base.center.numpy();scale=net.base.scale.numpy()
    def normalized(rows):
        x=np.stack([x[0] for x in rows]);x[:,:1639]=np.clip((x[:,:1639]-center)/scale,-10,10);return x
    overlap=[]
    for source in ['standing','goal_arm5','motion_walk_task']:
        for phase in ['startup','moving','contact_hold']:
            aa=reservoir.get((source,'near',phase),[]);bb=reservoir.get((source,'long',phase),[])
            if not aa or not bb:continue
            x,y=normalized(aa),normalized(bb);pa=np.array([z[1] for z in aa]);pb=np.array([z[1] for z in bb])
            for dims,label in [(2727,'full_actual_actor_input'),(1639,'restricted_history_subspace')]:
                xx,yy=x[:,:dims],y[:,:dims]
                cross=nearest(xx,yy)
                dist=np.maximum(0,(xx*xx).sum(1)[:,None]+(xx*xx).sum(1)[None,:]-2*xx@xx.T)
                dist[pa[:,None]==pa[None,:]]=np.inf
                within=np.sqrt(dist.min(1)/dims);valid=within[np.isfinite(within)]
                rad=float(np.quantile(valid,.95)) if len(valid) else None
                overlap.append(dict(source=source,phase=phase,space=label,near_frames=len(aa),long_frames=len(bb),near_parents=len(set(pa)),long_parents=len(set(pb)),cross_RMS_distance_median=med(cross),cross_RMS_distance_p95=float(np.quantile(cross,.95)),within_near_different_parent_NN_p95=rad,cross_inside_within_radius_fraction=float(np.mean(cross<=rad)) if rad is not None else None,interpretation='Archived on-policy training states only; not saved deterministic dev trajectories; descriptor not independence or causal proof'))
    write(a.output/'training-state-overlap.json',overlap)
    params=list(net.actor.parameters());gradient=[];selfcheck={}
    # Same post-batch48 theta, original raw actions/logp and each saved batch-global GAE normalization.
    pooled={}
    for it,(data,stats,adv) in last_batches.items():
        raw=torch.from_numpy(data['action']);old=torch.from_numpy(data['logp']);obs=torch.from_numpy(data['obs'])
        src=np.array([s['source'] for s in stats]);fam=np.array([s['family'] for s in stats])
        groups={'near':fam=='near','long':fam=='long','near_arm':(fam=='near')&(src=='goal_arm5'),'long_standing':(fam=='long')&(src=='standing')}
        gv={};gr={}
        for name,mask in groups.items():
            ix=torch.from_numpy(np.flatnonzero(mask))
            if not len(ix):continue
            dist=net.distribution(obs[ix]);logp=dist.log_prob(raw[ix]).sum(-1);ratio=(logp-old[ix]).exp()
            aa=adv[ix];loss=-torch.minimum(ratio*aa,ratio.clamp(.85,1.15)*aa).mean()
            g=torch.autograd.grad(loss,params,allow_unused=True)
            v=torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for p,x in zip(params,g)]).detach()
            gv[name]=v
            pooled[name]=pooled.get(name,torch.zeros_like(v))+v*len(ix)
            gr[name]=dict(samples=len(ix),norm=float(v.norm()),clip_fraction=float(((ratio-1).abs()>.15).float().mean()),advantage_mean=float(aa.mean()),loss=float(loss.detach()),gradient_descent_direction=True)
        cos={}
        for u,v in [('near','long'),('near_arm','long_standing')]:
            if u in gv and v in gv:
                den=gv[u].norm()*gv[v].norm()
                cos[u+'__'+v]=float(torch.dot(gv[u],gv[v])/den) if den>0 else None
        gradient.append(dict(saved_rollout=it,parameter_checkpoint=48,groups=gr,cosines=cos))
    pooled_cos={}
    for u,v in [('near','long'),('near_arm','long_standing')]:
        if u in pooled and v in pooled:
            den=pooled[u].norm()*pooled[v].norm()
            pooled_cos[u+'__'+v]=float(torch.dot(pooled[u],pooled[v])/den) if den>0 else None
    gradient.append(dict(pooled_saved_rollouts=[45,46,47,48],parameter_checkpoint=48,sample_weighted_gradients=True,cosines=pooled_cos,not_the_actual_historical_Adam_update=True))
    write(a.output/'fixed-point-gradients.json',gradient)
    out=dict(runtime='147b22b684b0c03a613559fa6d5e5881d380ab7f',kind='retrospective CPU-only archive diagnostics',physics_runs=0,optimizer_steps=0,RL_decisions=0,analysis_wall_s=time.monotonic()-start,
        phase_definition='startup fixed first1s after onset, sustained until partner stop; not exact first footstep',
        long_pair_changes=[dict(source=x['source'],parent=x['parent_id'],delta_error_cm=x['delta_error_cm']) for x in pairs if x['family']=='long'],
        near_pair_changes=[dict(source=x['source'],parent=x['parent_id'],delta_error_cm=x['delta_error_cm']) for x in pairs if x['family']=='near'],
        aggregate_notes='family medians are not median of paired deltas; do not infer a global local-vs-long conflict',
        missing='All-frame dev0/dev48 actor inputs and q/qdot were not persisted; exact overlap of the particular improved/degraded deterministic dev trajectories and first actual support event cannot be determined from saved summaries. Training-state overlap/gradients are proxies with stated scope, not substitute dev facts.')
    write(a.output/'summary.json',out)
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
