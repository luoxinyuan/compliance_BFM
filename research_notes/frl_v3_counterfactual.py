"""Complete-state frozen first-command interventions, no optimization/learning."""
import copy,gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch,mujoco
from frl_v2_high import build,sha,write
from frl_v2_task import Task,configure
from dual_track_runtime import TrainingBranch,restore_training_checkpoint
from dual_track_returns import advantages
from diagnostic_frozen import portable
from frl_v3_risk import score,metrics
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'experiments/frl-v2-fb-repair-20261110';ST=Path(__file__).resolve().parents[2]

def js(p):
    with gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open() as f:return json.load(f)
def tensorsame(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensorsame(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tensorsame(x,y) for x,y in zip(a,b))
    return a==b

def install_contact_details(e):
    original=e._original_step
    def sampled():
        original();d=e.sim.data;m=e.sim.model;details=[]
        for i,con in enumerate(d.contact[:d.ncon]):
            bodies=[int(m.geom_bodyid[con.geom1]),int(m.geom_bodyid[con.geom2])]
            if 0 not in bodies:continue
            body=max(bodies);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body) or ''
            if 'ankle' not in name and 'foot' not in name:continue
            wrench=np.zeros(6);mujoco.mj_contactForce(m,d,i,wrench)
            point=con.pos.copy();details.append(dict(body=name,normal_N=abs(float(wrench[0])),world_point=point.tolist(),body_local_point=(d.xmat[body].reshape(3,3).T@(point-d.xpos[body])).tolist(),geom1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom1)),geom2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(con.geom2))))
        assert len(details)==len(e.contact_sample['stage_foot_contacts'])
        for a,b in zip(details,e.contact_sample['stage_foot_contacts']):assert a['body']==b['body'] and a['normal_N']==b['normal_N']
        e.contact_sample['stage_foot_contact_details']=details
    e._original_step=sampled


def run(e,m,c,out,args):
    start=time.monotonic();plan=js(ST/'prep/locked-protocol.json');configure();nets={};identities={}
    for key,k in [('BC500',0),('B64',64),('B128',128)]:
        p=OLD/f'runs/frl-formal-B-01/candidate-{k}.pt';ck=torch.load(p,map_location='cpu',weights_only=False);n=build();n.load_state_dict(ck['model'],strict=True);n.eval().requires_grad_(False);nets[key]=n;identities[key]=dict(checkpoint=str(p),sha256=sha(p))
    # B65 only frozen same-state post-update logprob query, never a controller.
    ck65=torch.load(OLD/'runs/frl-formal-B-01/candidate-65.pt',map_location='cpu',weights_only=False);after65=build();after65.load_state_dict(ck65['model']);after65.eval().requires_grad_(False)
    z=np.load(OLD/'runs/frl-formal-B-01/rollout-65.npz');batch={k:torch.from_numpy(z[k]).cuda() for k in z.files}
    gae,ret=advantages(batch['reward'],batch['value'],batch['next_value'],batch['discount'],batch['terminated'],batch['truncated'],.95);adv=(gae-gae.mean())/gae.std(unbiased=False)
    logs=js(OLD/'runs/frl-formal-B-01/rollout-65-stats.json');t=Task(e,m,c,5801);install_contact_details(e);states=[];rawpool=[dict(index=r['index'],x=batch['obs'][r['index']].clone(),raw=batch['action'][r['index']].clone(),family=r['family']) for r in plan['P1']['actual_training_states']]
    def capture(meta,continuation_rng=None):
        x=torch.cat([t.student_features(),t.features()],-1).detach().clone();s=TrainingBranch(t,() if continuation_rng is None else (continuation_rng,));idx=len(states)
        # Full actual portable state and replayable prefix persist separately from branch predictions.
        torch.save(dict(meta=meta,input=x.cpu(),state=portable(t),spec=copy.deepcopy(t.spec),actual_raw_prefix=[d['raw'] for d in t.decision_rows],previous_z=t.last_command.cpu(),fingerprint=s.fingerprint),out/f'state-{idx:02d}.pt')
        states.append(dict(meta=dict(meta,state_id=idx),snapshot=s,x=x,rng=continuation_rng))
        print('STATE_SAVED',idx,meta,flush=True)
    # Six exposed independent dev parents, four real phase states each.
    for parent in dict.fromkeys(r['parent_id'] for r in plan['P1']['actual_development_states']):
        wanted=[r for r in plan['P1']['actual_development_states'] if r['parent_id']==parent];first=wanted[0];owner=first['policy'];spec=first['spec'];torch.manual_seed(56017);np.random.seed(56017);t.reset(spec,record=True,label=parent[:12]+'-source-'+owner)
        saved=js(OLD/f'runs/frl-formal-B-01/dev-{dict(BC500=0,B64=64,B128=128)[owner]}/'+(parent[:12]+'-main.json.gz'))
        j=0
        while not t.done:
            x=torch.cat([t.student_features(),t.features()],-1)
            for r in wanted:
                if r['decision_index']==j:
                    assert abs(float(e.tether.time)-r['time'])<1e-7
                    assert torch.equal(x[:,1639:].cpu(),torch.tensor(saved['decisions'][j]['obs'])[None]),'archived privileged prefix differs'
                    capture(dict(r,continuation='mean',frozen_owner=owner))
            raw=nets[owner].distribution(x).mean
            assert torch.equal(raw.cpu(),torch.tensor(saved['decisions'][j]['raw'])[None]),'archived deterministic mean differs'
            t.step(raw);j+=1
            if all(r['decision_index']<j for r in wanted):break
    # Reconstruct exact historicalB64 sampled prefix at start of rollout65.
    generator=torch.Generator(device='cuda');before=OLD/'runs/frl-formal-B-01/candidate-64.pt'
    restored=restore_training_checkpoint(before,t,nets['B64'],None,[generator]);episode=restored['episode'];wanted={r['index']:r for r in plan['P1']['actual_training_states']}
    for j in range(max(wanted)+1):
        if t.done:episode+=1;t.reset(record=True,label='train'+str(episode))
        x=torch.cat([t.student_features(),t.features()],-1);assert torch.equal(x[0],batch['obs'][j]),('archived onpolicy state differs',j,float((x[0]-batch['obs'][j]).abs().max()))
        dist=nets['B64'].distribution(x);raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device='cuda',generator=generator)
        assert torch.equal(raw[0],batch['action'][j]),'sampling RNG or actual action mismatch'
        if j in wanted:
            # The historical first draw has been consumed; identical future noise starts after it.
            gg=torch.Generator(device='cuda');gg.set_state(generator.get_state());rr=wanted[j]
            oldlp=float(dist.log_prob(raw).sum());newlp=float(after65.distribution(x).log_prob(raw).sum())
            assert abs(oldlp-float(batch['logp'][j]))<3e-4
            capture(dict(rr,source='actual_training_rollout65',phase=logs[j]['phase'],frozen_owner='B64',continuation='Gaussian shared future noise',historical_raw=raw[0].cpu().tolist(),historical_GAE=float(gae[j]),historical_PPO_adv=float(adv[j]),historical_oldlogp=float(batch['logp'][j]),recomputed_oldlogp=oldlp,after_B65_logp=newlp,post_update_logprob_delta=newlp-oldlp),gg)
        t.step(raw)
    assert len(states)==plan['P1']['states']
    outcomes=[];allsemantics=[];prefix_checks=[]
    for state in states:
        meta=state['meta'];owner=meta['frozen_owner'];ss=state['snapshot'];ss.restore();x=state['x'];at=float(e.tether.time);dist=nets[owner].distribution(x)
        gg=torch.Generator(device='cuda').manual_seed(590100+meta['state_id']);choices=[dict(kind='original_mean',raw=dist.mean,source='same-state '+owner)]
        if 'historical_raw' in meta:choices.append(dict(kind='archived_sample',raw=torch.tensor(meta['historical_raw'],device='cuda')[None],source='exact same state rollout65'))
        else:
            pool=[r for r in rawpool if r['family']==meta['family']] or rawpool;nearest=min(pool,key=lambda r:float((r['x']-x[0]).square().mean()));choices.append(dict(kind='archived_transferred',raw=nearest['raw'][None],source='rollout65 index'+str(nearest['index'])+';not current-state on-policy'))
        for i in range(4):choices.append(dict(kind='Gaussian_'+str(i),raw=dist.mean+dist.stddev*torch.randn(dist.mean.shape,device='cuda',generator=gg),source='same state std.008 jointGaussian'))
        for key in nets:
            if key!=owner:choices.append(dict(kind='alternative_mean_'+key,raw=nets[key].distribution(x).mean,source='same-state frozen '+key))
        good=[r for r in rawpool if float(adv[r['index']])>0] or rawpool;nearest=min(good,key=lambda r:float((r['x']-x[0]).square().mean()));choices.append(dict(kind='observed_better_transferred',raw=nearest['raw'][None],source='observed clean positiveadv sample '+str(nearest['index'])+'; transported proposal not causal good label'))
        seen={};unique=[]
        for ch in choices:
            key=ch['raw'].detach().cpu().numpy().tobytes()
            if key in seen:seen[key]['duplicate_sources'].append(ch['kind']);continue
            ch['duplicate_sources']=[];seen[key]=ch;unique.append(ch)
        baseline=None
        for ch in unique:
            ss.restore();assert ss.digest()==ss.fingerprint
            t.label=f"state{meta['state_id']:02d}-"+ch['kind'];nphys=len(t.physics);nctrl=len(t.controls);source_spec=copy.deepcopy(t.spec);original_prefix_physics=t.physics[:];original_prefix_controls=t.controls[:]
            # Store independent full pre-intervention critical state digest; no pose-only reset.
            before_hash=TrainingBranch(t).fingerprint;assert before_hash==ss.fingerprint
            t.step(ch['raw']);stop_time=at+.1+plan['P1']['continuation_s']
            while not t.done and float(e.tether.time)<stop_time-1e-8:
                nx=torch.cat([t.student_features(),t.features()],-1);dd=nets[owner].distribution(nx)
                command=dd.mean if state['rng'] is None else dd.mean+dd.stddev*torch.randn(dd.mean.shape,device='cuda',generator=state['rng'])
                t.step(command)
            assert t.physics[:nphys]==original_prefix_physics and t.controls[:nctrl]==original_prefix_controls
            physics=t.physics[nphys:];controls=t.controls[nctrl:];res=metrics(physics,controls,source_spec);sc=score(physics,controls,at);res['low_slip_task_success']=res['low_slip_task_success'] and t.boundary is None
            row=dict(state_id=meta['state_id'],parent_id=meta['parent_id'],phase=meta['phase'],family=meta['family'],candidate=ch['kind'],candidate_source=ch['source'],duplicate_sources=ch['duplicate_sources'],state_owner=owner,continuation=meta['continuation'],time=at,prefix_fingerprint=ss.fingerprint,prefix_exact=True,boundary=t.boundary,metrics=res,scores=sc)
            if baseline is None:baseline=row
            row['delta_return_vs_original_mean']=sc['old_discounted_return']-baseline['scores']['old_discounted_return'];row['delta_v3_return_vs_original_mean']=sc['v3_discounted_return']-baseline['scores']['v3_discounted_return']
            # Complete real physical counterfactual, NOT a learned/on-policy training sample.
            with gzip.open(out/(t.label+'.json.gz'),'wt') as f:json.dump(dict(meta=meta,proposal_raw=ch['raw'][0].cpu().tolist(),physical_counterfactual=True,physics=physics,control=controls,summary=row),f)
            torch.save(dict(input=x.cpu(),raw=ch['raw'].cpu(),logprob=float(dist.log_prob(ch['raw']).sum()),previous_z=ss.task['last_command'].cpu(),source=ch['source']),out/(t.label+'-command.pt'))
            outcomes.append(row);allsemantics.extend(dict(state_id=meta['state_id'],candidate=ch['kind'],**ev) for ev in sc['risk_events']);write(out/'outcomes.json',outcomes)
            print('COUNTERFACTUAL',meta['state_id'],ch['kind'],res['low_slip_task_success'],res['error_cm'],res['loaded_slip_peak'],flush=True)
        prefix_checks.append(dict(state_id=meta['state_id'],restored_prefix_sha256=ss.fingerprint,candidates=len(unique),shared_partner=True,shared_actual_history=True,shared_future_noise=True))
    # Existence gate is nontrivial dynamic/stop feedback, not easy contact onset or all-domain qualification.
    successes=[r for r in outcomes if r['metrics']['low_slip_task_success']];nontrivial=[r for r in successes if r['phase']!='entry' and r['metrics']['partner_dx_m']>.02]
    state_results=[];agreements=[]
    for state in states:
        meta=state['meta'];rr=[r for r in outcomes if r['state_id']==meta['state_id']];base=rr[0];sample=next((r for r in rr if r['candidate']=='archived_sample'),None)
        state_results.append(dict(state_id=meta['state_id'],parent_id=meta['parent_id'],phase=meta['phase'],owner=meta['frozen_owner'],clean_success_candidates=sum(r['metrics']['low_slip_task_success'] for r in rr),candidates=len(rr)))
        if sample:
            delta=sample['delta_return_vs_original_mean'];agreements.append(dict(state_id=meta['state_id'],parent_id=meta['parent_id'],historical_GAE=meta['historical_GAE'],PPO_adv=meta['historical_PPO_adv'],logprob_delta=meta['post_update_logprob_delta'],counterfactual_old_return_delta=delta,counterfactual_v3_return_delta=sample['delta_v3_return_vs_original_mean'],sample_clean_success=sample['metrics']['low_slip_task_success'],adv_sign_agrees_with_counterfactual=(meta['historical_PPO_adv']>0)==(delta>0),probability_sign_agrees_with_counterfactual=(meta['post_update_logprob_delta']>0)==(delta>0),scope='finite2.1s CRN feedback return, not whole-episode ground truth advantage'))
    write(out/'prefix-checks.json',prefix_checks);write(out/'load-semantics.json',allsemantics);write(out/'historical-credit.json',agreements);write(out/'state-results.json',state_results)
    gate=dict(P2_allowed=bool(nontrivial),existence_answer='exists within measured subset' if nontrivial else 'not established in current bounded candidates',clean_success_states=len(set(r['state_id'] for r in successes)),nontrivial_success_states=len(set(r['state_id'] for r in nontrivial)),nontrivial_parents=len(set(r['parent_id'] for r in nontrivial)),states=len(states),parents=plan['P1']['parents'],candidates=len(outcomes),new_training_updates=0,job_s=time.monotonic()-start,continuation='every state producer/mode explicitly recorded; other models only firstcommand',normal_threshold_unchanged_N=1e-6)
    write(out/'gate.json',gate);return dict(status='completed',**gate)
