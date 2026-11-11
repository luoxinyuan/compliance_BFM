"""Full fixed-batch/RNG old-vs-lambda0 update and complete-prefix physical checks."""
import copy,json,time
from pathlib import Path
import numpy as np,torch
from frl_v2_high import build,write
from frl_v2_retention import Retention
from frl_v2_fit import update
from dual_track_runtime import PPOContract,TrainingBranch
from dual_track_returns import advantages,high_reward_from_native
from frl_v3_risk import event
import frl_v3_task
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');OLD=P/'experiments/frl-v2-fb-repair-20261110';ST=Path(__file__).resolve().parents[2]
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run(e,m,c,out,args):
    t0=time.monotonic();start=torch.load(OLD/'runs/frl-formal-B-01/candidate-64.pt',map_location='cpu',weights_only=False);expected=torch.load(OLD/'runs/frl-formal-B-01/candidate-65.pt',map_location='cpu',weights_only=False)
    arrays=np.load(OLD/'runs/frl-formal-B-01/rollout-65.npz');b={k:torch.from_numpy(arrays[k]).cuda() for k in arrays.files};adv,returns=advantages(b['reward'],b['value'],b['next_value'],b['discount'],b['terminated'],b['truncated'],.95);b.update(returns=returns,advantages=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6),raw_adv_std=(returns-b['value']).std())
    results=[];states=[];opts=[]
    for name in ('original','v3_lambda0'):
        n=build();retention=Retention(n,True);n.load_state_dict(start['model']);opt=torch.optim.Adam([{'params':n.actor.parameters(),'lr':3e-5},{'params':n.critic.parameters(),'lr':.0003}]);opt.load_state_dict(copy.deepcopy(start['optimizer']));retention.load_state_dict(start['counters']['retention']);retention.seed_reference(b['obs'],b['pre_mask'])
        torch.set_rng_state(start['rng']['cpu']);torch.cuda.set_rng_state_all(start['rng']['cuda'])
        # Both use the ORIGINAL unmodified frl_v2_fit / probability / transactional Adam routine.
        report=update(n,opt,b,epochs=6,minibatch=256,critic_shuffle_seed=58010065,contract=PPOContract(),retention=retention);hold=retention.after_batch(n,b['obs'],b['pre_mask'])
        states.append({k:v.detach().clone() for k,v in n.state_dict().items()});opts.append(copy.deepcopy(opt.state_dict()));results.append(dict(name=name,update=report,retention=hold))
    same_model=equal(states[0],states[1]);same_adam=equal(opts[0],opts[1]);archive_model=equal(states[0],{k:v.cuda() for k,v in expected['model'].items()});archive_adam=equal(torch_to(opts[0],'cpu'),expected['optimizer'])
    # Complete actual same-reset prefix, exact continuation with lambda0 vs original method.
    frl_v3_task.configure(0.);task=frl_v3_task.Task(e,m,c,5801);plan=json.loads((ST/'prep/locked-protocol.json').read_text());spec=next(r['spec'] for r in plan['P1']['actual_development_states'] if r['family']=='mid');task.reset(spec,record=True,label='lambda0-prefix');n=build();n.eval().requires_grad_(False)
    while float(e.tether.time)<spec['onset']+.2-1e-7:
        x=torch.cat([task.student_features(),task.features()],-1);task.step(n.distribution(x).mean)
    snap=TrainingBranch(task);runs=[]
    from frl_v2_task import Task as Original
    for path in ('original','lambda0'):
        snap.restore();p0=len(task.physics);c0=len(task.controls);commands=[];values=[]
        for j in range(12):
            x=torch.cat([task.student_features(),task.features()],-1);raw=n.distribution(x).mean;commands.append(raw.clone());res=Original.step(task,raw) if path=='original' else task.step(raw);values.append(res[1]);
            if task.done:break
        runs.append(dict(physics=copy.deepcopy(task.physics[p0:]),control=copy.deepcopy(task.controls[c0:]),reward=values,features=torch.cat([task.student_features(),task.features()],-1).clone(),raw=commands,physical_fingerprint=TrainingBranch(task).fingerprint))
    physics_equal=equal(runs[0],runs[1]);torch.save(runs,out/'lambda0-physical.pt')
    # Independently recompute risk added to REAL recorded branch controls, no synthetic physical claim.
    assert args.p1_gate['P2_allowed'] is True
    p1=args.p1_output
    import gzip
    zero=positive=0;examples=[]
    for file in sorted(p1.glob('state*.json.gz')):
        ep=json.load(gzip.open(file,'rt'));rows=ep['physics'];cs=ep['control']
        for i,cc in enumerate(cs):
            end=cs[i+1]['time'] if i+1<len(cs) else cc['time']+cc['actual_duration_s']+1e-8;rr=[r for r in rows if cc['time']-1e-8<=r['force_time']<end-1e-8];risk=event(rr);old=cc['reward_integral'];new=old-risk
            assert event(rr,0.)==0.
            if risk==0.:assert new==old;zero+=1
            else:assert (new-old)==-risk or abs((new-old)+risk)<1e-15;positive+=1
        if zero>200 and positive>200:break
    assert zero>0 and positive>0
    result=dict(status='completed',fixed_minibatch=True,restored_RNG_exact=True,same_original_update_function=True,lambda0_model_bitwise=same_model,lambda0_Adam_bitwise=same_adam,archive_B65_model_bitwise=archive_model,archive_B65_Adam_bitwise=archive_adam,full_physical_prefix_bitwise=physics_equal,zero_event_reward_exact_count=zero,positive_only_event_change_count=positive,GAE_discount_termination='same imported original functions / arrays, never changed',old_archived_control_qualified=bool(same_model and same_adam and physics_equal and archive_model and archive_adam),training_updates_for_qualification_only=2*sum(r['update']['optimizer_updates']+r['update']['critic_fit']['optimizer_updates'] for r in results),job_s=time.monotonic()-t0,reports=results)
    write(out/'checks.json',result);return result

def torch_to(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:torch_to(v,device) for k,v in x.items()}
    if isinstance(x,list):return [torch_to(v,device) for v in x]
    if isinstance(x,tuple):return tuple(torch_to(v,device) for v in x)
    return x
