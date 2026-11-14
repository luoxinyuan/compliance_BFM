"""Approved, adapter-only offline isolation; no Agent.update or physics rollout."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];S=ROOT.parent;P=ROOT.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
import torch
from torch.nn import functional as F
from research.interaction_fb.agent import Agent
from research.interaction_fb.journal import load_state
from research.interaction_fb.schema import deploy,robot,to_device
from research.interaction_fb.envelope import Envelope
from diagnostic_fb_resume import tree_digest
from dual_track_fb_online import stack
from dual_track_task import global_rng


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))


def frozen_digest(agent):
    st=agent.state()
    return tree_digest(dict(nets={k:v for k,v in st['nets'].items() if not k.startswith('adapter.')},
        targets=st['targets'],optimizers={k:v for k,v in st['optimizers'].items() if k!='adapter'},
        normalizers=st['normalizers'],prompt=st['prompt'],goal=st['goal'],scale=st['scale'],
        updates=st['updates'],z_buffer=st['z_buffer'],audit=st['audit'],exploration=st['exploration']))


def losses(agent,batch,native,refs):
    o=to_device(stack([r['observation'] for r in batch]));eo=to_device(stack([r['observation'] for r in native]))
    ez=torch.cat([r['z'] for r in native]).cuda()
    a,z,zi=agent.action(o)
    # The fixed Q sees exactly the original V1 executable action.
    assert torch.equal(a,agent.last_action_audit['executed'])
    val=agent.nets['q'](agent.inp(o,a,z,zi)).mean()
    ea,_,_=agent.action(eo)
    with torch.no_grad():old=agent.base.act(robot(eo),ez,True)
    cpr=(((ea-old)*agent.scale)**2).mean()/.01
    adv=F.softplus(-agent.nets['disc'](torch.cat([o['state'],z],-1))).mean()
    actor=-val+.1*cpr+.01*adv
    ro=to_device(stack([r['observation'] for r in refs]));_,rz,ri=agent.action(ro)
    delta=agent.nets['adapter'](torch.cat([deploy(ro),rz.detach(),ri.detach()],-1))
    ret=((delta*agent.scale)**2).mean()/.01
    return actor,ret,dict(q=float(val.detach()),cpr=float(cpr.detach()),disc=float(adv.detach()),
                         raw_return=float(ret.detach()),raw_correction_rms=float(delta.detach().square().mean().sqrt()))


def step(agent,batch,native,refs,weight,baseline=False):
    frozen=frozen_digest(agent);rng=tree_digest(global_rng());op=agent.opts['adapter']
    op.zero_grad(set_to_none=True)
    with torch.enable_grad():
        actor,ret,detail=losses(agent,batch,native,refs)
        # Lambda0 is the original adapter-only actor update; compute and log the
        # same reference batch but add no dummy zero branch to its backward.
        loss=actor if baseline or weight==0 else actor+weight*ret
        grads_a=torch.autograd.grad(actor,tuple(agent.nets['adapter'].parameters()),retain_graph=True)
        grads_r=torch.autograd.grad(ret,tuple(agent.nets['adapter'].parameters()),retain_graph=True)
        na=sum(float(g.square().sum()) for g in grads_a)**.5
        nr=sum(float(g.square().sum()) for g in grads_r)**.5
        loss.backward()
    gg=[p.grad for p in agent.nets['adapter'].parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gg)
    ng=sum(float(g.square().sum()) for g in gg)**.5;assert ng>0
    op.step()
    assert frozen_digest(agent)==frozen,'Frozen module/Adam/targets/normalizers changed'
    assert tree_digest(global_rng())==rng,'Unexpected stochastic actor/loss draw'
    detail.update(actor=float(actor.detach()),total=float(loss.detach()),actor_gradient_norm=na,
                  return_gradient_norm=nr,weighted_return_gradient_norm=weight*nr,total_gradient_norm=ng,
                  frozen_digest=frozen)
    return detail


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gate=json.loads((S/'runs/reference-near-03/reference-gate.json').read_text())
    assert gate['status']=='reference_qualified','No training before valid reference domain'
    old=P/'experiments/interaction-pilot-v3-20261112/runs/pilot-B-v3/checkpoints/round-19'
    ck=load_state(old);assert sha(old/'core.ptz')=='0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984'
    approved=json.loads((ROOT/'research_notes/adapter_return_pilot/preparation-summary.json').read_text())
    indexfile=S/'data/archive-checks.json';assert sha(indexfile)==approved['full_index_manifest_sha256']
    idx=json.loads(indexfile.read_text())
    rows=[ck['rows'][i] for i in idx['replay_indices']];native=[ck['native'][i] for i in idx['native_indices']]
    assert len(rows)==1255 and len(native)==205
    # Known B19 exact off archive only qualifies its complete first input;
    # old successors lack deploy history and are never invented as labels.
    first=torch.load(P/'experiments/interaction-envelope-v1-20261112/runs/qualification-B19-01/envelope-sample/qualification-trace.pt',map_location='cpu',weights_only=False)
    b19ref=first['actions'][0];assert abs(b19ref['time']-11.8)<1e-7
    known=[]
    for mode in ('offmean','offsample'):
        basepath=P/f'experiments/interaction-pilot-adapter-off-20261112/runs/{mode}-B19'
        trace=torch.load(basepath/'probe-trace.pt',map_location='cpu',weights_only=False)
        summary=json.loads((basepath/'probe.json').read_text())
        assert summary['status']=='completed_3s' and len(trace['physics'])==600
        assert trace['task_spec']['parent_id']==idx['training_parent_ids'][-1]
        assert max(r['joint_limit_excess_max_rad'] for r in trace['physics'])==0
        assert all(any(r['support']) for r in trace['physics'])
        controls=[r for r in trace['controls'] if r['time']>=11.8-1e-8]
        assert torch.equal(torch.tensor(controls[0]['executed_z']),b19ref['z'])
        for k,v in controls[0]['bfm_obs'].items():
            assert torch.equal(torch.tensor(v),b19ref['observation'][k])
        known.append(dict(mode=mode,trace_sha256=sha(basepath/'probe-trace.pt')))
    near=[]
    for mode in ('off-mean','off-sample'):
        trace=torch.load(S/f'runs/reference-near-03/{mode}/trace.pt',map_location='cpu',weights_only=False)
        assert trace['mode']=='off' and len(trace['physics'])==600
        assert trace['task_spec']['parent_id']=='2670e363f8c204b64ab32dd614237f032a571e1cd03f4e5c50524c3caef42f79'
        near.extend(trace['actions'])
    refs=[[b19ref],near];assert len(near)==300
    from humanoidverse.agents import load_utils
    modeldir=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint'
    assert sha(modeldir/'model/model.safetensors')=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
    base=load_utils.load_model_from_checkpoint_dir(str(modeldir),device='cuda');base.eval().requires_grad_(False)
    base_hash=tree_digest(base.state_dict())
    operator=json.loads((ROOT/'research_notes/adapter_envelope_lock_v1.json').read_text())['operator']
    initial=ck['agent']
    def clone():
        agent=Agent(base,to_device(rows[0]['observation']),initial['goal'].cuda(),initial['scale'].cuda(),'B')
        agent.load(copy.deepcopy(initial));agent.envelope=Envelope(operator,'cuda')
        for name,net in agent.nets.items():net.requires_grad_(name=='adapter')
        agent.targets.requires_grad_(False)
        return agent
    gen=torch.Generator().manual_seed(6262013);schedule=[]
    for _ in range(64):
        ix=torch.randint(len(rows),(32,),generator=gen).tolist();ex=torch.randint(len(native),(32,),generator=gen).tolist()
        # Exactly16 per independent parent, with replacement within its qualified rows.
        rs=[(p,int(torch.randint(len(refs[p]),(),generator=gen))) for p in (0,1) for _ in range(16)]
        schedule.append(dict(replay=ix,native=ex,return_refs=rs))
    write(out/'batch-schedule.json',schedule)
    def batch(item):return ([rows[i] for i in item['replay']],[native[i] for i in item['native']],
                            [refs[p][i] for p,i in item['return_refs']])
    # Two disposable engineering optimizer steps, neither adopted by training.
    e0,e1=clone(),clone();assert tree_digest(e0.state())==tree_digest(e1.state())
    b=batch(schedule[0]);r0=step(e0,*b,0,baseline=True);r1=step(e1,*b,0)
    assert tree_digest(e0.state())==tree_digest(e1.state()),'Lambda0 baseline update mismatch'
    write(out/'engineering.json',dict(lambda0_exact_adapter_update=True,disposable_steps=2,
        first_loss=r0,control_loss=r1,base_frozen=True,other_optimizer_steps=0,
        full_Agent_update_called=False,known_reference_archive=known))
    del e0,e1
    counts={};identities={}
    for name,weight in [('control',0.),('return',.0008)]:
        agent=clone();dest=out/name;dest.mkdir();logs=[];fp=frozen_digest(agent)
        def save(it):
            path=dest/f'adapter-{it:02d}.pt';assert not path.exists()
            payload=dict(agent=agent.state(),parent_core_sha256=sha(old/'core.ptz'),parent_updates=38,
                         incremental_adapter_updates=it,weight=weight,seed=6262013,
                         runtime=os.popen('git -C '+str(ROOT)+' rev-parse HEAD').read().strip(),
                         batch_schedule_sha256=sha(out/'batch-schedule.json'),frozen_digest=fp)
            torch.save(payload,path)
            identities[f'{name}{it}']=dict(sha256=sha(path),agent_digest=tree_digest(agent.state()))
            write(out/'checkpoint-identities.json',identities)
        save(0)
        for i,item in enumerate(schedule,1):
            rec=step(agent,*batch(item),weight);rec['adapter_step']=i;logs.append(rec)
            write(dest/'learning.json',logs)
            if i in (16,32,64):save(i)
            if i%16==0:print('ADAPTER_COMMITTED',name,i,rec['raw_return'],flush=True)
        assert frozen_digest(agent)==fp;counts[name]=64
        del agent
    assert identities['control0']['agent_digest']==identities['return0']['agent_digest']
    assert tree_digest(base.state_dict())==base_hash
    result=dict(status='adapter_fit_completed_physical_evaluation_pending',updates=counts,
                disposable_steps=2,new_RL_transitions=0,wall_s=time.monotonic()-start,
                return_training_parents=2,return_rows_by_parent=[1,300],
                notes='Parent-balanced; one original known state versus real qualified successor rows. Not301 independent states.',
                checkpoints=identities)
    write(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
