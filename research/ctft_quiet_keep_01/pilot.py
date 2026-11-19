"""Paired isolated update of a verified U collection block, no writes to formal stage."""
from pathlib import Path
import os,sys,time,copy,json,hashlib,inspect,textwrap,types,gzip,pickle
ROOT=Path(os.environ['CTFT_PINNED_RUNTIME']);STAGE=ROOT.parent;OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(ROOT/'research/bfm_ctft_v1')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc
from cas_checkpoint import durable
def write(p,v):durable(p,(json.dumps(v,indent=2,allow_nan=False)+'\n').encode())
def read(p):return json.loads(p.read_text())
def bank(e,a,b0,contract):
    import numpy as np,mujoco as mj,torch
    from features import ForceHistory,actor_fields
    from agent import ACTOR_KEY,CRITIC_KEY
    records=[r for r in read(OUT/'train-source-audit.json') if r['prequalified']]
    assert records
    obs=[];zs=[];ys=[]
    model=e.sim.model;d=mj.MjData(model);w=model.body('right_wrist_yaw_link').id;r=model.body('pelvis').id
    for record in records:
        with gzip.open(STAGE/'runs/s0-native-sources-03'/f"source-{record['case']:02d}.pkl.gz",'rb') as f:x=pickle.load(f)
        for i in [-49,-33,-17,-1]:
            j=len(x['controls'])+i
            ds=x['states'][j]['mjdata_arrays'];d.qpos[:]=ds['qpos'];d.qvel[:]=ds['qvel'];mj.mj_forward(model,d)
            c=x['controls'][j];assert torch.equal(c['z'].reshape(-1).cpu(),e.standing.reshape(-1).cpu())
            prev=x['physics'][(j-10)*4:j*4:4];hist=np.array([h['pelvis_matrix'].T@h['force'] for h in prev]);assert not np.any(hist)
            jac=np.zeros((3,model.nv));mj.mj_jac(model,d,jac,None,d.xpos[w],w)
            v=actor_fields(force_history=ForceHistory(hist),known_mount=[0,0,0],wrist_world=d.xpos[w],root_world=d.qpos[:3],wrist_velocity_world=jac@d.qvel,root_velocity_world=d.qvel[:3],world_from_body=d.xmat[r].reshape(3,3),public_mode=1)
            obs.append(dict(c['bfm_observation'],**{ACTOR_KEY:torch.tensor(v[None]),CRITIC_KEY:torch.zeros(1,9)}));zs.append(c['z']);ys.append(c['q_PD'])
        del x
    o={k:torch.cat([x[k] for x in obs]).to(a.device) for k in obs[0]};z=torch.cat(zs).to(a.device);mat=a._model
    with torch.no_grad():o=mat._normalize(o)
    sc=torch.as_tensor(contract['q_PD_scale'],device=a.device).reshape(1,-1);off=torch.as_tensor(contract['q_PD_offset'],device=a.device).reshape(1,-1)
    def convert(action):return off+(5*action).clamp(-contract['clip'],contract['clip'])*sc/5
    with torch.no_grad():target=convert(b0(o,z,mat.cfg.actor_std).mean)
    err=float((target-torch.cat(ys).to(a.device)).abs().max())
    write(OUT/'bank.json',dict(ancestors=[r['ancestor'] for r in records],samples=len(obs),max_B0_qPD_record_error_rad=err))
    assert err<1e-5
    def loss():
        y=convert(mat._actor(o,z,mat.cfg.actor_std).mean)
        return ((y-target)/.1).square().mean()
    return loss
def study(e,original,contract,out):
    import torch
    from learning_checkpoint import digest,set_rng,rng
    from training_live import TrainingLive
    from training_data import episode
    from contact_replay import cat
    from packed_checkpoint import PackedTensorStore
    from cas_checkpoint import restore_learner
    from f1_mjdata_contract import canonical,metadata
    from m8_stage import reset_common
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
    speed(e);(out/'backend-prefix').mkdir();backend(e,original,out/'backend-prefix');reset_common(e,original,original.act)
    hist=STAGE/'runs/formal-U-v1a-01';store=PackedTensorStore(hist/'store')
    pkg=store.load('round-130048');model,buf=restore_learner(pkg['learner'])
    records=read(STAGE/'runs/s0-source-admission-01/result.json')['train']
    live=TrainingLive(e,STAGE,read(STAGE/'runs/s0-online-probe-01/agent-identity.json')['original_training_config']['aux_rewards_scaling'],records)
    live.restore(pkg['task']);set_rng(pkg['learner']['rng'])
    assert digest(canonical(live.snapshot(),metadata(e.sim.model,e.sim.data)))==digest(canonical(pkg['task'],metadata(e.sim.model,e.sim.data)))
    warm=PackedTensorStore(STAGE/'runs/formal-U-01/store');wm=warm.metadata('round-8192')['learner']['model']
    b0=copy.deepcopy(model._model._actor);b0.load_state_dict({k[7:]:warm.unpack(v) for k,v in wm.items() if k.startswith('_actor.')});b0.requires_grad_(False)
    sched=copy.deepcopy(pkg['scheduler']);plan=read(STAGE/'locks/exogenous-training-plan-v2.json');reset=read(STAGE/'locks/source-reset-domain-v1.json')
    rows=[]
    for _ in range(1024):
        if live.done:
            live.start(episode(plan,reset,sched['source_index'],'U'));sched['source_index']+=1
        ob=live.observation();z=live.command()
        with torch.no_grad():
            a=model.act(ob,z,mean=False) if live.spec['mode'] else b0(model._model._normalize(ob),z,model._model.cfg.actor_std).mean.float()
        rows.append(live.step(a,ob,z))
    data=cat(rows);start_rng=rng()
    actual=digest(data);expected=digest(torch.load(hist/'actual-rows-131072.pt',map_location='cpu',weights_only=False))
    write(OUT/'collection.json',dict(actual=actual,archive=expected,rows=1024));assert actual==expected,'archived block identity'
    for arm in ['control','keep']:
        if arm=='keep':model,buf=restore_learner(pkg['learner'])
        buf['train'].extend(data)
        hold=bank(e,model,b0,contract)
        if arm=='keep':
            fn=FBcprAuxAgent.update_actor
            body=textwrap.dedent(inspect.getsource(fn))
            anchor='        actor_loss.backward()'
            anchor='    actor_loss.backward()'
            assert body.count(anchor)==1
            env=dict(fn.__globals__,retention_loss=hold)
            exec(compile(body.replace(anchor,'    actor_loss = actor_loss + retention_loss()\n'+anchor),'<quiet-standing-keep>','exec'),env)
            model.update_actor=types.MethodType(env['update_actor'],model)
        set_rng(start_rng)
        log=[]
        for i in range(8):
            with torch.enable_grad():metrics=model.update(buf,pkg['learner']['step']+i)
            with torch.no_grad():log.append(dict(step=i+1,quiet_loss=float(hold()),native_actor_loss=float(metrics['actor_loss'])))
        sha=digest(model._model._actor.state_dict())
        write(OUT/(arm+'.json'),dict(actor_sha=sha,steps=log))
        if arm=='control':assert sha=='a4026430c821128e18a8f22f5531413dc393fa71d9f1483812b3ad928022569b','control endpoint mismatch'
        else:torch.save({k:v.cpu() for k,v in model._model._actor.state_dict().items()},OUT/'keep-actor.pt')
        del hold,model,buf,metrics
    return dict(status='complete')
if __name__=='__main__':
    from resource_guard import check
    import mujoco
    out=OUT/os.environ['ATTEMPT'];out.mkdir(exist_ok=False)
    start=time.monotonic();write(out/'preflight.json',check(out))
    ticks=0;step=mujoco.mj_step
    def counted(*args,**kw):
        global ticks
        ticks+=int(kw.get('nstep',1));assert ticks<=12000;return step(*args,**kw)
    mujoco.mj_step=counted
    os.environ.update(BFM_RECOVERY_OUTPUT=str(out),CCL_DEADLINE_MONOTONIC=str(start+1800))
    import m10_stage,m10_physics
    m10_stage.run_study=study;m10_physics.OUT=out
    try:m10_physics.main()
    finally:write(out/'cost.json',dict(ticks=ticks,seconds=time.monotonic()-start))
