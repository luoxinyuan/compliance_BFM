"""Recorded-state oracle labels, latent feasibility, identifiable inputs, and one fit.
No anchor/time/direction/teacher target ever enters the deployed Projector.
"""
import copy,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
from contact_projector import ContactGate,Projector,deploy_input,constrained,check_deadline,save
from ccl02a_teacher import arm_ik
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUNS=P/'bridge-v2-owned/runs'
SHARED=RUNS/'issue21-bidirectional-20261030T012400Z/attempt-01/outputs'
SINGLE=RUNS/'issue21-projector-20261030T004000Z/attempt-01/outputs'
PROBE=RUNS/'issue21-direction-20261030T020000Z/attempt-02/outputs'
TEACHER=RUNS/'issue21-directionteacher-20261030T031000Z/attempt-01/outputs'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def query_labels(env,episode,contract):
    """Kinematic oracle on a SEPARATE MjData, never advances or mutates real env.
    10Hz bounded correction uses start-of-block nominal; five labels use actual
    recorded Student 50Hz nominal actions plus that held correction.
    """
    import mujoco
    ep=copy.deepcopy(episode);m=env.sim.model;data=mujoco.MjData(m)
    arm=contract['arm_indices'];ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in contract['dof_names']]
    vadr=m.jnt_dofadr[ids];limits=m.jnt_range[ids];scale=np.asarray(contract['q_PD_scale'])[0];offset=np.asarray(contract['q_PD_offset'])[0]
    target_a=[];target_q=[];corrections=[];errors=[];point_checks=[];jacs=[]
    for i,d in enumerate(ep['decisions']):
        cs=ep['control'][i*5:i*5+5];first=cs[0];assert len(cs)==5
        data.qpos[:]=first['pre_qpos'];data.qvel[:]=first['pre_qvel'];mujoco.mj_forward(m,data)
        point=data.xpos[env.tether.body]+data.xmat[env.tether.body].reshape(3,3)@np.asarray(env.tether.cfg.local_point)
        jac=np.zeros((3,m.nv));mujoco.mj_jac(m,data,jac,None,point,env.tether.body)
        # Saved pre_error/point refer to this exact start; no hidden time or label shortcut.
        anchor=np.asarray(first['pre_error'])+np.asarray(first['pre_wrist']);error=anchor-point
        point_checks.append(float(abs(point-first['pre_wrist']).max()))
        active=False if i==0 else bool(ep['physics'][i*20-1]['active'])
        correction=np.zeros(7)
        if active:
            correction=arm_ik(jac[:,vadr[arm]],error)
            q0=np.asarray(first['nominal_q_PD']);correction=np.clip(q0[arm]+correction,limits[arm,0],limits[arm,1])-q0[arm]
        assert abs(correction).max()<.040001 and np.linalg.norm(correction)<.080001
        da=np.zeros(29);da[arm]=correction/scale[arm]
        for c in cs:
            # Float32, same arithmetic as actual teacher callback.
            action=(np.asarray(c['nominal'],dtype=np.float32)+da.astype(np.float32)).astype(np.float32)
            q=offset+np.clip(action,-1,1)*scale;target_a.append(action.tolist());target_q.append(q.tolist())
        corrections.append(correction.tolist());errors.append(error.tolist());jacs.append(jac[:,vadr[arm]].tolist())
    assert max(point_checks)<1e-10,'Recorded Student pre-state missing kinematic state'
    return dict(target_action=target_a,target_q=target_q,correction=corrections,error=errors,jacobian=jacs,point_reconstruction_error=max(point_checks),privileged_labels_only=True)

def make_dataset(env,model,standing,contract,net,out):
    paths=[SINGLE/'dev-2-teacher.json',SHARED/'dev-3-teacher.json',TEACHER/'plus-teacher.json',TEACHER/'minus-teacher.json',PROBE/'plus-student.json',PROBE/'minus-student.json']
    episodes=[read(p) for p in paths];dev=standing.device;keys=env.keys;records=[];queries=[];checks=[]
    # Rebuild labels of physically executed teacher first; validates Jacobian and chronology.
    for ep in episodes[2:4]:
        oracle=query_labels(env,ep,contract)
        q=np.asarray([c['q_PD'] for c in ep['control']]);act=np.asarray([c['action'] for c in ep['control']]);corr=np.asarray([d['correction_rad'] for d in ep['decisions']])
        chk=dict(q_max=float(abs(q-oracle['target_q']).max()),action_max=float(abs(act-oracle['target_action']).max()),correction_max=float(abs(corr-oracle['correction']).max()),point_max=oracle['point_reconstruction_error'])
        assert chk['q_max']<1e-6 and chk['action_max']<1e-6 and chk['correction_max']<1e-6
        checks.append(chk)
    for j,ep in enumerate(episodes):
        gate=ContactGate();visit=j>=4;oracle=query_labels(env,ep,contract) if visit else None
        if visit:queries.append(oracle)
        onset=1.1 if j<2 else 3.
        for i,d in enumerate(ep['decisions']):
            cs=ep['control'][i*5:i*5+5];force=d.get('force_xyz_n',d.get('force_xyz'));now=d.get('tether_time',d['time'])
            gate.update(now,force)
            # First 50Hz observation is exactly the decision observation.
            obs0=cs[0]['bfm_obs'];history=d['force_history']
            if 'bfm_obs' in d:assert all(d['bfm_obs'][k]==obs0[k] for k in keys)
            ob={k:torch.tensor(obs0[k],device=dev,dtype=torch.float32) for k in keys}
            h=deploy_input(model,ob,history,keys)[0].detach()
            if 'student_h' in d:assert torch.equal(h,torch.tensor(d['student_h'][0],device=dev))
            for c in cs:
                if visit:assert abs(c['gate']-gate.value(c['tether_time']))<1e-8
            records.append(dict(group=j,local_i=i,time=d['time'],h=h,force=force,obs={k:[c['bfm_obs'][k][0] for c in cs] for k in keys},
                target_q=oracle['target_q'][i*5:i*5+5] if visit else [c['q_PD'] for c in cs],
                target_action=oracle['target_action'][i*5:i*5+5] if visit else [c['action'] for c in cs],
                nominal_q=[c['nominal_q_PD'] for c in cs],gate=[gate.value(c['tether_time']) for c in cs],
                weight=[1. if c['tether_time']<onset else 5. if c['tether_time']<4.5 else 2. for c in cs],
                jac=oracle['jacobian'][i] if visit else (cs[0].get('rightarm_jacobian')),
                pre_error=oracle['error'][i] if visit else cs[0].get('pre_error')))
    dataset=dict(h=torch.stack([r['h'] for r in records]),obs={k:torch.tensor([r['obs'][k] for r in records],device=dev,dtype=torch.float32).flatten(0,1) for k in keys},
        target_q=torch.tensor([r['target_q'] for r in records],device=dev),target_action=torch.tensor([r['target_action'] for r in records],device=dev),
        nominal_q=torch.tensor([r['nominal_q'] for r in records],device=dev),gate=torch.tensor([r['gate'] for r in records],device=dev),weights=torch.tensor([r['weight'] for r in records],device=dev),
        group=torch.tensor([r['group'] for r in records],device=dev),metadata=[{k:r[k] for k in ['group','local_i','time','force','jac','pre_error']} for r in records])
    save(out,'query-label-parity.json',checks);save(out,'student-visited-oracle-labels.json',queries)
    save(out,'reused-data.json',{str(p):sha(p) for p in paths})
    save(out,'dataset-manifest.json',dict(decisions=len(records),controls=len(records)*5,group_counts=[int((dataset['group']==j).sum()) for j in range(6)],
        data_order=['old+','old-','new teacher+','new teacher-','Student visited+','Student visited-'],
        oracle_queries='20 recorded10Hz Student states; no physical rollout, no teacher intervention into Student; held q correction labels on5 actual recorded50Hz observations',
        normalization='existing checkpoint center/scale unchanged; force /20 unchanged',heldout_used=False))
    torch.save({k:({n:t.cpu() for n,t in v.items()} if k=='obs' else v.cpu() if torch.is_tensor(v) else v) for k,v in dataset.items()},out/'repair-dataset.pt')
    return dataset

class Alignment:
    def __init__(self,model,standing,data,contract):
        self.model,self.standing,self.data=model,standing,data;self.dev=standing.device
        self.norm={k:v.detach() for k,v in model._normalize(data['obs']).items()}
        self.scale=torch.tensor(contract['q_PD_scale'],device=self.dev);self.offset=torch.tensor(contract['q_PD_offset'],device=self.dev)
    def evaluate(self,raw,ids):
        ix=(ids[:,None]*5+torch.arange(5,device=self.dev)).reshape(-1)
        z,t=constrained(self.model,self.standing,raw.repeat_interleave(5,0),self.data['gate'][ids].flatten())
        action=self.model._actor({k:v[ix] for k,v in self.norm.items()},z,self.model.cfg.actor_std).mean.float()
        q=self.offset+action.clamp(-1,1)*self.scale;error=q-self.data['target_q'][ids].flatten(0,1)
        w=self.data['weights'][ids].flatten()
        loss=((error[:,22:].square().mean(-1)+.1*error[:,:22].square().mean(-1))*w).sum()/w.sum()+1e-6*raw.tanh().square().mean()
        return loss,action,q,error,z

def precheck(env,model,standing,contract,net,out):
    data=make_dataset(env,model,standing,contract,net,out);f=Alignment(model,standing,data,contract);dev=standing.device
    with torch.no_grad():
        public=model.act({k:v[:5] for k,v in data['obs'].items()},standing.expand(5,-1),mean=True)
        direct=model._actor({k:v[:5] for k,v in f.norm.items()},standing.expand(5,-1),model.cfg.actor_std).mean.float()
    parity=float((public-direct).abs().max());assert parity<1e-6
    selected=[i for i,r in enumerate(data['metadata']) if r['group'] in [3,5] and r['local_i'] in [2,4,5]];probes=[]
    for idx in selected:
        ids=torch.tensor([idx],device=dev);m=data['metadata'][idx]
        with torch.enable_grad():
            raw=torch.zeros(1,256,device=dev,requires_grad=True);opt=torch.optim.Adam([raw],lr=.05)
            _,_,_,e0,_=f.evaluate(raw,ids);initial=float(e0[:,22:].square().mean().sqrt().detach())
            for step in range(200):
                check_deadline();opt.zero_grad();loss,*_=f.evaluate(raw,ids);loss.backward();opt.step()
            _,act,q,error,z=f.evaluate(raw,ids)
            nominal=data['nominal_q'][idx];target=data['target_q'][idx];jac=torch.tensor(m['jac'],device=dev);err=torch.tensor(m['pre_error'],device=dev)
            cart=(q-nominal)[:,22:]@jac.T;desired=(target-nominal)[:,22:]@jac.T
            row=dict(group=m['group'],time=m['time'],gate=data['gate'][idx].tolist(),initial_arm_rms_rad=initial,
                final_arm_rms_rad=float(error[:,22:].square().mean().sqrt().detach()),nonarm_rms_rad=float(error[:,:22].square().mean().sqrt().detach()),
                target_arm_rms_rad=float((target-nominal)[:,22:].square().mean().sqrt()),target_nonarm_rms_rad=float((target-nominal)[:,:22].square().mean().sqrt()),
                q_match_peak_rad=float(error.abs().max().detach()),action_rms=float((act-data['target_action'][idx]).square().mean().sqrt().detach()),
                cart_target_mean_m=desired.mean(0).tolist(),cart_projected_mean_m=cart.mean(0).detach().tolist(),
                cart_toward_error_m=float((cart@err/err.norm().clamp_min(1e-12)).mean().detach()),
                angle_max_deg=float(torch.rad2deg(torch.acos(((z*standing).sum(-1)/256).clamp(-1,1))).max().detach()),
                norm_error=float((z.norm(dim=-1)-16).abs().max().detach()),raw_tanh_max=float(raw.tanh().abs().max().detach()))
            probes.append(row);save(out,'reachability.json',probes);print('REACHABILITY',json.dumps(row),flush=True)
        assert all(p.grad is None for p in model.parameters())
    # Exact/small-margin input diagnostics; no new classifier or sensor normalization is trained.
    x=((data['h']-net.center)/net.scale).detach();qdelta=data['target_q']-data['nominal_q'];diagnostics=[];collisions=[]
    indices=[i for i,r in enumerate(data['metadata']) if r['group']>=2]
    for i in indices:
        mi=data['metadata'][i];others=[j for j in indices if data['metadata'][j]['group']%2!=mi['group']%2]
        distances=(x[others]-x[i]).square().mean(-1).sqrt();j=others[int(distances.argmin())];mj=data['metadata'][j]
        delta=float((qdelta[j,:,22:]-qdelta[i,:,22:]).square().mean().sqrt())
        row=dict(group=mi['group'],time=mi['time'],nearest_group=mj['group'],nearest_time=mj['time'],input_rms=float(distances.min()),input_max=float((x[j]-x[i]).abs().max()),teacher_delta_difference_rms_rad=delta,force_xyz=mi['force'],opposite_force_xyz=mj['force'],force_feature_l2=float((x[j,-9:]-x[i,-9:]).norm()))
        diagnostics.append(row)
        if row['input_max']<1e-6 and delta>1e-4:collisions.append(row)
    pairs=[]
    for group in [2,4]:
        for local in range(10):
            i=next(i for i,m in enumerate(data['metadata']) if m['group']==group and m['local_i']==local);j=next(i for i,m in enumerate(data['metadata']) if m['group']==group+1 and m['local_i']==local)
            pairs.append(dict(group=group,time=data['metadata'][i]['time'],input_rms=float((x[i]-x[j]).square().mean().sqrt()),proprio_rms=float((x[i,:-9]-x[j,:-9]).square().mean().sqrt()),force_feature_l2=float((x[i,-9:]-x[j,-9:]).norm()),force_plus=data['metadata'][i]['force'],force_minus=data['metadata'][j]['force'],target_difference_rms_rad=float((qdelta[i,:,22:]-qdelta[j,:,22:]).square().mean().sqrt())))
    ident=dict(exact_conflicting_pairs=collisions,nearest_opposite=diagnostics,matched_times=pairs,public_actor_parity=parity,
        caveat='Finite noiseless recorded-data separability only; small distances alone do not prove unidentifiability; sensor noise robustness not established; no inference from task labels')
    save(out,'identifiability.json',ident)
    return dict(projection_cases=len(probes),identical_input_conflicts=len(collisions),student_updates=0,physics_steps=0,review_required_before_fit=True)

def balanced_ids(data):
    parts=[]
    for j,n in enumerate([4,4,2,2,2,2]):
        ids=torch.nonzero(data['group']==j,as_tuple=True)[0];parts.append(ids[torch.randperm(len(ids),device=ids.device)[:n]])
    return torch.cat(parts)

def fit_repair(model,standing,contract,net,data,out):
    f=Alignment(model,standing,data,contract);dev=standing.device;allids=torch.arange(len(data['h']),device=dev)
    buffers={n:v.clone() for n,v in net.named_buffers()};torch.manual_seed(0);net.train().requires_grad_(True)
    opt=torch.optim.Adam(net.parameters(),lr=.0003);trace=[]
    def stats():
        with torch.no_grad():
            raw=net(data['h']);loss,act,q,e,z=f.evaluate(raw,allids);groups={}
            for j in range(6):
                ix=(data['group']==j).repeat_interleave(5);groups[str(j)]=dict(arm_rms_rad=float(e[ix,22:].square().mean().sqrt()),nonarm_rms_rad=float(e[ix,:22].square().mean().sqrt()),action_rms=float((act[ix]-data['target_action'].flatten(0,1)[ix]).square().mean().sqrt()))
            return dict(loss=float(loss),groups=groups,z_norm_error=float((z.norm(dim=-1)-16).abs().max()),angle_max_deg=float(torch.rad2deg(torch.acos(((z*standing).sum(-1)/256).clamp(-1,1))).max()))
    before=stats();save(out,'fit-before.json',before)
    with torch.enable_grad():
        for step in range(2000):
            check_deadline();ids=balanced_ids(data);opt.zero_grad();loss,*_=f.evaluate(net(data['h'][ids]),ids);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            assert torch.isfinite(gn) and all(p.grad is None for p in model.parameters())
            opt.step()
            if step%100==0 or step==1999:
                row=dict(step=step+1,grad_norm=float(gn),**stats());trace.append(row);save(out,'fit-trace.json',trace);print('FIT_REPAIR',step+1,row['loss'],flush=True)
    after=stats();save(out,'fit-after.json',after)
    assert all(torch.equal(v,buffers[n]) for n,v in net.named_buffers())
    net.eval().requires_grad_(False)
    for p in net.parameters():p.grad=None
    torch.save(dict(state_dict=net.state_dict(),updates=2000,contract=contract,hidden=128,initial_checkpoint_sha256=sha(SHARED/'contact-projector-final.pt')),out/'contact-projector-repaired-final.pt')
    save(out,'checkpoint-frozen-before-evaluation.json',dict(sha256=sha(out/'contact-projector-repaired-final.pt'),updates=2000,test_results_seen=False))
    return dict(updates=2000,before=before,after=after)
