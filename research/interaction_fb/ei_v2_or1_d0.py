"""OR1-D0: archived true rollout, one Adam proposal per arm, no simulator or new sampling."""
import os,sys,json,time,copy,argparse,hashlib,subprocess,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;ARCHIVE=STAGE.parent/'contact-goal-ei-v2-20261115'
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc
import numpy as np
import torch
from torch.nn import functional as F
from .ei_v2_or1_optimizer import one_proposal,exact,rng_state
from .ei_v2_ppo import segments,current,advantages,retention,kl,device_obs
from .ei_v2_agent import ContactGoalAgent,SIGMA
from .ei_v2_checkpoint import load
from .journal import read,dump
from diagnostic_fb_resume import tree_digest
from dual_track_task import restore_global_rng

def quant(x):
    x=np.asarray(x,dtype=float)
    return dict(min=float(x.min()),p50=float(np.quantile(x,.5)),p95=float(np.quantile(x,.95)),max=float(x.max())) if x.size else None

def metrics(agent,rows,groups,ix,adv,keep,eps,lam,alpha,out,tag):
    with torch.no_grad():
        lp,old=current(agent,rows,groups,list(range(len(groups))));ratio=(lp-old).exp();task=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
        keep_loss,violation=retention(agent,keep,eps);objective=task[ix].mean()+lam*keep_loss
        a=[];q=[];tables=[];by={};selector=[]
        scale=torch.as_tensor(eps['qpd_scale'],device=agent.device);offset=torch.as_tensor(tag['action_contract']['q_PD_offset'],device=agent.device);clip=eps['original_action_clip']
        for i,r in enumerate(rows):
            ob=device_obs(r['observation'],agent.device);z=r['z'].to(agent.device);op=torch.tensor([r['option']],device=agent.device)
            mu=agent.mean(ob,z,op);delta=mu-r['old_mu'].to(agent.device)
            olda=r['old_mu'].to(agent.device).clamp(-1,1);act=mu.clamp(-1,1)
            qp=offset+(mu*5).clamp(-clip,clip)*scale/5;oldq=offset+(r['old_mu'].to(agent.device)*5).clamp(-clip,clip)*scale/5
            coupled_u=r['raw_u'].to(agent.device)+delta;coupled_action=coupled_u.clamp(-1,1);coupled_q=offset+(coupled_action*5).clamp(-clip,clip)*scale/5;original_executed_q=offset+(r['action'].to(agent.device)*5).clamp(-clip,clip)*scale/5
            a.append(delta.cpu().numpy());q.append((qp-oldq).cpu().numpy())
            pid=bytes(r['provenance']['parent_sha256'].reshape(-1).tolist()).hex();goal='FOLLOW' if r['goal_mode']==0 else 'HOLD'
            phase=tag['row_phase'][i];key=f'{pid}|{goal}|{phase}'
            stat=by.setdefault(key,dict(rows=0,gaussian_KL_sum=0.,action_delta_sumsq=0.,qpd_delta_sumsq=0.,qpd_absmax=0.))
            stat['rows']+=1;stat['gaussian_KL_sum']+=float(delta.square().sum()/(2*SIGMA**2));stat['action_delta_sumsq']+=float(delta.square().sum());stat['qpd_delta_sumsq']+=float((qp-oldq).square().sum());stat['qpd_absmax']=max(stat['qpd_absmax'],float(abs(qp-oldq).max()))
            tables.append(dict(row=i,parent=pid,episode=r['episode'],macro_slot=r['macro_slot'],goal=goal,phase=phase,previous_z=r['previous_z'],interpolated_z=r['z'],raw_u=r['raw_u'],old_mu=r['old_mu'],candidate_mu=mu.cpu(),candidate_action=act.cpu(),old_mean_action=olda.cpu(),same_noise_counterfactual_raw=coupled_u.cpu(),same_noise_counterfactual_action=coupled_action.cpu(),same_noise_counterfactual_q_PD=coupled_q.cpu(),original_actual_q_PD=original_executed_q.cpu(),counterfactual_not_physically_executed=True,q_PD=qp.cpu(),old_mean_q_PD=oldq.cpu(),clamp_fraction=float((abs(mu)>=1).float().mean())))
            if r['cat_boundary']:
                old_logits=r['old_logits'].to(agent.device);new_logits=agent.logits(device_obs(r['selector_obs'],agent.device),r['candidate_first_z'].to(agent.device));oldlog=F.log_softmax(old_logits,-1);newlog=F.log_softmax(new_logits,-1)
                d=newlog.exp()-oldlog.exp();selector.append(dict(row=i,parent=pid,goal=goal,phase=phase,option=r['option'],old_probability=oldlog.exp().cpu().tolist(),new_probability=newlog.exp().cpu().tolist(),probability_max_delta=float(d.abs().max()),conditional_KL_fp64=float((F.log_softmax(old_logits.double(),-1).exp()*(F.log_softmax(old_logits.double(),-1)-F.log_softmax(new_logits.double(),-1))).sum())))
        dd=np.concatenate(a);qq=np.concatenate(q)
        for s in by.values():
            s['gaussian_KL_native_mean']=s['gaussian_KL_sum']/s['rows'];s['mean_delta_RMS']=math.sqrt(s['action_delta_sumsq']/(29*s['rows']));s['qPD_delta_RMS']=math.sqrt(s['qpd_delta_sumsq']/(29*s['rows']))
        table_meta=dump(out/f'{agent.arm}-alpha-{alpha:g}-same-state.ptz',dict(rows=tables,selector=selector))
        return dict(KL=kl(agent,rows,groups),objective=float(objective),minibatch_task_loss=float(task[ix].mean()),full_block_task_loss=float(task.mean()),keep_loss=float(keep_loss),keep_violation=float(violation),lambda_keep=lam,ratio=quant(ratio.cpu().numpy()),log_ratio=quant((lp-old).cpu().numpy()),clip_fraction=float(((ratio-1).abs()>.2).float().mean()),minibatch_clip_fraction=float(((ratio[ix]-1).abs()>.2).float().mean()),S_mu=float(np.sqrt(np.mean(dd**2))/SIGMA),mean_qPD_abs_p95_rad=float(np.quantile(abs(qq),.95)),mean_qPD_RMS_rad=float(np.sqrt(np.mean(qq**2))),mean_qPD_abs_max_rad=float(abs(qq).max()),qPD_per_joint_abs_max=abs(qq).max(0).tolist(),qPD_per_joint_abs_p95=np.quantile(abs(qq),.95,axis=0).tolist(),selector_probability_max_delta=max([s['probability_max_delta'] for s in selector]or[0]),selector_draws=len(selector),parent_goal_phase=by,same_state_table=table_meta)

def row_phases(state,rows):
    parents=json.loads((ARCHIVE/'locks/execution-train.json').read_text())['parents'];by={p['parent_id']:p for p in parents};order=json.loads((ARCHIVE/'locks/training-order.json').read_text())['parent_ids'];cases=[sp for pid in order for sp in by[pid]['specs']]
    ep={i:p['spec']for i,p in enumerate(state['own']['episodes'])};ep[state['own']['reset_cursor']]=cases[state['own']['spec_id']]
    out=[]
    for r in rows:
        s=ep[r['episode']];t=float(r['provenance']['time'])
        if not s['enabled']:p='none_control'
        elif t<s['onset']-1e-7:p='precontact'
        elif t>=s['release']-1e-7:p='released_recovery'
        elif s.get('stop_time') is not None and s['stop_time']-1e-7<=t<s['stop_time']+s['deceleration']-1e-7:p='stopping'
        elif t>=(s['stop_time']+s['deceleration'] if s.get('stop_time')is not None else s['onset']+s['ramp'])-1e-7:p='holding'
        else:p='moving'
        out.append(p)
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
    lock=json.loads((STAGE/'locks/or1-approved.json').read_text());assert lock['status']=='APPROVED_RESOURCE_PREFLIGHT_PENDING'
    from m10_physics import MODEL
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from frozen_bfm_placement import place_inference_modules
    modelpath=MODEL/'checkpoint';weights=modelpath/'model/model.safetensors';h=hashlib.sha256(weights.read_bytes()).hexdigest();assert h=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
    start=time.monotonic();base=load_model_from_checkpoint_dir(modelpath,device='cpu');placement=place_inference_modules(base,'cuda');(out/'placement.json').write_text(json.dumps(placement,indent=2)+'\n')
    q=ARCHIVE/'runs/P1-frozen-sample-01';qm=json.loads((q/'qualification.json').read_text());keepdata=read(q/'qualified-retention-data.ptz',qm['retention']);keep,eps=keepdata['keep_by_parent'],keepdata['eps'];results=[]
    for arm,n in [('E',4),('I',3)]:
        folder=ARCHIVE/f'runs/P1-scratch-{arm}-02';report=json.loads((folder/'scratch.json').read_text());before=load(folder/'checkpoints/block-mid-last');final=load(folder/f'checkpoints/block-{n}');rows=final['rows'][-1024:];assert len(rows)==1024
        obs=device_obs(rows[0]['observation'],torch.device('cuda:0'));agent=ContactGoalAgent(base,obs,arm,torch.device('cuda:0'));agent.load(before['agent']);agent.selector.requires_grad_(True)
        # Last12 are already saved data; their original CUDA noise/physics are NOT rerun.
        # CPU randperm is independent of those CUDA draws. Exact proposal KL verifies this provenance.
        restore_global_rng(before['rng']);pre=copy.deepcopy(agent.state());groups=segments(rows)
        with torch.no_grad():
            lp,old=current(agent,rows,groups,list(range(len(groups))));err=float(abs(lp-old).max());assert err<=3e-4
        adv,ret,gae=advantages(agent,rows,groups);assert exact(gae,report['records'][n-1]['metrics']['GAE']),'pre-update GAE does not reproduce'
        ix=torch.randperm(len(groups)).tolist()[:64]
        if arm=='E':
            prior=json.loads((ARCHIVE/'runs/P1-step-diagnostic-02/original-proposal-reproduction.json').read_text());assert ix==prior['minibatch_group_indices']
        expected=report['records'][n-1]['metrics']['steps'][0]['KL'];tag=dict(row_phase=row_phases(final,rows),action_contract=json.loads((folder/'action-contract.json').read_text()));assert tag['action_contract']['q_PD_scale']==eps['qpd_scale'];dest=out/arm;dest.mkdir();lam=float(agent.retention_lambda)
        def loss():
            new,old=current(agent,rows,groups,ix);r=(new-old).exp();aa=adv[ix];task=-torch.minimum(r*aa,r.clamp(.8,1.2)*aa).mean();k,_=retention(agent,keep,eps);return task+lam*k
        def evaluate(alpha):return metrics(agent,rows,groups,ix,adv,keep,eps,lam,alpha,dest,tag)
        capsule=dump(dest/'input-capsule.ptz',dict(agent=pre,rows=rows,group_indices=ix,GAE=gae,RNG=rng_state(),source_mid_manifest_sha256=hashlib.sha256((folder/'checkpoints/block-mid-last/manifest.json').read_bytes()).hexdigest(),source_final_manifest_sha256=hashlib.sha256((folder/f'checkpoints/block-{n}/manifest.json').read_bytes()).hexdigest()))
        try:r=one_proposal(agent.modules,agent.opt,loss,evaluate,commit=False,evaluate_all=True,expected_first_KL=expected)
        except BaseException:
            dump(dest/'failed-pending-state.ptz',dict(agent=agent.state(),grads={k:p.grad for k,p in agent.modules.named_parameters()},RNG=rng_state()));raise
        assert exact(agent.state(),pre),'D0 changed archived agent/Adam/value/normalizer state'
        selected=next((v for v in r['trials']if v['alpha']==r['selected_alpha']),None)
        measurable=bool(selected and selected['S_mu']>=.005 and selected['mean_qPD_abs_p95_rad']>=1e-5)
        r.update(arm=arm,block=n,input_capsule=capsule,old_logprob_max_error=err,original_first_KL_exact=True,archived_agent_full_state_restored_exact=True,measurable_execution_step=measurable,source_phase='P3',formal_or_engineering_policy_committed=False,new_physics=0,new_samples=0)
        (dest/'result.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);(out/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print('D0',arm,'selected',r['selected_alpha'],'measurable',measurable,flush=True)
        del agent,before,final,rows,pre
    passed=all(r['measurable_execution_step']for r in results);result=dict(status='OR1_D0_PASS_FOR_NEW_ENGINEERING_ONLY' if passed else 'OR1_D0_STEP_UNUSABLE_STOP',runtime=runtime,results=results,wall_after_model_init_s=time.monotonic()-start,physics_ticks=0,new_on_policy_rows=0,new_policy_committed=False,formal_rows=0,sealed_read=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],flush=True)
if __name__=='__main__':main()
