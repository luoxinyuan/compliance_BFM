"""Replay ONE already attempted PPO proposal; uncommitted scaled-weight sensitivity.
No new policy learning, future task, sampling dataset, optimizer adoption or gate change.
Physical operations only reconstruct the already archived complete E state/continuation.
"""
import os,sys,json,argparse,copy,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import m11_ipc

def study(e,base,c,out):
    import torch,numpy as np
    from native_env import install as contact_install
    from m11_runtime_speed import install as speed
    from rolling_backend import install as backend
    from f1_mjdata_contract import install_restore_guard,exact
    from research.interaction_fb.ei_v2_sources import install
    from research.interaction_fb.ei_v2_env import GoalEnvironment
    from research.interaction_fb.ei_v2_agent import ContactGoalAgent
    from research.interaction_fb.ei_v2_checkpoint import restore,load
    from research.interaction_fb.ei_v2_ppo import segments,current,advantages,retention,kl,device_obs
    from research.interaction_fb.journal import read
    from diagnostic_fb_resume import tree_digest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contact_install();install();speed(e);(out/'backend-prefix').mkdir();backend(e,base,out/'backend-prefix');install_restore_guard()
    parents=json.loads((STAGE/'locks/execution-train.json').read_text())['parents'];order=json.loads((STAGE/'locks/training-order.json').read_text())['parent_ids'];by={p['parent_id']:p for p in parents};parents=[by[k] for k in order]
    env=GoalEnvironment(e,base,c,parents);env.reset(0)
    # Match original frozen contact diagnostics in the recorded physical data.
    from frl_v3_counterfactual import install_contact_details
    from f1_sequence_search import install_post_velocity
    install_contact_details(e);install_post_velocity(e)
    agent=ContactGoalAgent(base,env.proxy.enrich(env.task.extended_observation()),'E',e.standing.device);env.agent=agent;env.proxy.agent=agent
    source=STAGE/'runs/P1-scratch-E-02';summary=json.loads((source/'scratch.json').read_text());restore(source/'checkpoints/block-mid-last',agent,env,out)
    begin=len(env.rows)
    while len(env.rows)-begin<12:env.collect(min(5,12-(len(env.rows)-begin)))
    expected=read(source/'expected-fourth-update.ptz',summary['expected']);assert exact(env.rows[-12:],expected['last_rows'])
    rows=env.rows[-1024:];groups=segments(rows)
    q=STAGE/'runs/P1-frozen-sample-01';m=json.loads((q/'qualification.json').read_text());data=read(q/'qualified-retention-data.ptz',m['retention']);keep,eps=data['keep_by_parent'],data['eps']
    agent.selector.requires_grad_(True)
    with torch.no_grad():
        lp,old=current(agent,rows,groups,list(range(len(groups))));err=float((lp-old).abs().max());assert err<=3e-4
    adv,ret,gae=advantages(agent,rows,groups);order=torch.randperm(len(groups)).tolist();ix=order[:64]
    before=copy.deepcopy(agent.modules.state_dict());beforeopt=copy.deepcopy(agent.opt.state_dict());model_before=tree_digest(before)
    with torch.enable_grad():
        new,old=current(agent,rows,groups,ix);ratio=(new-old).exp();aa=adv[ix];task=-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa).mean();keep_loss,violation=retention(agent,keep,eps)
        loss=task+float(agent.retention_lambda)*keep_loss;agent.opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(agent.modules.parameters(),1.));assert np.isfinite(gn)
        agent.opt.step()
    proposal=copy.deepcopy(agent.modules.state_dict());proposed=kl(agent,rows,groups);original=summary['records'][3]['metrics']['steps'][0]['KL']
    (out/'original-proposal-reproduction.json').write_text(json.dumps(dict(actual=proposed,expected=original,exact=exact(proposed,original),minibatch_group_indices=ix),indent=2)+'\n')
    assert exact(proposed,original),'original already logged proposal not exactly reproduced; no scale interpretation'
    samples=[]
    for alpha in (1.,.5,.25,.125):
        candidate={k:before[k]+alpha*(proposal[k]-before[k]) for k in before};agent.modules.load_state_dict(candidate)
        if alpha==1.:assert all(torch.equal(candidate[k],proposal[k]) for k in before)
        k=kl(agent,rows,groups);deltas=[]
        with torch.no_grad():
            for row in rows:
                o=device_obs(row['observation'],agent.device);mu=agent.mean(o,row['z'].to(agent.device),torch.tensor([row['option']],device=agent.device));oldmu=row['old_mu'].to(agent.device)
                sc=torch.as_tensor(eps['qpd_scale'],device=agent.device);clip=eps['original_action_clip']
                delta=(((mu*5).clamp(-clip,clip)-(oldmu*5).clamp(-clip,clip))*sc/5).cpu().numpy();deltas.append(delta)
        d=np.concatenate(deltas)
        samples.append(dict(alpha=alpha,joint_KL=k,inside_original_KL=bool(k['joint']<=.02),mean_qPD_target_delta_RMS_rad=float(np.sqrt(np.mean(d*d))),mean_qPD_target_delta_abs_p95_rad=float(np.quantile(abs(d),.95)),mean_qPD_target_delta_abs_max_rad=float(abs(d).max()),model_weights_digest=tree_digest(agent.modules.state_dict()),physical_joint_motion='NOT_EXECUTED; these are same-state mean PD-target differences'))
    agent.modules.load_state_dict(before);agent.opt.load_state_dict(beforeopt);assert tree_digest(agent.modules.state_dict())==model_before
    result=dict(status='ARCHIVED_FIRST_PROPOSAL_EXACT_SCALE_DIAGNOSTIC_COMPLETE',source_runtime=summary['records'][3]['checkpoint']['runtime'],source_block=4,original_proposal_KL_exact=True,old_joint_logprob_error=err,actor_grad_norm=gn,scales=samples,formal_optimizer_rule_changed=False,new_policy_committed=False,new_training_rows=0,physical_reconstruction_roles='existing full state and12 already archived rows replayed exactly; no candidate physics rollout',joint_KL_limit=.02,std=.01,sealed=False)
    (out/'step-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from research.interaction_fb import benchmark
    from research.interaction_fb.ei_v2_counter_io import install
    install(benchmark);benchmark.study=study;sys.argv=[__file__,'--output',str(a.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
