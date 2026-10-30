"""Verify cached controls and sealed joint margins, without new physical rollout."""
import hashlib
import numpy as np
import torch
from wholebody_physics import read,save,RUNS

SOURCE=RUNS/'issue21-goalcalibration-20261030A/attempt-01/outputs'

def build_and_check(env,model,contract,common,out,protocol):
    for name,h in protocol['pins'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==h
    old=read(SOURCE/'anchored.json')
    assert old['same_initial_sha256']==hashlib.sha256(__import__('json').dumps(common,sort_keys=True).encode()).hexdigest()
    dev=env.standing.device
    goals=[torch.tensor(r['z'],device=dev,dtype=torch.float32) for r in read(SOURCE/'goal-path.json')['waypoints']]
    scale=torch.tensor(contract['q_PD_scale'],device=dev);offset=torch.tensor(contract['q_PD_offset'],device=dev)
    qtarget=lambda a:offset+(a*5).clamp(-contract['clip'],contract['clip'])*scale/5
    errors=[];envelope_clearance=[];clip_margin=[]
    caps=np.array(protocol['accumulated']['per_joint_caps_rad']);rows=[]
    # Same saved B observations: unchanged BFM command reproduced before new physics.
    for i,c in enumerate(old['control']):
        if c['tether_time']>=8-1e-8:continue
        obs={k:torch.tensor(v,device=dev,dtype=torch.float32) for k,v in c['bfm_obs'].items()}
        z=torch.tensor(c['executed_z'],device=dev,dtype=torch.float32)
        a=model.act(obs,z,mean=True);q=qtarget(a)[0].cpu().numpy()
        correction=np.array(old['decisions'][i//5]['correction'])
        expected=np.array(c['q_PD']);expected[22:]-=correction
        errors.append(float(abs(q-expected).max()))
        limits=np.array([x['limits_rad'] for x in protocol['margins']])
        envelope_clearance.append(float(np.minimum(q[22:]-caps-limits[:,0],limits[:,1]-q[22:]-caps).min()))
        clip_margin.append(float(np.min(contract['clip']/5-np.abs(a[0].cpu().numpy()[22:])-caps/scale[0,22:].cpu().numpy())))
        rows.append(dict(time=c['tether_time'],bfm_q_PD=q.tolist(),old_teacher_q_PD=correction.tolist()))
    assert max(errors)<2e-6, 'Original B model/normalizer/observation output changed'
    permitted=min(envelope_clearance)>.02 and min(clip_margin)>0
    summary=dict(physical_allowed=permitted,archived_B_forward_max_error_rad=max(errors),
        worst_box_envelope_joint_margin_rad=min(envelope_clearance),worst_box_action_clip_margin=min(clip_margin),
        preflight_note='Checks saved B states only; does not certify new closed-loop stability',baseline_reused=True)
    save(out,'precheck-decision.json',summary);save(out,'cached-forward-check.json',dict(summary=summary,controls=rows))
    for name in ['goal-path.json','geometry.json']:save(out,name,read(SOURCE/name))
    save(out,'reused-inputs.json',{str(SOURCE/k):v for k,v in protocol['pins'].items()})
    return goals,summary
