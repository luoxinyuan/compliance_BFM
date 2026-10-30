"""Independent saved-record contract checks; no model inference or simulation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from contact_gate import ContactGate
from alpha_memory import ForceMemory


def audit(run):
    read=lambda p:json.loads(Path(p).read_text())
    s=read(run/'analysis/alpha-results.json');protocol=read(Path(__file__).with_name('alpha_student_protocol.json'))
    gp=read(Path(__file__).with_name('contact_gate_protocol.json'))
    contract=read(run/'physics-attempt-01/outputs/action-contract.json')
    scale=np.asarray(contract['q_PD_scale']);offset=np.asarray(contract['q_PD_offset']);clip=contract['clip']
    records=[]
    for row in s['manifest']:
        if row['reused_prior_control']:continue
        p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        ep=read(p);cs=ep['control'];ph=ep['physics'];assert len(ph)==len(cs)*4
        previous=[ep['initial']]+ph[3::4][:-1]
        lag=max(float(abs(np.array(c['measured_force_xyz'])-prev['force']).max()) for c,prev in zip(cs,previous))
        pose=max(float(abs(np.array(c['pre_qpos'])-prev['qpos']).max()) for c,prev in zip(cs,previous))
        assert lag==0 and pose==0
        action=np.asarray([c['action'] for c in cs]);q=np.asarray([c['q_PD'] for c in cs])
        qe=offset+np.clip(action*5,-clip,clip)*scale/5
        pd_error=float(abs(q-qe).max());assert pd_error<2e-6
        norms=np.linalg.norm(np.asarray([c['executed_z'][0] for c in cs]),axis=1)
        assert abs(norms-16).max()<1e-4 and all(0<=c['alpha']<=1 for c in cs)
        shadow_error=None
        if ep['controller'] in ('A','B'):
            gate=ContactGate(gp['gate']);shadow_error=max(abs(gate.step(c['measured_force_xyz'])['alpha']-c['shadow_alpha']) for c in cs)
            assert shadow_error==0
        records.append(dict(label=ep['label'],force_lag_max=lag,pre_qpos_lag_max=pose,PD_contract_max_rad=pd_error,shadow_replay_max=shadow_error,z_norm_max_error=float(abs(norms-16).max())))
    import torch
    out=run/'train-attempt-01/outputs';data=torch.load(out/'training-dataset.pt',weights_only=True,map_location='cpu')
    start=0
    for name in protocol['training_conditions']:
        ep=read(run.parent/protocol['teacher_source']/(name+'__contact.json'));mem=ForceMemory(320)
        f=np.asarray([mem.append(c['measured_force_xyz']) for c in ep['control']],dtype=np.float32).reshape(len(ep['control']),-1)*np.float32(.2) # Exact CUDA reciprocal-multiply order
        assert np.array_equal(f,data['x'][start:start+len(f),465:].numpy());start+=len(f)
    assert start==1560
    a=torch.load(out/'alpha_A_final.pt',weights_only=True,map_location='cpu');b=torch.load(out/'alpha_B_final.pt',weights_only=True,map_location='cpu')
    assert a['updates']==b['updates']==2000 and a['initial_state_sha256']==b['initial_state_sha256']
    for k in ('center','scale'):assert torch.equal(a['state_dict'][k],b['state_dict'][k])
    fit=s['fit'];assert fit['full_A']['batch_sequence_sha256']==fit['full_B']['batch_sequence_sha256']
    result=dict(records=records,all_passed=True,training_force_history_exact=True,training_samples=1560,paired_initialization_batches_normalizers_verified=True,
        no_model_inference=True,no_physics=True,no_training=True)
    (run/'analysis/independent-record-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(all_passed=True,new_branches=len(records),training_samples=start)))

if __name__=='__main__':audit(Path(sys.argv[1]))
