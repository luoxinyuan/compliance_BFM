"""Materialize the signed finite v2 protocol; no sealed records or performance selection."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def prepare():
    locks=STAGE/'locks'
    signed=ROOT/'research_notes/CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md'
    assert sha(signed)=='69f10a9143932351c740d5208d561f81f9aeeafe69da49c3efc9da773ac5c3b9'
    qual=json.loads((STAGE/'runs/P0-corrected-arm-sources-01/source-qualification.json').read_text());assert qual['status']=='CORRECTED_ARM_SOURCE_EXECUTION_QUALIFIED'
    manifests={};streams={};counts={}
    for split in ('train','dev'):
        data=json.loads((locks/f'expanded-corrected-{split}.json').read_text());gs=[]
        for p in data['parents']:
            key=p['execution_source_parent_id'];p['source_parent_id']=key
            for i,s in enumerate(p['specs']):
                s['original_source_parent_id']=p['original_source_parent_id'];s['source_parent_id']=key
                s['split']=split
                stream=dict(parent_id=p['parent_id'],sibling_index=i,mode=p['goal_stream'],change_after_handoff_s=1.5 if '_TO_' in p['goal_stream'] else None)
                if '_TO_' in stream['mode']:
                    q=s['source_until']+1.5
                    windows=(max(0.,min(q,s['release'])-s['onset']),max(0.,s['release']-max(q,s['onset'])))
                    assert min(windows)>=.5-1e-7,(split,p['parent_id'],windows)
                    stream['active_window_s']=windows
                gs.append(stream)
        manifests[split]=data;streams[split]=gs;counts[split]=dict(parents=len(data['parents']),episodes=len(gs))
        write(locks/f'execution-{split}.json',data);write(locks/f'goal-stream-{split}.json',gs)
    assert not ({p['source_parent_id'] for p in manifests['train']['parents']}&{p['source_parent_id'] for p in manifests['dev']['parents']})
    train=manifests['train']['parents'];rng=np.random.default_rng(660302)
    # Stratified first mandatory coverage, then fixed seeded order; no performance/readout input.
    first=[]
    for family in ('none','zero','near','mid','stop'):
        first.append(next(i for i,p in enumerate(train) if p['family']==family))
    rest=[i for i in range(len(train)) if i not in first];rng.shuffle(rest);order=first+rest
    ordered=[train[i] for i in order];write(locks/'training-order.json',dict(seed=660302,parent_indices=order,parent_ids=[p['parent_id'] for p in ordered],rule='first stable none/zero/near/mid/stop for required P2 coverage; remaining parent groups seed-shuffled, siblings contiguous; subsequent sweeps same fixed order'))
    write(locks/'training-parents-ordered.json',dict(parents=ordered))
    qualification=[]
    for family in ('none','zero','near','mid','stop'):qualification.extend([p for p in train if p['family']==family][:2])
    write(locks/'qualification-parents.json',dict(parents=qualification,selection='stable first2 each family, no performance selection'))
    protocol=dict(version='CONTACT-GOAL-EI-v2-execution-lock',contract_sha256=sha(signed),source_fix_authorization='Yi: 这种你自己的实现问题可以自己修正后继续',seed=660301,training_order_seed=660302,arm_names=['E','I'],transitions=dict(P2=12000,P3=20000,each_arm=32000),scratch_each_arm=4096,scratch_total=8192,std=.01,relative_noise_clip=None,raw_clamp=[-1,1],option_dwell_slots=5,command_Hz=10,native_Hz=50,physics_Hz=200,gamma20=.99**.2,lambda20=.95**.2,gamma100=.99,joint_KL=.02,PPO_clip=.2,epochs=2,minibatch_decision_segments=64,rollout_native_rows=1024,actor_lr=1e-5,value_lr=3e-4,adam_betas=[.9,.999],adam_eps=1e-8,weight_decay=0.,max_grad_norm=1.,categorical_entropy=0.,auxiliary_updates=4,auxiliary_batch=64,auxiliary_coefficient=.1,auxiliary_recent_native_rows=8192,retain_lambda_initial=0.,retain_lambda_rate=.05,retain_lambda_max=3.,sentinel_native_nodes=[0,4000,8000,12000,20000,28000,32000],counts=counts,qualification_parent_count=len(qualification),sealed_read=False,asset_manifest_sha256='ea9c3323e21abb55d300835d79c3b4981c4a47fa59036f39b9dfc25df17e1fd7')
    protocol['manifest_hashes']={f:sha(locks/f) for f in ['execution-train.json','execution-dev.json','training-order.json','goal-stream-train.json','goal-stream-dev.json','qualification-parents.json']}
    protocol['administrative_chunk_note']='A 1024-row block can contain 206 genuine decision fragments when it begins inside a five-row command, rather than the draft nominal ceil(1024/5)=205. Keep every actual conditional draw, exact times and real macro slots; never drop, duplicate or relabel a draw to fit a buffer. This is a storage bound correction only; joint probability, controls, 1024 rows and GAE unchanged.'
    write(locks/'protocol.json',protocol)
    schema=dict(version='contact-goal-ei-v2-causal',source='current original checkpoint observation and qualified dual_track_task.contact_vector; no new learned normalizers',fields=[
      dict(name='state,last_action,history_actor',producer='original environment real robot observation, actual executed actions and BFM history',roles=['actor','selector','predictor','value'],normalization='original producer scales; original frozen normalizer inside BFM actor'),
      dict(name='contact[0:6]',producer='known wrist installation',frame='mount local point SI m',scale=1,roles=['actor','selector','predictor','value'],missing='known mounting only; never active label'),
      dict(name='contact[6:9],contact[22:97]',producer='ideal measured current and25 real 50Hz wrist-force samples',frame='current pelvis',units='N',scale='divide5N, not10N/K',roles=['actor','selector','predictor','value'],missing='left zero-pad only absent pre-reset history'),
      dict(name='contact[15:18]',producer='own wrist kinematics minus own root',frame='current pelvis',units='m',scale=1,roles=['actor','selector','predictor','value']),
      dict(name='_previous_z,_first_interpolated_z',producer='own actual executed z and causal command interpolation memory',roles=['actor','selector','predictor','value_previous_only'],scale='previous divide16 in deploy; first valid native256D raw norm16'),
      dict(name='_goal_y,_goal_mask,_goal_mode',producer='public current Goal; HOLD uses ideal own odometry and wrist kinematics, FOLLOW current measured force direction',roles=['actor','selector','predictor','value'],units='H0 result SI slots',scale='H0 fixed24D SCALES; mask/mode0or1',missing='unspecified result slots0 withmask0; no hidden task/phase'),
      dict(name='_option_memory',producer='own selected option onehot and age/5',roles=['selector','actor_condition'],scale='own controller memory, never global clock'),
      dict(name='privileged_state,contact[9:15],contact[18:22]',producer='current robot training privilege and current partner error/velocity/actual active',roles=['value','reward','evaluation'],denied=['actor','selector','predictor','publicGoalIssuer']),
      dict(name='_target_z,_slot',producer='own actual command target/interpolation slot',roles=['execution','audit'],denied=['state_value_baseline'],reason='do not condition state V on just-selected action'),
      dict(name='future_y',producer='same-arm actual future physical rows with explicit horizon/episode masks',roles=['auxiliary_training_only'],denied=['online_actor','online_selector','online_predictor_input']),
    ],ideal_force_not_hardware_verified=True,ideal_odometry_assumption=True,global_clock_partner_future_taskID_hidden_active_to_policy=False)
    write(locks/'contact_schema.json',schema)
    return dict(protocol_sha256=sha(locks/'protocol.json'),schema_sha256=sha(locks/'contact_schema.json'),counts=counts,source_qualified=12,qualification_parents=len(qualification),formal_updates=0)
if __name__=='__main__':print(json.dumps(prepare(),indent=2))
