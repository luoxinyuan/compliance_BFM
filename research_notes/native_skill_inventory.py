"""Read-only native BFM/source/motion inventory. No actor, rollout, FK, or training.

Only the explicitly specified JSON output is written. Target joint deltas describe
LaFAN references, not commands or physically executed motions.
"""
import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
import joblib
import numpy as np

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
MODEL = P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view'
DATA = Path('/mnt/data-oss/users/yi.gao/compliance_BFM/data')
COMMON = P/'bridge-v2-owned/runs/issue21-direction-20261030T020000Z/attempt-02/outputs/common-state.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def inventory(root):
    cfg = json.loads((MODEL/'checkpoint/model/config.json').read_text())
    source_names = [
        'humanoidverse/agents/fb/model.py', 'humanoidverse/agents/fb_cpr_aux/model.py',
        'humanoidverse/goal_inference.py', 'humanoidverse/tracking_inference.py',
        'humanoidverse/reward_inference.py', 'humanoidverse/utils/helpers.py',
        'humanoidverse/agents/envs/humanoidverse_isaac.py',
        'humanoidverse/envs/g1_env_helper/bench/reward_eval_hv.py',
        'humanoidverse/force_compliance/environment.py',
        'humanoidverse/data/robots/g1/goal_frames_lafan29dof.json']
    expected = {'data/lafan_29dof.pkl':'f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11',
                'model/model.safetensors':'33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'}
    assets = {}
    for key, path in [('data/lafan_29dof.pkl',DATA/'lafan_29dof.pkl'),
                      ('model/model.safetensors',MODEL/'checkpoint/model/model.safetensors')]:
        digest = sha(path); assert digest == expected[key]
        assets[key] = dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest)
    assert cfg['name']=='FBcprAuxModel' and cfg['archi']['z_dim']==256 and cfg['archi']['norm_z']
    motion = joblib.load(DATA/'lafan_29dof.pkl')
    keys = list(motion)
    goals = json.loads((root/'humanoidverse/data/robots/g1/goal_frames_lafan29dof.json').read_text())
    standing = np.array(json.loads(COMMON.read_text())['env_bookkeeping']['last_qpos'])[7:]
    motion_meta = []
    for i,(name,v) in enumerate(motion.items()):
        dof_from_pose = v['pose_aa'][:,1:].sum(-1)
        error = float(abs(dof_from_pose-v['dof']).max())
        assert error < 1e-5
        motion_meta.append(dict(id=i,name=name,frames=len(v['dof']),fps=v['fps'],
            duration_s=(len(v['dof'])-1)/v['fps'],dof_shape=list(v['dof'].shape),
            pose_aa_to_dof_error_rad=error, has_policy_action='action' in v))
    targets=[]
    for goal in goals:
        assert keys[goal['motion_id']]==goal['motion_name']
        v=motion[goal['motion_name']];fps=v['fps'];duration=(len(v['dof'])-1)/fps
        for frame in goal['frames']:
            # The native helper resamples motion at env.dt=.02; JSON indices are
            # NOT raw 30Hz pickle frame indices. Only joint interpolation here.
            t=frame*.02; assert t<duration
            f=t*fps;lo=int(np.floor(f));hi=min(lo+1,len(v['dof'])-1);w=f-lo
            q=(1-w)*v['dof'][lo]+w*v['dof'][hi];delta=np.rad2deg(q-standing)
            rms=lambda sl:float(np.sqrt(np.mean(delta[sl]**2)))
            targets.append(dict(name=goal['motion_name']+'_'+str(frame),motion_id=goal['motion_id'],
                resampled_50Hz_index=frame,motion_time_s=t,raw_30Hz_index=f,
                right_arm_joint_delta_RMS_deg=rms(slice(22,29)),left_arm_joint_delta_RMS_deg=rms(slice(15,22)),
                waist_joint_delta_RMS_deg=rms(slice(12,15)),legs_joint_delta_RMS_deg=rms(slice(0,12)),
                max_joint_delta_deg=float(abs(delta).max()),target_dof_rad=q.tolist(),
                meaning='REFERENCE target delta only, not inferred BFM command or physical capability'))
    reward_tree=ast.parse((root/'humanoidverse/reward_inference.py').read_text())
    tasks=next(ast.literal_eval(n.value) for n in ast.walk(reward_tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='tasks' for t in n.targets))
    paths=['checkpoint/buffers/train','checkpoint/buffers/train_reduced',
           'reward_inference/reward_locomotion.pkl','goal_inference/goal_reaching.pkl','tracking_inference']
    stat={n:dict(exists=(MODEL/n).exists(),resolved=str((MODEL/n).resolve())) for n in paths}
    found=[]
    for parent in [P/'bridge-v2-owned/runs',P/'outputs',P/'runs']:
        if not parent.exists():continue
        for d,dirs,files in os.walk(parent):
            dirs[:]=[n for n in dirs if n not in ['cache','source','.git']]
            for name in files:
                if name in ['reward_locomotion.pkl','goal_reaching.pkl'] or name.startswith('zs_'):
                    found.append(str(Path(d)/name))
    return dict(source_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        source_hashes={n:sha(root/n) for n in source_names},assets=assets,
        checkpoint=dict(name=cfg['name'],z_dim=cfg['archi']['z_dim'],norm_z=cfg['archi']['norm_z'],
            seq_length=cfg['seq_length'],actor_std=cfg['actor_std'],amp=cfg['amp'],
            actor_keys=cfg['archi']['actor']['input_filter']['key'],
            backward_keys=cfg['archi']['b']['input_filter']['key']),
        asset_paths=stat,found_named_inference_exports=found,reward_script_tasks=tasks,
        motion_metadata=motion_meta,goal_targets=targets,
        checks=dict(asset_sha_match=True,all_goal_names_match_ids=True,all_goal_indices_in_bounds=True,
            pose_aa_dof_consistent=True,physical_rollouts=0,model_forward_calls=0,policy_updates=0),
        limitations='No reference trajectory or goal name establishes an executable stable skill. No MuJoCo stepping, actor inference, new latent, reward relabeling, or source modification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'Preserve previous inventory'
    result=inventory(args.root);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(checks=result['checks'],checkpoint=result['checkpoint'],motions=len(result['motion_metadata']),
        goals=len(result['goal_targets']),reward_task_names=len(result['reward_script_tasks']),named_exports=result['found_named_inference_exports'])))
