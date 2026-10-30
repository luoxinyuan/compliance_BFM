"""Read-only audit of current model identity and bounded project asset locations.

No model loading, pickle deserialization, simulator, GPU, or asset downloading.
Run with the project Python; stdout is the JSON evidence record.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
OSS = Path('/mnt/data-oss/users/yi.gao/compliance_BFM')
MODEL = OSS / 'pretrained/official/new_model_for_training_code_inference'
REPO = P / 'bridge-v2-owned/trees/issue-21-native-interface/repo'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def scan(root):
    result = {'root': str(root), 'max_depth': 7, 'files_examined': 0,
              'candidate_matches': [], 'errors': [], 'pruned': []}
    def error(e):
        result['errors'].append(str(e))
    for path, dirs, files in os.walk(root, followlinks=False, onerror=error):
        if len(Path(path).relative_to(root).parts) >= 7:
            result['pruned'].extend(str(Path(path)/d) for d in dirs)
            dirs[:] = []
        result['files_examined'] += len(files)
        for name in files:
            if name in {'buffer.hdf5', 'reward_locomotion.pkl', 'goal_reaching.pkl'} or name.startswith('zs_'):
                result['candidate_matches'].append(str(Path(path)/name))
    return result


paths = [MODEL / rel for rel in ('config.json', 'checkpoint/model/config.json',
         'checkpoint/model/init_kwargs.json', 'checkpoint/model/model.safetensors')]
source_paths = ['humanoidverse/reward_inference.py', 'humanoidverse/agents/fb/model.py',
                'humanoidverse/agents/buffers/trajectory.py',
                'humanoidverse/agents/buffers/transition.py',
                'humanoidverse/envs/g1_env_helper/bench/reward_eval_hv.py',
                'humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml']
contract_path = P / 'bridge-v2-owned/runs/issue21-causallatent-20261030A/attempt-01/outputs/action-contract.json'
result = {
    'kind': 'read_only_asset_identity_audit',
    'model_files': {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths},
    'scans': [scan(OSS), scan(P / '.cache/huggingface')],
    'shared_checkout': {'head': git(P/'repo', 'rev-parse', 'HEAD'), 'status': git(P/'repo', 'status', '--porcelain')},
    'source_checkout': {'head': git(REPO, 'rev-parse', 'HEAD'), 'status': git(REPO, 'status', '--porcelain')},
    'sources': {r: digest(REPO/r) for r in source_paths},
    'action_contract_path': str(contract_path),
    'action_contract_sha256': digest(contract_path),
    'action_contract': json.loads(contract_path.read_text()),
    'simulation_steps': 0, 'model_forwards': 0, 'training_updates': 0,
}
print(json.dumps(result, indent=2))
