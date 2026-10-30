"""Final saved-artifact/source/resource audit; no model execution or physics."""
import ast, hashlib, json, math, subprocess
from pathlib import Path

P = Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
W = P / 'bridge-v2-owned/trees/issue-21-alpha-interruption/repo'
R = P / 'bridge-v2-owned/runs/issue21-alpha-interruption-20261030A'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def git(w, *args):
    return subprocess.check_output(['git', '-C', str(w), *args], text=True).strip()
def read(p): return json.loads(Path(p).read_text())

protocol = read(W / 'research_notes/interruption_protocol.json')
summary = read(R / 'analysis/interruption-results.json')
ledger = read(R / 'compute-ledger.json')
train = P / 'bridge-v2-owned/runs' / protocol['training_output']
assert len(summary['episodes']) == 8
assert all(v['boundary'] is None for v in summary['episodes'].values())
assert len(ledger['attempts']) == 2
stages = []
for a in ledger['attempts']:
    result = read(Path(a['path']) / 'outputs/result.json')
    assert a['result_status'] == 'completed' and a['exit_code'] == 0
    assert not a['remaining_processes'] and a['resource_boundary'] is None
    assert not Path('/proc', str(a['pid'])).exists()
    assert result['original_bfm_unchanged'] and result['student_optimizer_updates'] == 0
    stages.append({k:a[k] for k in ['stage','commit','pid','wall_seconds','peak_rss_bytes','peak_gpu_mib','result_status','remaining_processes']})
hashes = {m: sha(train / f'alpha_{m}_final.pt') for m in ('A','B')}
assert hashes == protocol['checkpoints']
assert sha(W / 'research_notes/contact_gate.py') == protocol['teacher_gate_sha256']
contract = read(R / 'physics-attempt-01/outputs/action-contract.json')
assert all(sha(W / f) == h for f,h in contract['source_pins'].items())
model = P / 'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors'
assert sha(model) == protocol['model_sha256']
shared_status = git(P / 'repo','status','--porcelain')
shared_head = git(P / 'repo','rev-parse','HEAD')
assert not shared_status and shared_head == '6023c6e257fab8fe58d13d9207fb5e85c32ee9c3'
changes = git(W,'diff','--name-status',protocol['base_commit']).splitlines()
assert all(line.startswith('A\tresearch_notes/') for line in changes)
assert all(sha(v['path']) == v['sha256'] for v in summary['manifest'])
continuity = {}
for m in summary['manifest']:
    ep = read(m['path']); ph = ep['physics']; cs = ep['control']
    start, end = (ep['condition'][k] for k in ('disconnect_start','disconnect_end'))
    near = lambda t: min(ph, key=lambda row: abs(row['force_time']-t))
    v = {str(t): near(t)['force'] for t in [start-.005,start,end-.005,end]}
    v['force_5ms_step_max_n'] = max(math.sqrt(sum((a-b)**2 for a,b in zip(y['force'],z['force']))) for y,z in zip(ph[1:],ph[:-1]))
    v['alpha_20ms_step_max'] = max(abs(y['alpha']-z['alpha']) for y,z in zip(cs[1:],cs[:-1]))
    v['torque_5ms_joint_step_max_Nm'] = max(max(abs(a-b) for a,b in zip(y['torque'],z['torque'])) for y,z in zip(ph[1:],ph[:-1]))
    continuity[m['label']] = v
assert continuity == read(R / 'analysis/transition-continuity.json')
for f in W.glob('research_notes/interruption*.py'): ast.parse(f.read_text())
diag = read(R / 'diagnosis-attempt-01/outputs/frozen-diagnosis.json')
assert diag['policy_updates'] == 0 and diag['new_physical_branches'] == 0 and diag['no_model_changes']
audit = dict(status='passed', shared_head=shared_head, shared_status=shared_status,
    private_head=git(W,'rev-parse','HEAD'), private_status=git(W,'status','--porcelain'),
    only_new_research_note_files=True, changes=changes, frozen_sha256=hashes,
    model_sha256=protocol['model_sha256'], original_source_pins_unchanged=True,
    original_gate_sha256=protocol['teacher_gate_sha256'], all_saved_episode_hashes_match=True,
    independent_physics_audit=summary['independent_physics_audit'], stages=stages,
    cumulative_model_wall_seconds=sum(a['wall_seconds'] for a in stages),
    all_owned_processes_exited=True, training_updates=0,
    gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
(R / 'analysis/final-audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
