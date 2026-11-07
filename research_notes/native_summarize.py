"""Small public summaries only; raw physics/observations/checkpoints stay private."""
import json,math,csv,argparse,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'public';out.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def save(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
q=load(a.root/'qualification-03/summaries.json');save('native-qualification.json',[{k:r[k] for k in ['label','error_cm','force_rms','root_max_displacement','boundary','phase_metrics','geometry_contract','qpd_max_deg']} for r in q])
prob=load(a.root/'nvidia-probability.json');save('probability.json',{'device':prob['device'],'torch':prob['torch'],'initial_checks':[{'input_scale':r['input_scale'],'joint_logprob_max':r['batch_one_vs_batched_joint_logprob_max'],'forced_rejection_KL':r['forced_rejection_KL'],'rollback_parameters_equal':r['rollback_parameters_equal']} for r in prob['rows']],'fitted_real_observation_checks':load(a.root/'native-real-probability.json')})
curves=[];models=[]
for arm in ['ordinary','guided','coordinate']:
 root=a.root/f'ppo-{arm}';launch=load(root/'launch.json');sel=load(root/'selection.json');result=load(root/'result.json');models.append({'arm':arm,'runtime':launch['commit'],'selected':sel,'result':result,'initialization':load(root/'resume.json')})
 for line in (root/'training.jsonl').read_text().splitlines():
  r=json.loads(line);u=r['optimization'];curves.append({'arm':arm,'iteration':r['iteration'],'rl_decisions':r['rl_decisions'],'force_rms':r['force_rms'],'cap_fraction':r['cap_fraction'],'reward':r['reward'],'raw_mean_norm':r['raw_mean_norm'],'raw_marginal_std_mean':r['raw_marginal_std_mean'],'sampling_seconds':r['sampling_seconds'],'optimization_seconds':r['optimization_seconds'],'coverage':r['coverage'],'optimizer_updates':u['optimizer_updates'],'pre_logprob_max_error':u['pre_logprob_max_error'],'joint_KL':u['epochs'][-1]['joint_gaussian_KL'],'clip_fraction':u['epochs'][-1]['clip_fraction'],'critic_EV':u['updated_value_EV'],'gradient':u['gradients'],'reward_terms':r['reward_terms'],'elapsed_s':r['elapsed_s']})
save('learning-curves.json',curves);save('training-identities.json',models)
# Detailed per-phase physical metrics contain no raw states or model parameters.
rs=load(a.root/'final-01/summaries.json');save('final-physical.json',rs);save('models.json',load(a.root/'final-01/models.json'));save('common-prefix.json',load(a.root/'final-01/common-prefix.json'));save('exploration-proposal-versions.json',load(a.root/'native-direction-versions.json'));save('source-coverage-v1.json',load(a.root/'exploration-source-coverage.json'))
flat=[]
for r in rs:
 flat.append({k:r.get(k) for k in ['model','case','label','error_cm','force_rms','cap_fraction','root_max_displacement','qpd_max_deg','actual_torso_turn_deg','actual_pelvis_turn_deg','work_j','boundary']})
with (out/'final-matrix.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
ledger=load(a.root/'ledger.json');save('resource-ledger.json',ledger)
print(json.dumps({'models':models,'episodes':len(rs),'job_seconds_completed':sum(j.get('elapsed_s',0) for j in ledger['jobs']),'RL_decisions':sum(m['result']['decisions'] for m in models)},indent=2))

import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).with_name('native_metrics.py')),'--root',str(a.root)],check=True)

# Same deterministic raw control function can have different full checkpoint SHA
# because critic/covariance/optimizer differ. Do not count this as new learning.
import torch
identities=load(a.root/'final-01/models.json')
for name,v in identities.items():
 ck=torch.load(v['path'],map_location='cpu',weights_only=True);h=hashlib.sha256()
 st=ck['state_dict']
 if 'base_weight' in st:
  st=dict(st);M=st['raw_L']/.008;st['actor.4.weight']=st['base_weight']+M@st['actor.4.weight'];st['actor.4.bias']=st['base_bias']+M@st['actor.4.bias']
 for key,value in sorted(st.items()):
  if key.startswith('actor.') or key in ['center','scale']:
   h.update(key.encode());h.update(value.detach().contiguous().numpy().tobytes())
 v['deterministic_raw_function_sha256']=h.hexdigest()
save('models.json',identities)
# Supplemental pre-guided probability diagnostic was independent and short.
ledger['supplemental_diagnostics']=[load(a.root/n) for n in ['native-direction-versions.json','update-geometry-diagnostic.json','native-coordinate-check.json']]
ledger['completed_job_seconds_plus_supplemental']=sum(j.get('elapsed_s',0) for j in ledger['jobs'])+sum(v['seconds'] for v in ledger['supplemental_diagnostics'])
save('resource-ledger.json',ledger)

save('update-geometry-diagnostic.json',load(a.root/'update-geometry-diagnostic.json'));save('coordinate-check.json',load(a.root/'native-coordinate-check.json'))
