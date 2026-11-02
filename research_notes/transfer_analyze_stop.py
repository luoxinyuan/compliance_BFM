from pathlib import Path
import json,gzip,ast,numpy as np
R=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/direct-transfer-20261103');D=R/'A-final-07/job'
refs={'teacher_initial':R/'A-prepare-01/job/initial__source_single.json.gz','teacher_ppo15':R/'A-prepare-01/job/ppo15__source_single.json.gz','teacher_B1':R/'B1-train-02/job/dev-0010/single__mean.json.gz','student_initial':R/'A-final-04/job/initial__student_final_single.json.gz','student_ppo15':R/'A-final-04/job/ppo15__student_final_single.json.gz'}
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
tree=ast.parse((R/'analyze_transfer.py').read_text());base_tree=ast.parse(Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/maintenance/research-handoff-20261102/research_notes/evaluate_clean_saved.py').read_text());scope={'np':np};fn=next(n for n in base_tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics');exec(compile(ast.Module(body=[fn],type_ignores=[]),'metrics','exec'),scope)
result={'protocol':dict(stop_time=8.,deceleration=.6,original_onset=4.25,original_distance=.82,original_ramp=7.8,release=14.05,end=20.05,no_training=True,no_future_partner_input=True),'summaries':json.loads((D/'summaries.json').read_text()),'prefix':{},'physical':{}}
for row in result['summaries']:
 label=row['label'];ep=read(D/(label+'.json.gz'));ep['phases']={'pre':[3,4.25],'move':[4.25,8.],'decelerate':[8.,8.6],'hold':[8.6,14.05],'recover':[14.05,20.05]};result['physical'][label]=scope['metrics'](ep)
 if label in refs:
  a=[r for r in ep['physics'] if r['force_time']<8.];b=[r for r in read(refs[label])['physics'] if r['force_time']<8.];assert len(a)==len(b)
  error=float(np.max(np.abs(np.array([r['qpos'] for r in a])-np.array([r['qpos'] for r in b]))));result['prefix'][label]=dict(samples=len(a),qpos_max_abs_error=error);assert error<1e-8
 print(label,round(row['error_cm'],3),round(row['force_rms'],3),round(row['cap_fraction'],3),'hold',result['physical'][label]['phases']['hold']['signed_error_xyz_cm'])
(R/'analysis/stop-evidence.json').write_text(json.dumps(result,indent=2));print('prefix',result['prefix'])
