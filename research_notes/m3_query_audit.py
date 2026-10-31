"""Independent original expert-query replay on saved M3 states, no model or simulation."""
import copy,gzip,json
from pathlib import Path
import numpy as np,mujoco
from m1_query_audit import replay_query
ROOT=Path(__file__).resolve().parents[1];P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=P/'bridge-v2-owned/runs/m3-continuous-body-20261031A'
def main():
 m=mujoco.MjModel.from_xml_path(str(ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));results=[]
 ledger=json.loads((RUN/'compute-ledger.json').read_text())
 for a in ledger['attempts']:
  if a['stage']!='qualification' or a.get('result_status')!='completed':continue
  out=Path(a['path'])/'outputs';contract=json.loads((out/'action-contract.json').read_text())
  for f in sorted(out.glob('*.json.gz')):
   e=json.loads(gzip.decompress(f.read_bytes()))
   if e['controller'] not in ['teacher','local_teacher']:continue
   label=e['label'];valid=all(c['oracle_query_valid'] for c in e['control'])
   for c in e['control']:
    for k in ['action','q_PD','base_action','base_q_PD']:c[k]=c['oracle_'+k]
    c['teacher_correction']=c['oracle_correction']
   e['controller']='teacher_goal';targets,point=replay_query(e,m,contract);errors={k:float(abs(np.array([c[k] for c in e['control']])-np.array([c[k] for c in targets])).max()) for k in ['action','q_PD']};assert point<1e-10 and max(errors.values())<1e-6
   results.append(dict(label=label,query_within_original_bounds=valid,point_parity_m=point,**errors))
 (RUN/'analysis/query-parity.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
