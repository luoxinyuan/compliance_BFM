"""Audit saved M9 artifacts, original assets and private/shared source state."""
import gzip,json,hashlib,subprocess
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/m9-contact-locomotion-20261101A';ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ledger=json.loads((R/'compute-ledger.json').read_text());plan=json.loads((ROOT/'research_notes/m9_protocol.json').read_text());out=R/'final-attempt-01/outputs';summary={'episodes':[],'prefix_parity':[]};references={};free={}
 for f in sorted(out.glob('*__*.json.gz')):
  with gzip.open(f,'rt') as h:e=json.load(h)
  c=e['control'];p=e['physics'];q=np.array([r['qpos'] for r in p]);v=np.array([r['qvel'] for r in p]);force=np.array([r['force'] for r in p]);z=np.array([r['executed_z'][0] for r in c]);times=np.array([r['time'] for r in c]);normerr=float(np.abs(np.linalg.norm(z,axis=1)-16).max())
  assert all(np.isfinite(x).all() for x in [q,v,force,z]);assert normerr<1e-4;assert np.linalg.norm(force,axis=1).max()<5.00001;assert np.max(abs(np.diff(times)-.02))<1e-7
  name=e['label'].rsplit('__',1)[0];ready=np.r_[e['initial']['qpos'],e['initial']['qvel']];anchor=np.asarray(e['anchor']);ref=references.setdefault(name,(ready,anchor));assert np.array_equal(ready,ref[0]) and np.array_equal(anchor,ref[1]),'Unpaired ready state or anchor'
  prefix=json.loads((out/(e['label']+'-prefix.json')).read_text());assert max(prefix.values())==0;summary['prefix_parity'].append(e['label'])
  summary['episodes'].append(dict(label=e['label'],complete=e['boundary'] is None,norm_error=normerr,source=str(f)))
  if name=='new_free':free[e['label'].split('__')[1]]=[q,v,force]
 assert len(summary['episodes'])==len(plan['execution']['final']['schedule'])
 summary['new_free_m8_student_max_difference']=max(float(abs(a-b).max()) for a,b in zip(free['m8'],free['student']))
 ck=R/'train2-attempt-01/outputs/projected-z-final.pt';summary['final_checkpoint_sha256']=sha(ck);assert summary['final_checkpoint_sha256']==plan['final_checkpoint_sha256']
 contract=json.loads((out/'action-contract.json').read_text());summary['original_source_pins']={k:sha(ROOT/k)==v for k,v in contract['source_pins'].items()};assert all(summary['original_source_pins'].values())
 model=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model/model.safetensors';summary['original_weight_sha256']=sha(model);assert summary['original_weight_sha256']==contract['model_sha256']
 old=P/'bridge-v2-owned/runs/m8-contact-body-20261101A/train2-attempt-01/outputs/distill-final.pt';summary['m8_pz_sha256']=sha(old);assert summary['m8_pz_sha256']=='74f624a96abc3fd4ea63da101dfb0177bf8f426d6fa4da8717d7b219aaee086a'
 summary['shared_head']=subprocess.check_output(['git','-C',str(P/'repo'),'rev-parse','HEAD'],text=True).strip();summary['shared_status']=subprocess.check_output(['git','-C',str(P/'repo'),'status','--porcelain'],text=True);assert summary['shared_head']=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3' and not summary['shared_status']
 summary['private_changed_files']=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','554324f22ebedcee3498257c4348434c5d937d59','HEAD'],text=True).splitlines();assert all(x.startswith('research_notes/') for x in summary['private_changed_files'])
 summary['all_supervised_finished']=all(a['status']=='finished' and not a.get('remaining_processes') for a in ledger['attempts']);assert summary['all_supervised_finished'];summary['compute_seconds']=sum(a.get('wall_seconds',0) for a in ledger['attempts']);assert summary['compute_seconds']<=plan['resources']['compute_cumulative_wall_seconds']
 (R/'final-audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['episodes','prefix_parity','private_changed_files']},indent=2))
if __name__=='__main__':main()
