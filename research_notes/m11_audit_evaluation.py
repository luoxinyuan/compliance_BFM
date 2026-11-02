"""Check paired initial states, prescribed partner paths, legal z and actual force."""
import argparse,gzip,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();out=[]
 for f in sorted(a.folder.glob('*__prior.json.gz')):
  name=f.name.split('__')[0];es={};files={}
  for mode in ['prior','m10_expert','learned']:
   q=a.folder/(name+'__'+mode+'.json.gz');es[mode]=json.loads(gzip.decompress(q.read_bytes()));files[mode]=hashlib.sha256(q.read_bytes()).hexdigest()
  ref=es['prior'];rows={}
  for mode,e in es.items():
   z=np.array([c['executed_z'][0] for c in e['control']]);base=np.array([c['prior_z'][0] for c in e['control']]);F=np.array([r['force'] for r in e['physics']]);P=np.array([r['anchor'] for r in e['physics']]);R=np.array([r['anchor'] for r in ref['physics']]);prefix=e['condition'].get('prefix_until',0)
   prefixdiff=[float(np.max(np.abs(np.array(c['pre_qpos'])-r['pre_qpos']))) for c,r in zip(e['control'],ref['control']) if c['time']<prefix-1e-8]
   same_init=max(float(np.max(np.abs(np.array(e['initial'][k])-ref['initial'][k]))) for k in ['qpos','qvel'])
   row=dict(same_condition=e['condition']==ref['condition'],initial_max_abs_difference=same_init,partner_anchor_max_difference=float(np.max(np.abs(P-R))) if P.shape==R.shape else None,legal_z_max_norm_error=float(np.abs(np.linalg.norm(z,axis=1)-16).max()),force_peak=float(np.linalg.norm(F,axis=1).max()),finite=bool(np.isfinite(z).all() and np.isfinite(F).all()),boundary=e['boundary'],physics_steps=len(e['physics']),controls=len(e['control']),end_time=e['physics'][-1]['time'],scheduled_end=e['condition']['end'],prefix_state_max_difference=max(prefixdiff,default=0),executed_vs_prior_angle_max_deg=float(np.degrees(np.arccos(np.clip(np.sum(z*base,axis=1)/256,-1,1))).max()),raw_residual_norm_max=max(c['residual_uncapped_norm'] for c in e['control']))
   assert row['same_condition'] and same_init<1e-10
   assert row['legal_z_max_norm_error']<1e-4 and row['force_peak']<5.00001 and row['finite']
   assert row['partner_anchor_max_difference'] is not None and row['partner_anchor_max_difference']<1e-10
   rows[mode]=row
  out.append(dict(condition=name,files_sha256=files,controllers=rows))
 assert len(out)==6
 (a.folder/'paired-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
