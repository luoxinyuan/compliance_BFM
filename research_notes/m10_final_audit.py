"""Read-only scientific provenance checks; writes only this milestone's audit record."""
from pathlib import Path
import json,gzip,hashlib,subprocess
import numpy as np
from m10_stage import RUN,P
read=lambda p:json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ledger=read(RUN/'compute-ledger.json');assert all(a['status']=='finished' and not a.get('resource_boundary') and not a.get('remaining_processes') for a in ledger['attempts'])
 rows=[];znorm=0.;controls=0;physics=0
 for a in ledger['attempts']:
  out=Path(a['path'])/'outputs';result=read(out/'result.json');assert result['status']=='completed'
  if a['stage']!='videos':assert result['original_bfm_unchanged']
  for p in out.glob('*__*.json.gz'):
   ep=read(p);z=np.array([c['executed_z'][0] for c in ep['control']]);assert np.isfinite(z).all();znorm=max(znorm,float(abs(np.linalg.norm(z,axis=1)-16).max()));controls+=len(z);physics+=len(ep['physics']);rows.append(dict(stage=a['stage'],label=ep['label'],source=str(p),sha256=sha(p),boundary=ep['boundary']))
 assert znorm<1e-4 and len(rows)==33 and all(r['boundary'] is None for r in rows)
 F=RUN/'final-attempt-01/outputs';a=read(F/'no_contact__candidate.json.gz');b=read(F/'no_contact__standing.json.gz');free={k:float(abs(np.array([r[k] for r in a['physics']])-np.array([r[k] for r in b['physics']])).max()) for k in ['qpos','qvel','force','wrist']};assert max(free.values())==0
 ref0=read(RUN/'qualification-attempt-01/outputs/references.json');ref1=read(RUN/'source_control-attempt-01/outputs/references.json');assert all(ref0['variants'][k]==ref1['variants'][k] for k in ['arm','task']) and ref0['body_goals']==ref1['body_goals']
 plan=read(Path(__file__).with_name('m10_protocol.json'));checkpoint=RUN/'train1-attempt-01/outputs/student-final.pt';assert sha(checkpoint)==plan['final_candidate_sha256'];train=read(checkpoint.with_name('training.json'))
 held=[plan['execution']['final']['conditions'][k] for k in ['new_straight','new_body']];assert all(m.get('condition') not in held for m in train['manifest'])
 source=P/'repo';shared_head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();shared_status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True);assert shared_head=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3' and not shared_status
 result=dict(physical_branches=len(rows),all_complete=True,physics_steps_200hz=physics,control_steps_50hz=controls,simulated_seconds=physics/200,legal_latent_norm_max_error=znorm,no_contact_exact_parity=free,unchanged_reference_variants=True,final_checkpoint_sha256=sha(checkpoint),final_data_groups=len(train['manifest']),final_data_samples=sum(m['samples'] for m in train['manifest']),supervised_updates=3600,RL_samples=0,held_out_conditions_not_in_training=True,shared_head=shared_head,shared_clean=True,compute_wall_seconds=sum(a['wall_seconds'] for a in ledger['attempts']),peak_owned_RSS_bytes=max(a['peak_rss_bytes'] for a in ledger['attempts']),peak_compute_app_VRAM_MiB=max(a['peak_gpu_mib'] for a in ledger['attempts']),graphics_note='EGL graphics allocations are not reported by compute-app query; GPU free memory returned to baseline after rendering',output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),attempts=[{k:a.get(k) for k in ['stage','commit','wall_seconds','result_status','remaining_processes']} for a in ledger['attempts']],episodes=rows)
 (RUN/'final-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','attempts']},indent=2))
if __name__=='__main__':main()
