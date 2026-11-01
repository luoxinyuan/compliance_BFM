"""Saved-evidence audit, no model execution or source mutation."""
import json,gzip,hashlib,subprocess
from pathlib import Path
import numpy as np
from m9c_stage import RUN,P

def main():
 ledger=json.loads((RUN/'compute-ledger.json').read_text());items=[];pairs={};free={}
 for a in ledger['attempts']:
  out=Path(a['path'])/'outputs'
  for f in sorted(out.glob('*.json.gz')):
   e=json.loads(gzip.decompress(f.read_bytes()))
   if not e.get('control'):continue
   cs=e['control'];rs=e['physics']
   if a['stage']=='final' and e['label'].startswith('no_contact__'):free[e['label']]=np.array([r['qpos'] for r in rs])
   z=np.array([c['executed_z'][0] for c in cs]);force=np.array([r['force'] for r in rs]);applied=np.array([r['applied_contact_wrench'][:3] for r in rs]);time=np.array([r['time'] for r in rs]);normerr=float(abs(np.linalg.norm(z,axis=1)-16).max());forceerr=float(abs(force-applied).max());assert normerr<1e-4 and forceerr<1e-10;assert np.allclose(np.diff(time),.005,atol=1e-8)
   rec=dict(stage=a['stage'],label=e['label'],path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),physics_steps=len(rs),controls=len(cs),complete=e['boundary'] is None,boundary=e['boundary'],z_norm_max_error=normerr,force_wrench_max_error=forceerr,actual_joint_limit_max=max(r['joint_limit_excess_max_rad'] for r in rs),torque_saturation_max=max(r['torque_saturation'] for r in rs),nonfoot_contacts_max=max(r['undesired_ground_contacts'] for r in rs),action_clip_max=max(c['action_saturation_fraction'] for c in cs));items.append(rec)
   if not e['label'].startswith('train-'):
    key=json.dumps(e['condition'],sort_keys=True);ref=pairs.setdefault(key,dict(anchor=np.asarray(e['anchor']),qpos=np.asarray(cs[0]['pre_qpos']),obs={k:np.asarray(v) for k,v in cs[0]['bfm_obs'].items()},members=[]));assert np.allclose(ref['anchor'],e['anchor'],atol=1e-12);assert np.allclose(ref['qpos'],cs[0]['pre_qpos'],atol=1e-12)
    for k,v in ref['obs'].items():assert np.allclose(v,cs[0]['bfm_obs'][k],atol=1e-12)
    ref['members'].append(a['stage']+'/'+e['label'])
 completed=[];failures=[]
 for a in ledger['attempts']:
  out=Path(a['path'])/'outputs';r=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
  if a.get('exit_code')==0 and a.get('result_status')=='completed':
   if not a['stage'].startswith('videos'):assert r.get('original_bfm_unchanged') is True
   completed.append(a['stage'])
  elif a['status']=='finished':failures.append(dict(stage=a['stage'],path=a['path'],failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else None))
 free_comparison={k:float(abs(v-free['no_contact__standing']).max()) for k,v in free.items()} if free else {}
 shared=P/'repo';git=lambda *x:subprocess.check_output(['git','-C',str(shared),*x],text=True).strip();result=dict(no_contact_qpos_max_difference=free_comparison,episodes=items,paired_conditions=[v['members'] for v in pairs.values()],completed_stages=completed,failed_attempts=failures,supervised_seconds=sum(a.get('wall_seconds',0) for a in ledger['attempts']),peak_rss=max(a['peak_rss_bytes'] for a in ledger['attempts']),peak_compute_vram_mib=max(a['peak_gpu_mib'] for a in ledger['attempts']),remaining_owned=[a for a in ledger['attempts'] if a.get('remaining_processes') or a['status']=='running'],shared_head=git('rev-parse','HEAD'),shared_status=git('status','--porcelain'));assert result['shared_head']=='6023c6e257fab8fe58d13d9207fb5e85c32ee9c3' and result['shared_status']=='';(RUN/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['episodes','paired_conditions']},indent=2));print('episodes',len(items),'paired',len(pairs))
if __name__=='__main__':main()
