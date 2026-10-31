"""Saved physical trace only: stage-specific contact and actual directional response."""
import json,gzip,argparse
from pathlib import Path
import numpy as np
from m0_analysis import summarize
from m2_analysis import window
RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m7-continuous-20261101A')
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def summarize_transition(e):
 rows=e['physics'];cs=e['control'];s=e['condition'];tr=s.get('transition');ts=np.array([r['force_time'] for r in rows]);force=np.array([r['force'] for r in rows]);active=np.array([r['active'] for r in rows],bool)
 for c in cs:c['alpha']=c['native_coords'][0] if c.get('native_coords') is not None else 0. if e['controller']=='native_hold' else c['alpha']
 result=summarize(e,{});result['recovery_full']=window(e,s['release'],s['end']);result['transition']={}
 release_force=force[(ts>=s['release'])&(ts<s['end'])]
 result['recovery_full']['force_peak_n']=float(np.linalg.norm(release_force,axis=1).max())
 result['recovery_full']['force_scope']='actual force_time >= release; no last pre-release force included'
 spans=dict(forward=(s['onset'],tr['unload']),gap=(tr['unload'],tr['restart']),reverse=(tr['restart'],s['release']),reverse_move=(tr['restart'],tr['reverse_end']),reverse_hold=(tr['reverse_end'],s['release'])) if tr else {}
 for k,(lo,hi) in spans.items():
  mask=(ts>=lo-1e-8)&(ts<hi-1e-8);track=mask&active;rr=[r for r,v in zip(rows,mask) if v];cc=[c for c in cs if lo-1e-8<=c['time']<hi-1e-8]
  v=window(e,lo,hi);err=np.array([r['anchor'] for r in rows])-np.array([r['wrist'] for r in rows]);f=force[track];fn=np.linalg.norm(f,axis=1)
  v.update(error_cm=float(np.sqrt(np.mean(np.sum(err[track]**2,axis=1)))*100) if len(f) else None,xyz_error_cm=(np.sqrt(np.mean(err[track]**2,axis=0))*100).tolist() if len(f) else None,force_rms_n=float(np.sqrt(np.mean(fn**2))) if len(f) else 0.,force_peak_n=float(fn.max()) if len(f) else 0.,force_xyz_mean=f.mean(0).tolist() if len(f) else [0]*3,force_xyz_rms=np.sqrt(np.mean(f*f,axis=0)).tolist() if len(f) else [0]*3,cap=float((fn>=5-1e-6).mean()) if len(f) else 0.,negative_fx_fraction=float((f[:,0]<-.1).mean()) if len(f) else None,alpha_first=cc[0]['alpha'],alpha_last=cc[-1]['alpha'],alpha_min=min(c['alpha'] for c in cc),alpha_max=max(c['alpha'] for c in cc),first_negative_fx=next((r['force_time'] for r in rr if r['active'] and r['force'][0]<-.1),None))
  for field in ['wrist','shoulder_xyz','root']:
   v[field+'_delta_xyz_cm']=((np.array(rr[-1][field])-rr[0][field])*100).tolist()
  v['wrist_relative_root_delta_xyz_cm']=(np.array(v['wrist_delta_xyz_cm'])-v['root_delta_xyz_cm']).tolist()
  v['waist_delta_deg']=np.rad2deg(np.array(rr[-1]['qpos'][19:22])-rr[0]['qpos'][19:22]).tolist()
  v['torso_rotation_from_initial_max_deg']=float(np.rad2deg(np.arccos(np.clip((np.einsum('nij,ij->n',np.array([r['torso_matrix'] for r in rr]),np.array(e['initial']['torso_matrix']))-1)/2,-1,1))).max())
  v['extra_qPD_rms_deg']={n:float(np.rad2deg(np.sqrt(np.mean(np.array([c['delta_q_PD'] for c in cc])[:,sl]**2)))) for n,sl in [('arm',slice(22,29)),('other',slice(0,22))]}
  v['actual_20ms_step_max_deg']=float(np.rad2deg(abs(np.diff(np.array([c['pre_qpos'][7:] for c in cc]),axis=0))).max())
  result['transition'][k]=v
 result['events']=[dict(time=c['time'],event=c['rule_record']['event'],alpha=c['alpha']) for c in cs if c['rule_record'].get('event')]
 result['projection']=[c['projection'] for c in cs if c.get('projection')]
 result['expert_queries']=dict(valid=sum(c.get('oracle_query_valid',False) for c in cs),frames=len(cs))
 result['initial_sha']=e['ready_physical_state_sha256'];result['z_norm_error']=float(abs(np.linalg.norm(np.array([c['executed_z'][0] for c in cs]),axis=1)-16).max())
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',default=None);a=parser.parse_args();res={}
 for entry in read(RUN/'compute-ledger.json')['attempts']:
  if a.stage and entry['stage']!=a.stage:continue
  for p in sorted((Path(entry['path'])/'outputs').glob('*__*.json.gz')):
   e=read(p);x=summarize_transition(e);x['source']=str(p);key=Path(entry['path']).name+'/'+e['label'];res[key]=x
   print(key,[(k,round(v['error_cm'] or 0,3),round(v['force_rms_n'],3),round(v['alpha_last'],3),round(v['qPD_step_max_deg'],3)) for k,v in x['transition'].items() if k in ['forward','reverse','reverse_hold']],flush=True)
 dest=RUN/'analysis';dest.mkdir(exist_ok=True);(dest/((a.stage or 'all')+'-summary.json')).write_text(json.dumps(res,indent=2,allow_nan=False))
if __name__=='__main__':main()
