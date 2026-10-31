"""M2 actual paired contact intervention; no new model or physics execution."""
import gzip,json
from pathlib import Path
import numpy as np
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs')
SRC=P/'m2-unseen-body-20261031A/comparison-attempt-01/outputs';OUT=P/'m3-continuous-body-20261031A/analysis'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text())
def main():
 result=[]
 for ctl in ['native_hold','teacher','rule','distill','alignment']:
  a=read(SRC/f'midpoint__{ctl}.json.gz');b=read(SRC/f'midpoint__{ctl}__free.json.gz');on=a['condition']['onset'];release=a['condition']['release'];times=np.array([r['force_time'] for r in a['physics']]);assert len(a['physics'])==len(b['physics'])
  ar=lambda e,k:np.array([r[k] for r in e['physics']]);w,w0=ar(a,'wrist'),ar(b,'wrist');anchor=ar(a,'anchor');delta=w-w0;before=times<on-1e-8;assert np.max(abs(delta[before]))==0
  phases={}
  for phase,lo,hi in [('move',on,on+4),('hold',on+4,release),('all_contact',on,release),('recovery',release,a['condition']['end'])]:
   m=(times>=lo-1e-8)&(times<hi-1e-8);idx=np.flatnonzero(m)[-1];droot=ar(a,'root')-ar(b,'root');dshoulder=ar(a,'shoulder_xyz')-ar(b,'shoulder_xyz')
   x=dict(contact_increment_wrist_xyz_mean_cm=(delta[m].mean(0)*100).tolist(),contact_increment_wrist_xyz_end_cm=(delta[idx]*100).tolist(),contact_increment_root_xyz_end_cm=(droot[idx]*100).tolist(),contact_increment_shoulder_xyz_end_cm=(dshoulder[idx]*100).tolist(),contact_increment_wrist_relative_root_end_cm=((delta[idx]-droot[idx])*100).tolist(),actual_force_xyz_mean_n=ar(a,'force')[m].mean(0).tolist())
   if phase!='recovery':
    actual=anchor[m]-w[m];virtual=anchor[m]-w0[m];x.update(actual_error_rms_cm=float(np.sqrt(np.mean(np.sum(actual**2,axis=-1)))*100),no_contact_trace_virtual_error_rms_cm=float(np.sqrt(np.mean(np.sum(virtual**2,axis=-1)))*100),error_squared_reduction_cm2=float(np.mean(np.sum(virtual**2-actual**2,axis=-1))*10000))
   phases[phase]=x
  ready=min(a['control'],key=lambda c:abs(c['time']-a['control_ready_s']));start=min(a['control'],key=lambda c:abs(c['time']-on));result.append(dict(controller=ctl,ready_wrist=ready['pre_wrist'],precontact_reconfiguration_xyz_cm=((np.array(start['pre_wrist'])-ready['pre_wrist'])*100).tolist(),phases=phases))
 x=dict(pairs=result,scope='Real contact/no-contact intervention, exact common ready and precontact histories within each controller. Total contact effect includes direct physical compliance and BFM/Student feedback. Virtual no-contact tracking error uses positive branch anchor only for geometry, not a real force-bearing trial. Individual actor factors require separate saved-state decomposition; no force-only causal claim.')
 OUT.mkdir(exist_ok=True);(OUT/'m2-paired-contact-increments.json').write_text(json.dumps(x,indent=2));print(json.dumps(x,indent=2))
if __name__=='__main__':main()
