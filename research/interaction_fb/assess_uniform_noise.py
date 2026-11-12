import json,math,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--baseline-stage',type=Path,required=True);args=parser.parse_args()
S=args.stage;O=args.baseline_stage
names={'old_sample':O/'runs/sample-r3a/probe.json','mean':O/'runs/mean-r3/probe.json','uniform':S/'runs/uniform-r1/probe.json'}
r={}
for n,p in names.items():
 j=json.loads(p.read_text());ph=j['physics'];matched=[v for v in ph if v['time']<=7.80000001]
 assert len(matched)==500,(n,len(matched))
 def stats(rows):
  er=np.array([v['error'] for v in rows]);fo=np.array([v['force'] for v in rows]);sl=np.array([v['stage_loaded_slip_mps'] for v in rows]);
  peak=max(rows,key=lambda p:p['stage_loaded_slip_mps']);loaded=[c for c in peak['stage_foot_contacts'] if c['normal_N']>1e-6];cp=max(loaded,key=lambda c:c['tangent_speed_mps']);
  return dict(slip_peak_time=peak['time'],slip_peak_contact=cp,slip_exceed_ticks=sum(p['stage_loaded_slip_mps']>.45 for p in rows),ticks=len(rows),start=rows[0]['time'],end=rows[-1]['time'],error_rms_cm=float(np.sqrt(np.mean(er**2))*100),force_rms_N=float(np.sqrt(np.mean(np.sum(fo**2,axis=-1)))),slip_peak_mps=float(max(sl)),slip_p90_mps=float(np.quantile(sl,.9)),joint_excess_max_rad=max(v['joint_limit_excess_max_rad'] for v in rows),cap_fraction=float(np.mean(np.linalg.norm(fo,axis=-1)>=9.999)),right_ankle_q_max_rad=max(v['qpos'][18] for v in rows))
 r[n]=dict(status=j['status'],matched=stats(matched),full=stats(ph),trace_sha256=j['trace_sha256'],agent=j['frozen_agent_digest'],rng=j['restored_rng_digest'])
assert len({x['agent'] for x in r.values()})==1
assert len({x['rng'] for x in r.values()})==1
checks={k:r['uniform']['matched'][k]<=r['old_sample']['matched'][k] for k in ('error_rms_cm','force_rms_N','slip_peak_mps')}
checks['complete_3s_no_hard_boundary']=r['uniform']['status']=='completed_3s' and r['uniform']['full']['ticks']==600 and r['uniform']['full']['joint_excess_max_rad']<=.02
out=dict(protocol='INTERACTION_PILOT_NOISE_REVISION.md',checks=checks,first_gate_pass=all(checks.values()),cases=r)
(S/'uniform-gate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
