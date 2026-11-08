"""Retrospective saved-trajectory diagnosis, CPU only, no physical inference/training."""
import json,gzip,math,pathlib,torch
S=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments/native-transfer-20261107')
OLD=S.parent/'native-rolling-h100-20261107'
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
a=read(OLD/'runs/rolling-01/rolling_continue.json.gz') if (OLD/'runs/rolling-01/rolling_continue.json.gz').exists() else None
if a is None:
 manifest=json.loads((S/'runs/data-prepare-01/manifest.json').read_text());src=next(r for r in manifest if r['episode']=='rolling_continue');a=read(pathlib.Path(src['source']))
b=read(S/'runs/diagnostic-01/S4000_rolling_continue.json.gz')
ac={round(x['time']*50):x for x in a['control']};bc={round(x['time']*50):x for x in b['control']}
def angle(x,y):return math.degrees(math.acos(max(-1.,min(1.,sum(float(v)*float(w) for v,w in zip(x,y))/256))))
rows=[]
for k in sorted(set(ac)&set(bc)):
 x,y=ac[k],bc[k];zx=torch.tensor(x['executed_z']).reshape(-1).tolist();zy=torch.tensor(y['executed_z']).reshape(-1).tolist()
 rows.append(dict(time=k/50,z_angle_deg=angle(zx,zy),qpd_max_difference_deg=max(abs(v-w) for v,w in zip(x['q_PD'],y['q_PD']))*180/math.pi))
first=next((rows[i]['time'] for i in range(len(rows)-4) if min(r['z_angle_deg'] for r in rows[i:i+5])>5),None)
queries=torch.load(S/'runs/diagnostic-01/S4000_rolling_continue-actual-queries.pt',map_location='cpu',weights_only=True)
align=json.loads((S/'runs/diagnostic-01/main4000-train-alignment.json').read_text());table=[]
for t in [4.,4.5,5.,5.1,5.2,5.5,6.,6.2,7.,8.1]:
 q=min(queries,key=lambda r:abs(r['time']-t));ca=min(rows,key=lambda r:abs(r['time']-t));sames=[r for r in align if r['role']=='S' and r['episode']=='rolling_continue'];sa=min(sames,key=lambda r:abs(r['time']-t))
 table.append(dict(time=q['time'],own_state_nearest_train_normalized_RMS=q['nearest_main_train_normalized_RMS'],closed_loop_difference=ca,teacher_state_fit=sa))
out=dict(scope='time-matched distinct closed-loop trajectories are descriptive, not same-state causal interventions',first_z_gt5deg_sustained_100ms=first,earliest_control_rows=rows[:80],checkpoints=table,interpretation='Exact Teacher-state fit may be good while own executed history moves out of demonstrated states; nearest normalized RMS is descriptive, not a universal OOD gate. No proof of force information insufficiency.')
(S/'runs/diagnostic-01/trained-S-first-divergence.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(dict(first=first,checkpoints=table),indent=2))
