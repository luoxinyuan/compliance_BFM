import csv,hashlib,json
from pathlib import Path
s=Path(__file__).resolve().parent
j=lambda p:json.loads(p.read_bytes())
def table(p,arr):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(arr[0]));w.writeheader();w.writerows(arr)
a=[]
for r in j(s/'comparisons.json'):
    for k in ['a','b']:
        a.append(dict(test=r['test'],case=r['case'],candidate=r['candidate'],arm=k,status=r['status'],**(r[k] or {key:None for key in ['root','wrist','p95','path','slip_cost','aux','contact','force']})))
table(s/'paired-two-seconds.csv',a)
a=[]
for which in ['reference','d2']:
    for p in sorted((s/which).glob('[0-9]*.json')):
        v=j(p)
        for phase,seg in v['stages'].items():
            force=seg['force_xyz_rms'];err=seg.get('lag_xyz_rms') or [None]*3
            a.append(dict(condition=which,parent=v['parent'],branch=v['branch'],phase=phase,ticks=seg['ticks'],force_x_N=force[0],force_y_N=force[1],force_z_N=force[2],lag_x_m=err[0],lag_y_m=err[1],lag_z_m=err[2],foot_switches_active=v['alternating_support_changes'],yaw_deg=v['yaw_abs_deg'],slip=v['slip_result'],stopped=v['stopped'],postrelease_root_m=v['release_root_path'],postrelease_yaw_deg=v['release_yaw'],hard=v['hard']))
table(s/'physical-phases.csv',a)
a=[]
for r in j(s/'prospective-events.json'):
    def packed(v):return json.dumps(v,separators=(',',':'))
    a.append(dict(parent=r['parent'],branch=r['branch'],event=r['event'],tick=r['tick'],time_s=r['time_s'],active=r['active'],force_xyz_N=packed(r['force']),foot_loads_N=packed(r['loads']),native_slip_raw=(r['native_terms'] or {}).get('penalty_slippage'),all_native_raw=packed(r['native_terms']),aux=packed(r['composite']),left_wrist_action_qPD_q_dq=packed(r['left_wrist_pitch_joint']),right_wrist_action_qPD_q_dq=packed(r['right_wrist_pitch_joint']),root=packed(r['root']),wrist=packed(r['wrist'])))
table(s/'events.csv',a)
manifest={}
for p in sorted(s.rglob('*')):
    if p.is_file() and ('/ipc/' not in str(p)) and p.name!='evidence-hashes.json':
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        manifest[str(p.relative_to(s))]=h.hexdigest()
(s/'evidence-hashes.json').write_text(json.dumps(manifest,indent=2))
