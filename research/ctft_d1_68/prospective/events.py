from pathlib import Path
import sys,os,pickle,gzip,json,csv
import numpy as np
s=Path(sys.argv[1]);contract=json.loads((s/'reference/action-contract.json').read_bytes())
names=contract['dof_names']; out=[]
for path in sorted((s/'reference').glob('[0-9][0-9]-*.gz')):
    if '.state.' in path.name:continue
    with gzip.open(path,'rb') as f:x=pickle.load(f)
    met=json.loads(path.with_suffix('.json').read_bytes())
    ph=x['physics'];controls=x['controls'];spec=x['spec'];q0=x['initial_qpos']
    def put(i,kind,seg=None):
        r=ph[i];k=i//4;c=controls[k]
        target=np.array(contract['q_PD_offset']).reshape(-1)+np.clip(5*c['action'].flatten(),-contract['clip'],contract['clip'])*np.array(contract['q_PD_scale']).reshape(-1)/5
        row=dict(parent=spec['parent'],branch=spec['branch'],event=kind,tick=i,time_s=float(r['force_time']-x['start']),active=bool(r['active']),force=np.asarray(r['force']).tolist(),loads=np.asarray(r['foot_normal_force_n']).tolist(),root=np.asarray(r['root']).tolist(),wrist=np.asarray(r['wrist']).tolist(),native_terms=c.get('native_terms'),composite=c.get('reward_components'),span_p95=seg['p95'] if seg else None,span_path=seg['path'] if seg else None)
        for n in ('left_wrist_pitch_joint','right_wrist_pitch_joint','left_ankle_roll_joint','right_ankle_roll_joint'):
            j=names.index(n);row[n]=[float(c['action'].reshape(-1)[j]),float(target[j]),float(r['qpos'][7+j]),float(r['qvel'][6+j])]
        out.append(row)
    for ok in (False,True):
        opts=sorted([v for v in met['stances'] if v['measured'] and (v['p95']<=.05 and v['path']<=.02)==ok],key=lambda z:(z['start'],z['foot']))
        if opts:put(opts[0]['start'],'first_pass_stance' if ok else 'first_fail_stance',opts[0])
    if spec['branch']=='no_force':put(min(100,len(ph)-1),'never_loaded')
    else:
        p=spec['partner'];dec=p['distance_m']/p['plateau_speed_mps']; rel=dec+p['accel_s']+p['hold_s']
        for k,v in [('deceleration',dec),('connected_hold',rel-.5*p['hold_s']),('released',rel+.5)]:
            i=int(np.ceil(v/.005));
            if i<len(ph):put(i,k)
(s/'prospective-events.json').write_text(json.dumps(out,indent=2,allow_nan=False))
