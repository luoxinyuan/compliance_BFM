"""First command in actual executed units and later physical propagation."""
from pathlib import Path
import sys,gzip,pickle,json,numpy as np
s=Path(sys.argv[1]);out=[]
def read(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)
for p in sorted((s/'d1').glob('*-U.gz')):
    u=read(p);cs=u['contract'];scale=np.array(cs['q_PD_scale']).reshape(-1);cap=cs['clip']
    for n in ['B0','T']:
        v=read(p.with_name(p.name[:-4]+n+'.gz'))
        assert np.array_equal(u['initial_qpos'],v['initial_qpos'])
        a,b=[ep['controls'][0]['action'].flatten() for ep in [u,v]];dp=(np.clip(b*5,-cap,cap)-np.clip(a*5,-cap,cap))*scale/5
        dif=[]
        for ti in [3,19,99,199,399]:
            if min(len(u['physics']),len(v['physics']))<=ti:continue
            f,g=u['physics'][ti],v['physics'][ti]
            dif.append(dict(tick=ti,droot_mm=float(1000*np.linalg.norm(np.subtract(f['qpos'][:2],g['qpos'][:2]))),wrist_mm=float(1000*np.linalg.norm(np.subtract(f['wrist'],g['wrist']))),dforce_N=float(np.linalg.norm(np.subtract(f['force'],g['force'])))))
        out.append(dict(case=p.name[:2],candidate=n,rms_action=float(np.mean((b-a)**2)**.5),rms_target_deg=float(np.rad2deg(np.mean(dp**2)**.5)),peak_target_deg=float(np.rad2deg(np.max(abs(dp)))),peak_joint=cs['dof_names'][np.argmax(abs(dp))],effects=dif))
(s/'d1-physical-effects.json').write_text(json.dumps(out,indent=2))
