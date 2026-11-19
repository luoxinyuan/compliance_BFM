import os,csv,gzip,pickle,json,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
s=Path.cwd();m=mj.MjModel.from_xml_path(str(Path(os.environ['CTFT_PINNED_RUNTIME'])/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));d=mj.MjData(m)
def hand(q):
    d.qpos[:]=q;mj.mj_kinematics(m,d)
    pt=lambda k:d.xpos[m.body(k).id]
    wr=pt('right_wrist_yaw_link');sh=pt('right_shoulder_pitch_link');R=d.xmat[m.body('torso_link').id].reshape(3,3)
    return [100* (R.T@(wr-sh))[0],100*(wr-pt('left_wrist_yaw_link'))[2],np.rad2deg(q[m.jnt_qposadr[m.joint('right_elbow_joint').id]]),100*q[0],100*wr[0]]
rows=[];meta={};initial=None
for name,folder in [('U','../ctft_quiet_01/resumed'),('keep','evaluation')]:
    eps={}
    for b in ['no_force','continue','stop']:
        p=s/folder/f'{name}-{b}.gz'
        with gzip.open(p,'rb') as f:x=pickle.load(f)
        eps[b]=x
        if initial is None:initial=x['initial_qpos']
        assert np.array_equal(x['initial_qpos'],initial)
        ph=x['physics'];arr=lambda k:np.asarray([v[k] for v in ph]);ff=arr('force');t=arr('force_time')-x['start'];load=np.linalg.norm(ff,axis=1)>=1;dom=(ff[:,0]>0)&(ff[:,0]>=abs(ff[:,1]))&(ff[:,0]>=abs(ff[:,2]))
        if b=='no_force':assert not np.any(ff)
        j=json.loads(p.with_suffix('.json').read_text());r=dict(model=name,branch=b,source_sha=hashlib.sha256(p.read_bytes()).hexdigest(),**{k:j[k] for k in ['hard','slip_result','stopped','force_rms','lag_rms','root_path','yaw_abs_deg','release_root_path','release_yaw','alternating_support_changes','root_forward','wrist_progress_ratio']})
        for label,mask in [('early',np.arange(len(t))<400),('moving',t<x['spec']['partner']['distance_m']/.08)]:
            sel=mask&load;r[label+'_loaded_ticks']=int(sel.sum());r[label+'_frac']=float(dom[sel].mean()) if sel.any() else None;r[label+'_fwdN']=float(ff[mask,0].mean());r[label+'_upN']=float(ff[mask,2].mean());r[label+'_peakN']=float(np.linalg.norm(ff[mask],axis=1).max())
        for i,label in [(None,'init'),(99,'half'),(399,'two')]:
            vals=hand(initial if i is None else ph[i]['qpos']) if i is None or i<len(ph) else [None]*5
            for k,v in zip(['reach','heightdiff','elbow','rootX','wristX'],vals):r[label+'_'+k]=v
        rows.append(r)
    n=min(300,len(eps['stop']['physics']),len(eps['continue']['physics']))
    for k in ['qpos','qvel','torque','force','foot_normal_force_n','anchor']:
        assert np.array_equal([v[k] for v in eps['stop']['physics'][:n]],[v[k] for v in eps['continue']['physics'][:n]])
    meta[name]=dict(common_prefix_ticks=n)
with open('results.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
Path('verification.json').write_text(json.dumps(meta))
for r in rows:print(r['model'],r['branch'],{k:round(r[k],3) if isinstance(r[k],float) else r[k] for k in ['early_frac','moving_frac','early_fwdN','early_upN','moving_peakN','two_reach','two_elbow','two_rootX','root_path','force_rms','lag_rms','stopped','release_root_path','release_yaw','hard','slip_result']})
