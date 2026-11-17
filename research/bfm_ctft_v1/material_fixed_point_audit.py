"""Fixed sphere-surface material points over actual contact-valid intervals.

Current model has four spherical sole collision geoms per foot. A point is
tracked in body material coordinates while its sphere has an active original
ground constraint and the point remains in the ground intersection cap. Every
recorded active contact spawns a point; no adverse stance frames are trimmed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from source_evidence_readonly import read
from quality import stance_segments


class MaterialTracks:
    def __init__(self):self.live={};self.closed=[]

    def step(self,tick,geometry,poses,post_poses,ground_z):
        # geometry maps geomID ->(bodyID, actual bottom surface local point).
        for gid,(body,seed) in geometry.items():
            key=(gid,tick)
            self.live[key]=dict(geom=gid,body=body,birth_tick=tick,last_tick=tick-1,
                point=seed.copy(),path=0.)
        finish=[]
        for key,p in self.live.items():
            gid,body=p['geom'],p['body']
            pos,R=poses[body];after_pos,after_R=post_poses[body]
            before=pos+R@p['point']
            # Only machine-scale tolerance for evaluating the analytic plane.
            # No physical penetration/contact threshold is added or raised.
            eps=64*np.finfo(float).eps*max(1.,float(np.max(abs(before))))
            if gid not in geometry or before[2]>ground_z+eps:
                finish.append(key);continue
            after=after_pos+after_R@p['point']
            p['path']+=float(np.linalg.norm((after-before)[:2]));p['last_tick']=tick
        for key in finish:self.close(key)

    def close(self,key):
        p=self.live.pop(key)
        if p['last_tick']>=p['birth_tick']:
            self.closed.append(dict(geom=p['geom'],body=p['body'],birth_tick=p['birth_tick'],
                end_tick_exclusive=p['last_tick']+1,valid_duration_s=(p['last_tick']-p['birth_tick']+1)*.005,
                material_point_body=p['point'].tolist(),cumulative_tangent_path_m=p['path']))

    def end(self):
        for key in list(self.live):self.close(key)
        return self.closed


def fixtures():
    body=1;gid=2;r=.005;center=np.array([0.,0.,r-.0001]);tracks=MaterialTracks()
    for i in range(60):
        before=center+np.array([.1*i*.005,0,0]);after=before+np.array([.1*.005,0,0])
        tracks.step(i,{gid:(body,np.array([0.,0.,-r]))},{body:(before,np.eye(3))},{body:(after,np.eye(3))},0.)
    rows=tracks.end();maximum=max(x['cumulative_tangent_path_m'] for x in rows)
    assert abs(maximum-.03)<1e-12 and maximum>.02
    return dict(analytic_sliding_01mps_03s_path_m=maximum,full_interval_included=True,updates=0,physics_ticks=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    model=mujoco.MjModel.from_xml_path(str(a.runtime_root/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
    d=mujoco.MjData(model);weight=float(model.body_mass.sum()*np.linalg.norm(model.opt.gravity))
    summary=json.loads((a.source/'progress.json').read_text());results=[]
    for info in summary:
        path=a.source/info['evidence_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['evidence_sha256']
        x=read(path);ph=x['physics'];bodies=set(c['body'] for row in ph for c in row['ctft_material_contacts'])
        pre=[];post=[];previous=x['states'][0]['mjdata_arrays']['qpos']
        for row in ph:
            d.qpos[:]=previous;mujoco.mj_kinematics(model,d)
            pre.append({b:(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for b in bodies})
            d.qpos[:]=row['qpos'];mujoco.mj_kinematics(model,d)
            post.append({b:(d.xpos[b].copy(),d.xmat[b].reshape(3,3).copy()) for b in bodies});previous=row['qpos']
        records=[];unsupported=[]
        for segment in stance_segments([r['foot_normal_force_n'] for r in ph],weight):
            if not segment['qualified']:continue
            tracker=MaterialTracks();velocities=[];missing=[]
            for i in range(segment['start'],segment['end_exclusive']):
                contacts=[c for c in ph[i]['ctft_material_contacts'] if c['foot']==segment['foot'] and c['normal_n']>0]
                geometry={};speed=[]
                for c in contacts:
                    pair=c['geom_pair'];gid=next(g for g in pair if int(model.geom_bodyid[g])==c['body'])
                    ground=next(g for g in pair if int(model.geom_bodyid[g])==0)
                    if int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_SPHERE) or int(model.geom_type[ground])!=int(mujoco.mjtGeom.mjGEOM_PLANE):
                        unsupported.append(dict(tick=i,geom=gid,ground=ground));continue
                    assert np.array_equal(model.geom_quat[ground],np.array([1.,0,0,0]))
                    ground_z=float(model.geom_pos[ground,2]);assert ground_z==0.
                    b=c['body'];pos,R=pre[i][b]
                    radius=float(model.geom_size[gid,0]);seed=model.geom_pos[gid]-R.T@np.array([0.,0.,radius])
                    surface=pos+R@seed
                    eps=64*np.finfo(float).eps*max(1.,float(np.max(abs(surface))))
                    if surface[2]>ground_z+eps:missing.append(dict(tick=i,geom=gid,surface_height=float(surface[2]-ground_z)));continue
                    geometry[gid]=(b,seed)
                    # Real post-step velocity of this same body-fixed point.
                    d.qpos[:]=ph[i]['qpos'];d.qvel[:]=ph[i]['qvel'];mujoco.mj_kinematics(model,d);mujoco.mj_comPos(model,d)
                    world=d.xpos[b]+d.xmat[b].reshape(3,3)@seed;jac=np.zeros((3,model.nv));mujoco.mj_jac(model,d,jac,None,world,b)
                    speed.append(float(np.linalg.norm((jac@d.qvel)[:2])))
                if not geometry:missing.append(dict(tick=i,no_valid_contact_material=True))
                tracker.step(i,geometry,pre[i],post[i],0.)
                velocities.append(max(speed) if speed else None)
            tracks=tracker.end();qualified=not missing and not unsupported and len(tracks)>0
            maximum=max((r['cumulative_tangent_path_m'] for r in tracks),default=None)
            p95=float(np.percentile(velocities,95)) if all(v is not None for v in velocities) else None
            records.append(dict(**segment,tracking_measured=qualified,missing=missing,fixed_material_tracks=tracks,
                fixed_point_max_cumulative_path_m=maximum,material_velocity_p95_mps=p95,
                bounds_pass=(p95<=.05 and maximum<=.02) if qualified else None,
                final_unloading_step_included=True))
        results.append(dict(case=info['case'],ancestor=info['ancestor'],source_sha256=info['evidence_sha256'],
            source_complete=info['complete'],unsupported_collision_geometries=unsupported,segments=records))
        (a.output/'progress.json').write_text(json.dumps(results,indent=2)+'\n');del x,pre,post
    result=dict(status='FIXED_SPHERE_MATERIAL_CONTACT_INTERVALS_MEASURED',fixtures=fixtures(),sources=results,
        contact_validity='Same original positive-normal sphere-ground constraint; same body-fixed point remains in actual analytic ground intersection cap. End at material-point lift or loss of sphere contact; entire final loaded integration step retained.',
        scope='No force threshold raised, no foot/ankle displacement proxy, all contact births and full stance frames included; unsupported/missing intervals UNMEASURED.',
        updates=0,new_physics_ticks=0,wall_s=time.monotonic()-start)
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],wall_s=result['wall_s'],sources=len(results))),flush=True)


if __name__=='__main__':main()
