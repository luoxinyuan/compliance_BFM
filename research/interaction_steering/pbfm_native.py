"""Isolated inference port of Roboparty P-BFM pages2401ffa1 (CC BY-NC4.0).
Native exported PD/history/depth preserved; local MuJoCo is a separate backend.
No training or import-time simulator; every physical step passes TickBudget.
"""
import json,os,time,math,hashlib
from pathlib import Path
import numpy as np
import mujoco as mj
import onnxruntime as ort

F=np.float32

def torque(target,q,dq,c):
    lim=np.asarray(c['effortLimit']);vel=np.asarray(c['velocityLimit']);v=np.clip(dq,-2*vel,2*vel)
    return np.clip(np.asarray(c['kp'])*(target-q)-np.asarray(c['kd'])*dq,np.maximum(-lim,lim*(-1-v/vel)),np.minimum(lim,lim*(1-v/vel)))

def project_gravity(quat):
    w,x,y,z=np.asarray(quat,dtype=F).astype(float)/np.linalg.norm(np.asarray(quat,dtype=F).astype(float))
    return np.array([2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1],dtype=F)

def yaw(q):
    q=np.asarray(q,F).astype(float);w,x,y,z=q/np.linalg.norm(q);return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def process_depth(raw):
    a=np.asarray(raw,dtype=F).reshape(18,32);a=np.where(np.isfinite(a)&(a>=0),np.minimum(a,F(2.5)),F(2.5))
    e=np.array([math.exp(-.5),1,math.exp(-.5)],dtype=F);e=e/F(F(e[0]+e[1])+e[2]);w=np.outer(e,e).astype(F);p=np.pad(a,1,mode='reflect');out=np.zeros_like(a)
    for i in range(3):
        for j in range(3):out=(out+(p[i:i+18,j:j+32]*w[i,j]).astype(F)).astype(F)
    return (np.rint((np.clip(out,0,F(2.5))/F(2.5)*F(255)).astype(F))/255).astype(F)

class Feed:
    def __init__(self,c,z):
        self.c=c;self.z=np.array(z,F);self.last=np.zeros(29,F);self.hist={k:np.zeros((4,n),F) for k,n in [('actions',29),('base_ang_vel',3),('dof_pos',29),('dof_vel',29),('projected_gravity',3)]};self.depth=None
    def build(self,q,dq,quat,omega,depth,speed):
        pos=(np.array(q,F).astype(float)-np.asarray(self.c['defaultJointPosition'])).astype(F);vel=np.array(dq,F);ang=(np.array(omega,F)*F(.25)).astype(F);g=project_gravity(quat);y=yaw(quat)
        if self.depth is None:self.depth=np.repeat(depth[None],37,axis=0)
        self.depth=np.concatenate([depth[None],self.depth[:-1]],axis=0)
        obs={'state':np.concatenate([pos,vel,g,ang])[None], 'last_action':self.last[None].copy(),'history_actor':np.concatenate([self.hist[k].ravel() for k in self.c['historyKeys']])[None], 'heading':np.array([[1-math.cos(-y),math.sin(-y)]],F),'velocity_command':np.array([[speed*math.cos(y),-speed*math.sin(y),1]],F),'depth_image':self.depth[self.c['depth']['sampledAges']][None], 'z':self.z[None]}
        for k,v in dict(actions=self.last,base_ang_vel=ang,dof_pos=pos,dof_vel=vel,projected_gravity=g).items():self.hist[k]=np.concatenate([v[None],self.hist[k][:-1]],axis=0)
        return obs
    def action(self,raw):
        self.last=np.clip(np.array(raw,F)*5,-5,5).astype(F)
        # JS computes target in double then stores Float32.
        return (np.asarray(self.c['defaultJointPosition'])+np.asarray(self.c['actionScale'])*self.last.astype(float)).astype(F)

class Depth:
    def __init__(self,m,d):
        self.m=m;self.d=d;self.body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'torso_link');assert self.body>=0
        self.mount=np.array([.0487988662332928,.01,.4378029937970051]);q=np.array([.25558171377860706,-.6579550475696607,.6623183568544073,-.25121840449386046]);rot=np.empty(9);mj.mju_quat2Mat(rot,q/np.linalg.norm(q));rot=rot.reshape(3,3)
        xx,yy=np.meshgrid(np.arange(32),np.arange(18));fx=64/(2*np.tan(np.deg2rad(89.51)/2));fy=36/(2*np.tan(np.deg2rad(58.29)/2));r=np.stack([(xx+16+.5-32)/fx,(yy+18+.5-18)/fy,np.ones((18,32))],-1).reshape(-1,3);self.cos=1/np.linalg.norm(r,axis=1);self.rays=(r*self.cos[:,None])@rot.T
        head=mj.mj_name2id(m,mj.mjtObj.mjOBJ_MESH,'head_link');self.hide=[i for i in range(m.ngeom) if m.geom_bodyid[i]==self.body and m.geom_dataid[i]==head and m.geom_type[i]==mj.mjtGeom.mjGEOM_MESH];self.groups=np.array([1,0,1,0,0,1],np.uint8)
        reach=2.5/self.cos.min();self.bounds=[]
        for i in range(m.ngeom):
            if m.body_weldid[m.geom_bodyid[i]]!=0 or m.geom_group[i] not in [0,5]:continue
            if m.geom_type[i]==mj.mjtGeom.mjGEOM_BOX:radius=np.linalg.norm(m.geom_size[i])
            elif m.geom_type[i]==mj.mjtGeom.mjGEOM_HFIELD:
                z=m.hfield_size[m.geom_dataid[i]];radius=np.linalg.norm([z[0],z[1],max(z[2:])])
            else:continue
            self.bounds.append((i,(reach+radius+1e-6)**2))
    def capture(self):
        m,d=self.m,self.d;R=d.xmat[self.body].reshape(3,3);origin=d.xpos[self.body]+R@self.mount;dirs=self.rays@R.T;changed=list(self.hide)
        changed += [i for i,b in self.bounds if np.sum((d.geom_xpos[i]-origin)**2)>b];old=m.geom_group[changed].copy();g=np.empty(1,np.int32);raw=np.empty(576,F)
        try:
            m.geom_group[changed]=4
            for i,v in enumerate(dirs):
                dist=mj.mj_ray(m,d,origin,v,self.groups,1,-1,g);raw[i]=min(2.5,dist*self.cos[i]) if dist>=0 and np.isfinite(dist) else 2.5
        finally:m.geom_group[changed]=old
        return process_depth(raw)

class TickBudget:
    def __init__(self,stage):
        self.stage=Path(stage);self.file=self.stage/'physical-counter.json';self.start=json.loads((self.stage/'start.json').read_text());self.n=json.loads(self.file.read_text())['reserved_ticks'] if self.file.exists() else 0
    def reserve(self,k):
        if self.n+k>self.start['physics_tick_cap']:raise RuntimeError('TOTAL_PHYSICS_BUDGET')
        if time.time()-self.start['epoch']>self.start['cap_job_s']-300:raise RuntimeError('TIME_BUDGET')
        if os.statvfs(self.stage).f_bavail*os.statvfs(self.stage).f_frsize<100*1024**3:raise RuntimeError('STORAGE_GUARD')
        self.n+=k;tmp=self.file.with_suffix('.tmp');tmp.write_text(json.dumps({'reserved_ticks':self.n,'failed_reserved_calls_count':True}));os.replace(tmp,self.file)
    def step(self,m,d):mj.mj_step(m,d)

class Native:
    def __init__(self,stage):
        self.stage=Path(stage);self.asset=self.stage/'evidence/interactive/pbfm';self.c=json.loads((self.asset/'config.json').read_text());self.ctx=json.loads((self.asset/'contexts.json').read_text())['presets'][0]
        assert self.ctx['id']=='forward08-arms-spread';assert hashlib.sha256((self.asset/'policy.onnx').read_bytes()).hexdigest()==self.c['modelSha256']
        self.m=mj.MjModel.from_xml_path(str(self.asset/'robot/scene.xml'));self.d=mj.MjData(self.m);m=self.m;assert (m.nq,m.nv,m.nu)==(36,35,29)
        ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n) for n in self.c['jointNames']];assert min(ids)>0;self.ids=np.array(ids);self.qadr=m.jnt_qposadr[ids];self.vadr=m.jnt_dofadr[ids];self.aids=np.array([np.where(m.actuator_trnid[:,0]==i)[0].item() for i in ids]);assert np.allclose(m.jnt_range[ids],self.c['jointRanges'],atol=1e-7,rtol=0);assert np.allclose(m.dof_armature[self.vadr],self.c['armature'],atol=1e-8,rtol=0)
        o=ort.SessionOptions();o.intra_op_num_threads=4;o.inter_op_num_threads=1;self.session=ort.InferenceSession(str(self.asset/'policy.onnx'),sess_options=o,providers=['CPUExecutionProvider']);self.depth=Depth(m,self.d);self.budget=TickBudget(stage);self.footids=[i for i in range(m.nbody) if 'ankle_roll_link' in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i) or '')]
    def reset(self):
        mj.mj_resetData(self.m,self.d);self.d.qpos[:3]=[-2.5,-15,.8];self.d.qpos[3:7]=[1,0,0,0];self.d.qpos[self.qadr]=self.c['defaultJointPosition'];mj.mj_forward(self.m,self.d);self.feed=Feed(self.c,self.ctx['z'])
    def contacts(self):
        m,d=self.m,self.d;rows=[]
        for idx in range(d.ncon):
            c=d.contact[idx];ids=m.geom_bodyid[[c.geom1,c.geom2]]
            for bid in ids:
                if bid not in self.footids:continue
                force=np.empty(6);mj.mj_contactForce(m,d,idx,force)
                jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mj.mj_jac(m,d,jp,jr,c.pos,int(bid));v=jp@d.qvel;normal=c.frame[:3];speed=np.linalg.norm(v-normal*np.dot(normal,v))
                rows.append(dict(body=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(bid)),normal_N=float(abs(force[0])),speed_mps=float(speed),position=c.pos.tolist()))
        return rows
    def trial(self,speed,repeat,out):
        self.reset();m,d=self.m,self.d;physics=[];controls=[];boundary=None;start=time.time();out=Path(out);out.mkdir(exist_ok=False,parents=True)
        for k in range(600):
            t=k*.02
            u=0. if t<2 or t>=8 else speed if 3<=t<7 else speed*((t-2)**2*(3-2*(t-2))) if t<3 else speed*(1-(t-7)**2*(3-2*(t-7)))
            dep=self.depth.capture();obs=self.feed.build(d.qpos[self.qadr],d.qvel[self.vadr],d.qpos[3:7],d.qvel[3:6],dep,u)
            raw,est=self.session.run(None,obs);target=self.feed.action(raw[0]);self.budget.reserve(4)
            controls.append(dict(time=t,u=u,obs={key:value.copy() for key,value in obs.items()},raw=raw.copy(),estimated_velocity=est.copy(),scaled_action=self.feed.last.copy(),q_PD=target.copy()))
            for _ in range(4):
                d.ctrl[self.aids]=torque(target,d.qpos[self.qadr],d.qvel[self.vadr],self.c);d.xfrc_applied[:]=0;d.qfrc_applied[:]=0;self.budget.step(m,d)
                contacts=self.contacts();q=d.qpos[self.qadr];ranges=m.jnt_range[self.ids];violation=np.maximum(ranges[:,0]-q,q-ranges[:,1]);worst=int(np.argmax(violation))
                physics.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),q_PD=target.copy(),torque=d.ctrl.copy(),yaw=yaw(d.qpos[3:7]),contacts=contacts,range_excess=float(violation[worst]),worst_joint=self.c['jointNames'][worst],external_force=0.))
                if not np.isfinite(d.qpos).all():boundary='nonfinite'
                elif violation[worst]>.02:boundary='joint-range+.02:'+self.c['jointNames'][worst]
                elif d.qpos[2]<.45:boundary='baseheight<.45'
                if boundary:break
            mj.mj_forward(m,d)
            if boundary:break
        import torch
        artifact=out/'trace.pt';torch.save(dict(physics=physics,controls=controls,context=self.ctx,config=self.c,backend=mj.__version__),artifact)
        phases={}
        for name,lo,hi in [('entry',0,3),('moving',3,7),('deceleration',7,8),('stop',8,12),('final',10,12)]:
            rows=[x for x in physics if lo<=x['time']<hi+1e-6]
            if rows:
                v=np.array([x['qvel'][:3] for x in rows]);sl=[c['speed_mps'] for x in rows for c in x['contacts'] if c['normal_N']>1e-6]
                phases[name]=dict(ticks=len(rows),mean_vx=float(v[:,0].mean()),horizontal_speed_rms=float(np.sqrt(np.mean(np.sum(v[:,:2]**2,axis=1)))),slip_peak=max(sl,default=0.),yaw_end_deg=float(np.rad2deg(rows[-1]['yaw'])),root_dx=float(rows[-1]['qpos'][0]-rows[0]['qpos'][0]))
        slips=[(c['speed_mps'],c['normal_N'],x['time']) for x in physics for c in x['contacts'] if c['normal_N']>1e-6]
        result=dict(speed=speed,repeat=repeat,seconds=time.time()-start,actual_ticks=len(physics),reserved_total=self.budget.n,boundary=boundary,phases=phases,root_dx=float(d.qpos[0]+2.5),root_dy=float(d.qpos[1]+15),max_slip=max(slips,default=(0,0,0)),trace_sha=hashlib.sha256(artifact.read_bytes()).hexdigest())
        (out/'result.json').write_text(json.dumps(result,indent=2));return result
