"""Offline rendering of actual saved physics qpos; no policy/simulation steps."""
import os,json,gzip,hashlib,time
from pathlib import Path
import numpy as np
import mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
ROOT=Path(__file__).resolve().parents[1]
RUN=P/'bridge-v2-owned/runs/m9c-expert-20261101'
SRC=P/'bridge-v2-owned/runs/m9-feedback-continuation-20261101B/final-attempt-01/outputs'
ITEMS=[('held_straight__expert','Original motion + handwritten feedback expert'),('held_straight__student','M9B shared Student (own command memory)'),('held_left__student','Limited LEFT positive - not general turning'),('held_right__student','RIGHT counterexample: position follows, heading wrong'),('down_test__student','Old torso-down regression')]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def geom(scene,typ,pos,size,color,end=None):
 g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,typ,np.asarray(size,float),np.asarray(pos,float),np.eye(3).flatten(),np.asarray(color,np.float32))
 if end is not None:mujoco.mjv_connector(g,typ,size[0],np.asarray(pos,float),np.asarray(end,float))
 scene.ngeom+=1

def main():
 out=RUN/'videos';out.mkdir(exist_ok=True);xml=ROOT/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml';model=mujoco.MjModel.from_xml_path(str(xml));model.vis.global_.offwidth=640;model.vis.global_.offheight=640
 data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=640,width=640);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 manifest=[]
 for label,title in ITEMS:
  source=SRC/(label+'.json.gz');e=json.loads(gzip.decompress(source.read_bytes()));rs=[e['initial']]+e['physics'];ts=np.array([r['time'] for r in rs]);roots=np.array([r['root'] for r in rs]);center=(roots.min(0)+roots.max(0))/2;center[2]=.6
  cams=[]
  for az,el in [(125,-18),(90,-82)]:
   c=mujoco.MjvCamera();c.lookat[:]=center;c.distance=2.9;c.azimuth=az;c.elevation=el;cams.append(c)
  times=np.arange(ts[0],ts[-1]+1e-8,1/25);path=out/(label+'.mp4');assert not path.exists();w=imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart','-threads','2']);started=time.monotonic()
  for n,t in enumerate(times):
   ix=int(np.argmin(abs(ts-t)));r=rs[ix];data.qpos[:]=r['qpos'];data.qvel[:]=r['qvel'];mujoco.mj_forward(model,data);views=[];point=np.array(r['wrist']);anchor=np.array(r.get('anchor',e['anchor']));force=np.array(r.get('force',[0,0,0]));root=np.array(r['root']);R=np.array(r.get('pelvis_matrix',np.eye(3)))
   for view,c in enumerate(cams):
    renderer.update_scene(data,camera=c);sc=renderer.scene
    geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,point,[.018]*3,[0,1,1,1]);geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,anchor,[.025]*3,[1,.15,.15,1])
    geom(sc,mujoco.mjtGeom.mjGEOM_LINE,point,[.003]*3,[1,.3,.3,1],anchor)
    if np.linalg.norm(force)>1e-5:geom(sc,mujoco.mjtGeom.mjGEOM_ARROW,point,[.009]*3,[1,.75,0,1],point+.065*force)
    pos=root.copy();pos[2]=.04;direction=R[:,0].copy();direction[2]=0;geom(sc,mujoco.mjtGeom.mjGEOM_ARROW,pos,[.012]*3,[.1,.3,1,1],pos+.25*direction)
    for j in range(0,ix,20):
     k=min(j+20,ix);a=roots[j].copy();b=roots[k].copy();a[2]=b[2]=.015
     if np.linalg.norm(a-b)>1e-6:geom(sc,mujoco.mjtGeom.mjGEOM_LINE,a,[.003]*3,[.05,.4,1,1],b)
    views.append(renderer.render().copy())
   frame=Image.fromarray(np.concatenate(views,axis=1));d=ImageDraw.Draw(frame);d.rectangle((0,0,1280,110),fill=(15,20,28));d.text((12,6),title,font=font,fill='white');d.text((12,31),f'ACTUAL qpos replay | 1x / 25fps | physics t={r["time"]:.2f}s | contact={r.get("active",False)}',font=small,fill='white');d.text((12,54),f'Force XYZ N: {force[0]:+.2f}, {force[1]:+.2f}, {force[2]:+.2f}   |F|={np.linalg.norm(force):.2f}   error={np.linalg.norm(anchor-point)*100:.1f}cm',font=small,fill='white');d.text((12,77),'Cyan: contact | red: anchor | yellow: ACTUAL force (0.065m/N) | blue: heading/root path',font=small,fill='white');d.text((12,115),'FIXED WORLD oblique',font=small,fill='black');d.text((652,115),'FIXED WORLD overhead',font=small,fill='black');w.append_data(np.asarray(frame))
   if n in [0,len(times)//2,len(times)-1]:frame.save(out/(label+f'-frame{n}.jpg'))
  w.close();active=[r for r in rs if r.get('active')];F=np.array([r['force'] for r in active]);E=np.array([np.array(r['anchor'])-r['wrist'] for r in active]);manifest.append(dict(label=label,title=title,source=str(source),source_sha256=digest(source),video=str(path),video_sha256=digest(path),frames=len(times),fps=25,expected_duration=len(times)/25,physics_range=[float(ts[0]),float(ts[-1])],condition=e['condition'],error_cm=float(np.sqrt((E**2).sum(1).mean())*100),force_rms=float(np.sqrt((F**2).sum(1).mean())),boundary=e['boundary'],render_wall_s=time.monotonic()-started));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(label,'DONE',manifest[-1]['render_wall_s'],flush=True)
 renderer.close()
 Path(os.environ['BFM_RECOVERY_OUTPUT'],'result.json').write_text(json.dumps(dict(status='completed',videos=manifest,no_physics_rerun=True)))
if __name__=='__main__':main()
