"""Paired 1x movies from real saved qpos only; no dynamics, policy or reference animation."""
import os,json,gzip,hashlib,time
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');RUN=Path(os.environ.get('M11_VIDEO_RUN',str(P/'bridge-v2-owned/runs/m11-privileged-expert-20261101')))
from m9c_render import geom
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=Path(os.environ['BFM_RECOVERY_OUTPUT']);specs=json.loads((RUN/'video-pairs.json').read_text());m=mujoco.MjModel.from_xml_path(str(Path(os.environ.get("BFM_RENDER_ASSET_ROOT",str(ROOT)))/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'));m.vis.global_.offwidth=640;m.vis.global_.offheight=640;m.site_rgba[:,3]=0
 d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=640,width=640);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);manifest=[]
 for item in specs:
  entries=[]
  for v in item['sides']:
   p=Path(v['path']);e=json.loads(gzip.decompress(p.read_bytes()));rs=e['physics'];entries.append(dict(**v,ep=e,rs=rs,ts=np.array([r['time'] for r in rs]),roots=np.array([r['root'] for r in rs]),sha256=digest(p)))
  assert all({k:x for k,x in v['ep']['condition'].items() if k!='enabled'}=={k:x for k,x in entries[0]['ep']['condition'].items() if k!='enabled'} for v in entries),'Only explicitly labelled physical contact enabled may differ'
  for v in entries:assert v['ep']['condition']['enabled'] or 'NO FORCE' in v['title']
  start=min(v['ts'][0] for v in entries);end=max(v['ts'][-1] for v in entries);times=np.arange(start,end+1e-8,.04);path=out/(item['name']+'.mp4');assert not path.exists();writer=imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart','-threads','2']);started=time.monotonic()
  center=np.array([.45,0,.6]) if entries[0]['ep']['condition'].get('family')=='walking' else np.array([0,0,.6]);cameras=[]
  for az,el in [(125,-18),(90,-82)]:
   cam=mujoco.MjvCamera();cam.lookat[:]=center;cam.distance=2.9;cam.azimuth=az;cam.elevation=el;cameras.append(cam)
  for n,t in enumerate(times):
   panels=[]
   for side in entries:
    ix=int(np.argmin(abs(side['ts']-t)));r=side['rs'][ix];d.qpos[:]=r['qpos'];d.qvel[:]=r['qvel'];mujoco.mj_forward(m,d);point=np.array(r['wrist']);anchor=np.array(r.get('anchor',side['ep']['anchor']));force=np.array(r['force']);root=np.array(r['root']);R=np.array(r['pelvis_matrix']);views=[];marks=[]
    for vi,cam in enumerate(cameras):
     renderer.update_scene(d,camera=cam);sc=renderer.scene
     contacts=r.get('contacts',[])
     for contact in contacts:
      pp=np.array(contact['point']);aa=np.array(contact['anchor']);ff=np.array(contact['force'])
      geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,pp,[.018]*3,[0,1,1,1]);geom(sc,mujoco.mjtGeom.mjGEOM_SPHERE,aa,[.022]*3,[1,.15,.15,1]);geom(sc,mujoco.mjtGeom.mjGEOM_LINE,pp,[.003]*3,[1,.3,.3,1],aa)
      if np.linalg.norm(ff)>1e-5:geom(sc,mujoco.mjtGeom.mjGEOM_ARROW,pp,[.009]*3,[1,.75,0,1],pp+.065*ff)
     target=np.array([np.cos(r.get('target_yaw',0)),np.sin(r.get('target_yaw',0)),0]);gp=root.copy();gp[2]=.08;geom(sc,mujoco.mjtGeom.mjGEOM_ARROW,gp,[.009]*3,[0,.9,.2,1],gp+.3*target)
     pos=root.copy();pos[2]=.04;direction=R[:,0].copy();direction[2]=0;geom(sc,mujoco.mjtGeom.mjGEOM_ARROW,pos,[.012]*3,[.1,.3,1,1],pos+.25*direction)
     for j in range(0,ix,20):
      a=side['roots'][j].copy();b=side['roots'][min(j+20,ix)].copy();a[2]=b[2]=.015
      if np.linalg.norm(a-b)>1e-6:geom(sc,mujoco.mjtGeom.mjGEOM_LINE,a,[.003]*3,[.05,.4,1,1],b)
     views.append(renderer.render().copy());camera=sc.camera[0];cp=np.mean([np.array(c.pos) for c in sc.camera],axis=0);forward=np.array(camera.forward);up=np.array(camera.up);right=np.cross(forward,up);focal=640*float(camera.frustum_near)/(float(camera.frustum_top)-float(camera.frustum_bottom))
     def pixel(v):
      dv=v-cp;z=float(dv@forward);return None if z<=0 else (vi*640+320+focal*float(dv@right)/z,320-focal*float(dv@up)/z)
     
     for contact in contacts:
      pp=np.array(contact['point']);aa=np.array(contact['anchor']);ff=np.array(contact['force']);marks.append((pixel(pp),pixel(aa),pixel(pp+.065*ff)))
    frame=Image.fromarray(np.concatenate(views,axis=1));draw=ImageDraw.Draw(frame)
    for C,A,F in marks:
     if C is not None and F is not None:draw.line([C,F],fill=(255,220,0),width=3)
     for xy,color,label in [(C,(0,255,255),'C'),(A,(255,40,40),'A')]:
      if xy is not None:
       x,y=xy;draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2);draw.text((x+7,y-16 if label=='A' else y+4),label,font=small,fill=color)
    draw.rectangle((0,0,1280,126),fill=(15,20,28));draw.text((12,5),side['title'],font=font,fill='white')
    draw.text((12,29),f'ACTUAL qpos | 1x | t={r["time"]:.2f}s | attached force active={r.get("active",False)} | net yaw moment COM={r.get("net_yaw_moment_com",0):+.3f}Nm',font=small,fill='white')
    for j,contact in enumerate(r.get('contacts',[])[:2]):
     ff=contact['force'];error=100*np.linalg.norm(np.array(contact['anchor'])-contact['point']);draw.text((12,51+j*21),f'Point {contact["slot"]} {contact["link"]}: Fxyz=({ff[0]:+.2f},{ff[1]:+.2f},{ff[2]:+.2f})N | '+('tracking' if r.get('active') else 'non-task geometry')+f' {error:.1f}cm',font=small,fill='white')
    T=np.array(r['torso_matrix']);draw.text((12,100),f'Blue pelvis {np.degrees(np.arctan2(R[1,0],R[0,0])):+.1f}deg | torso {np.degrees(np.arctan2(T[1,0],T[0,0])):+.1f}deg | green target {np.degrees(r.get("target_yaw",0)):+.1f}deg | yellow real force',font=small,fill='white');panels.append(np.asarray(frame))
   full=np.concatenate(panels,axis=0);writer.append_data(full)
   if n in [0,len(times)//2,len(times)-1]:Image.fromarray(full).save(out/(item['name']+f'-check{n}.jpg'))
  writer.close();row=dict(name=item['name'],path=str(path),sha256=digest(path),frames=len(times),fps=25,duration=len(times)/25,seconds=time.monotonic()-started,sides=[{k:v[k] for k in ['title','path','sha256']} for v in entries],condition=entries[0]['ep']['condition']);manifest.append(row);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('M11_VIDEO',item['name'],row['seconds'],flush=True)
 renderer.close();(out/'result.json').write_text(json.dumps(dict(status='completed',videos=manifest,no_physics_rerun=True)))
if __name__=='__main__':main()
