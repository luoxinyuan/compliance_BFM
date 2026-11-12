"""Saved-state inference diagnostics and actual-qpos rendering; zero mj_step."""
import json,time,hashlib,subprocess,sys,os,math
from pathlib import Path
import numpy as np,torch,mujoco as mj,onnxruntime as ort
from PIL import Image,ImageDraw

def main():
 import imageio_ffmpeg
 s=Path(sys.argv[1]);out=s/'analysis02';out.mkdir(exist_ok=False);a=s/'evidence/interactive/pbfm';t=time.time();c=json.loads((a/'config.json').read_text());cfg=ort.SessionOptions();cfg.intra_op_num_threads=4;cfg.inter_op_num_threads=1;p=ort.InferenceSession(str(a/'policy.onnx'),sess_options=cfg,providers=['CPUExecutionProvider'])
 traces=[torch.load(s/f'runs/free-01/v{v:.2f}-r0/trace.pt',weights_only=False) for v in [0.,.09,.18]];x=traces[0];rows=[]
 for i in [0,100,200,350,500]:
  obs=x['controls'][i]['obs'];baseline=p.run(None,obs)[0];e=[]
  for speed,valid in [(0,0),(0,1),(.09,1),(.18,1),(.8,1)]:
   feed={k:v.copy() for k,v in obs.items()};ph=x['physics'][min(i*4,len(x['physics'])-1)];y=-math.atan2(float(obs['heading'][0,1]),1-float(obs['heading'][0,0]));feed['velocity_command']=np.array([[speed*np.cos(y),-speed*np.sin(y),valid]],np.float32);action,est=p.run(None,feed);target=np.asarray(c['defaultJointPosition'])+np.asarray(c['actionScale'])*np.clip(action[0]*5,-5,5);bt=np.asarray(c['defaultJointPosition'])+np.asarray(c['actionScale'])*np.clip(baseline[0]*5,-5,5)
   e.append(dict(speed=speed,valid=valid,raw_action_l2=float(np.linalg.norm(action-baseline)),max_qpd_difference_deg=float(np.rad2deg(np.max(np.abs(target-bt)))),estimated_velocity=est[0].tolist()))
  rows.append(dict(time=x['controls'][i]['time'],cases=e))
 (out/'same_state_command.json').write_text(json.dumps(rows,indent=2));raw=(a/'contexts.json').read_bytes();(out/'context_identity.json').write_text(json.dumps({'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'expected_git_blob_sha1':'1a2ab29fd6456ca69e21e4e871202845f369c2e8','file_sha256':hashlib.sha256(raw).hexdigest()},indent=2))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
 for v,x in zip([0,.09,.18],traces):
  ph=x['physics'];co=x['controls'];tt=np.array([z['time'] for z in ph]);axes[0].plot([z['time'] for z in co],[z['u'] for z in co],label=str(v));axes[1].plot(tt,[z['qvel'][0] for z in ph],label=str(v));axes[2].plot(tt,[max([j['speed_mps'] for j in z['contacts'] if j['normal_N']>1e-6],default=0) for z in ph],label=str(v))
 for ax,l in zip(axes,['command m/s','actual root vx m/s','loaded slip max m/s']):ax.set_ylabel(l);ax.legend();ax.grid(alpha=.2)
 axes[2].axhline(.45,color='r',ls=':');axes[2].set_xlabel('actual seconds');fig.tight_layout();fig.savefig(out/'eri-response.png',dpi=140);plt.close(fig)
 # Native independent robot/world; render saved qpos only, never step physics.
 m=mj.MjModel.from_xml_path(str(a/'robot/scene.xml'));d=mj.MjData(m);m.vis.global_.offwidth=480;m.vis.global_.offheight=360;render=mj.Renderer(m,height=360,width=480);cam=mj.MjvCamera();cam.distance=3.;cam.azimuth=135;cam.elevation=-15
 video=out/'pbfm-eri-zero-and-speed.mp4';proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x420','-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-y',str(video)],stdin=subprocess.PIPE)
 for frame in range(300):
  canvas=Image.new('RGB',(1440,420),'white');draw=ImageDraw.Draw(canvas)
  for j,(speed,x) in enumerate(zip([0.,.09,.18],traces)):
   row=x['physics'][min(frame*8,len(x['physics'])-1)];d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel'];mj.mj_forward(m,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.7;render.update_scene(d,camera=cam);canvas.paste(Image.fromarray(render.render()),(j*480,60));u=x['controls'][min(frame*2,len(x['controls'])-1)]['u'];draw.text((j*480+8,8),f'Native P-BFM: peak command {speed:.2f} m/s',fill='black');draw.text((j*480+8,25),f't={row["time"]:.2f}s u={u:.3f} actual vx={row["qvel"][0]:.3f}',fill='black');draw.text((j*480+8,42),'Real qpos 1x / no external force / fixed native z',fill='black')
  proc.stdin.write(np.asarray(canvas).tobytes())
  if frame in [75,250]:canvas.save(out/f'frame{frame}.jpg')
 proc.stdin.close();assert proc.wait()==0;render.close();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],check=True)
 (out/'cost.json').write_text(json.dumps({'job_s':time.time()-t,'physics_steps':0,'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'fps':25,'frames':300,'duration_s':12},indent=2));print('postprocess complete',time.time()-t,flush=True)
if __name__=='__main__':main()
