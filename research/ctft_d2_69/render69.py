from pathlib import Path
import gzip, pickle, json, os, subprocess, hashlib, time
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as io
import imageio_ffmpeg as ff

start=time.monotonic()
base=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/experiments')
s=Path(__file__).resolve().parent
o=Path(os.environ['REVIEW_OUT'])
o.mkdir()
model=mj.MjModel.from_xml_path(str(base/'bfm_ctft_v1/runtime-v1a/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
W,H=480,352
model.vis.global_.offwidth=W
model.vis.global_.offheight=H
data=mj.MjData(model)
renderer=mj.Renderer(model,height=H,width=W)
cam=mj.MjvCamera()
cam.azimuth=90
cam.elevation=0
cam.distance=2.65
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
gold='#ffd33a'
stats={}
files={}
for mode,tag in [('reference',''),('d2','-walk')]:
    for b in ['continue','stop','no_force']:
        path=s/mode/f'00-{b}{tag}.gz'
        with gzip.open(path,'rb') as f:
            x=pickle.load(f)
        ph=x['physics']
        cam.lookat=[x['initial_qpos'][0]+.28,x['initial_qpos'][1],.72]
        name=f'{mode}-{b}'
        forces=np.array([r['force'] for r in ph])
        act=np.array([r['active'] for r in ph],bool)
        loaded=act & (np.linalg.norm(forces,axis=1)>=1.)
        stats[name]=dict(source_sha=hashlib.sha256(path.read_bytes()).hexdigest(),ticks=len(ph),anchor_z_range_active=np.ptp([r['anchor'][2] for r in ph if r['active']]) if act.any() else None,loaded_ticks=int(loaded.sum()),down_fraction=float((forces[loaded,2]<0).mean()) if loaded.any() else None,mean_xyz=forces[loaded].mean(0).tolist() if loaded.any() else None)
        pathout=o/(name+'.mp4')
        files[name]=(pathout,(len(ph)+7)//8)
        with io.get_writer(str(pathout),fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-threads','2']) as w:
            for i in range(0,len(ph),8):
                r=ph[i]
                data.qpos[:]=r['qpos']
                mj.mj_forward(model,data)
                renderer.update_scene(data,camera=cam)
                c=renderer.scene.camera[0]
                right=np.cross(c.forward,c.up)
                pos=(c.pos+renderer.scene.camera[1].pos)/2
                def project(v):
                    rel=np.asarray(v)-pos
                    gain=H*c.frustum_near/(c.frustum_top-c.frustum_bottom)/np.dot(rel,c.forward)
                    return np.array([W/2+gain*np.dot(rel,right),H/2-gain*np.dot(rel,c.up)])
                im=Image.new('RGB',(W,448),'#172635')
                im.paste(Image.fromarray(renderer.render()),(0,32))
                d=ImageDraw.Draw(im)
                wrist=project(r['wrist'])+[0,32]
                anchor=project(r['anchor'])+[0,32]
                force=forces[i]
                state=('NO FORCE TRIAL' if b=='no_force' else 'CONNECTED' if r['active'] else 'RELEASED' if act[:i].any() else 'PRE-CONTACT')
                if r['active']:
                    tip=project(np.asarray(r['wrist'])+.035*force)+[0,32]
                    v=tip-wrist
                    n=np.linalg.norm(v)
                    if n>=2:
                        unit=v/n
                        ort=np.array([-unit[1],unit[0]])
                        d.line([tuple(wrist),tuple(tip)],fill=gold,width=3)
                        d.polygon([tuple(tip),tuple(tip-min(n,9)*unit+3*ort),tuple(tip-min(n,9)*unit-3*ort)],fill=gold)
                    d.ellipse([tuple(anchor-7),tuple(anchor+7)],outline='#ff514a',width=3)
                d.ellipse([tuple(wrist-4),tuple(wrist+4)],outline='#2898ff',width=2)
                z='15% walk' if mode=='d2' and i<200 else 'stand'
                d.text((8,6),f'U   {b.replace("_"," ")}   |   z: {z}   |   {i*.005:5.2f} s   |   {state}',font=font,fill='white')
                d.text((8,387),'anchor',font=font,fill='#ff6c63')
                d.text((75,387),'wrist',font=font,fill='#4facff')
                d.text((135,387),f'applied {np.linalg.norm(force):5.2f} N    (0.035 m/N)',font=font,fill=gold)
                d.text((8,409),f'World X {force[0]:+5.2f}    Y {force[1]:+5.2f}    Z up {force[2]:+5.2f} N',font=font,fill='white')
                d.text((8,428),f'Side view  +X right / +Z up     foot load L/R: {r["foot_normal_force_n"][0]:.0f}/{r["foot_normal_force_n"][1]:.0f} N',font=font,fill='white')
                w.append_data(np.asarray(im))
renderer.close()
plans={'prior':['reference-continue','d2-continue'],'branches':['reference-continue','reference-stop','reference-no_force'],'stop-and-no-force':['reference-stop','d2-stop','reference-no_force','d2-no_force']}
for label,ids in plans.items():
    maxn=max(files[n][1] for n in ids)
    cmd=[ff.get_ffmpeg_exe(),'-loglevel','error','-filter_complex_threads','1']
    pipes=[]
    for j,n in enumerate(ids):
        p,length=files[n]
        cmd+=['-i',str(p)]
        # A black panel marks the actual end of a shorter branch.
        pipes.append(f'[{j}:v]tpad=stop={maxn-length}:color=black[p{j}]')
    coords=['0_0','w0_0','0_h0','w0_h0'] if len(ids)==4 else ['0_0','w0_0','w0+w1_0']
    pipes.append(''.join(f'[p{j}]' for j in range(len(ids)))+f'xstack=inputs={len(ids)}:layout='+'|'.join(coords[:len(ids)])+'[v]')
    out=o/f'U-00-{label}-1x.mp4'
    subprocess.run(cmd+['-filter_complex',';'.join(pipes),'-map','[v]','-c:v','libx264','-crf','17','-threads','2','-pix_fmt','yuv420p',str(out)],check=True)
    subprocess.run([ff.get_ffmpeg_exe(),'-loglevel','error','-i',str(out),'-f','null','-'],check=True)
(o/'force-audit.json').write_text(json.dumps(dict(series=stats,wall_s=time.monotonic()-start),indent=2))
