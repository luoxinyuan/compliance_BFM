"""Render the lowest-index train-lineage native regression; pose kinematics only."""
from pathlib import Path
import gzip,pickle,hashlib,json,sys,time
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg,subprocess
s,o=map(Path,sys.argv[1:]);o.mkdir(exist_ok=True);start=time.monotonic()
m=mj.MjModel.from_xml_path(str(s/'runtime-v1a/humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml')); d=mj.MjData(m)
weight=m.body_mass.sum()*np.linalg.norm(m.opt.gravity)
eps=[]; ident={}
for a in ['B0','U','T']:
    p=s/f'runs/endpoint-evaluation-02/ctft-native-regression-00-{a}-native_continuation.pkl.gz'
    with gzip.open(p,'rb') as f:x=pickle.load(f)
    assert x['spec']['mode']==0 and x['spec']['source']['case']==0
    eps.append(x);ident[a]=hashlib.sha256(p.read_bytes()).hexdigest()
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
movie=o/'native-00-B0-U-T-1x.mp4'
def at(v):
    dx,dy,z=v
    return (int(140+137*(.86*dx-.5*dy)),int(327-137*z+28*(.5*dx+.86*dy)))
wr=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
with imageio.get_writer(str(movie),fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-threads','2','-pix_fmt','yuv420p']) as w:
    for t in range(0,max(len(x['physics']) for x in eps),8):
        frames=[]
        for x in eps:
            ph=x['physics'];r=ph[min(t,len(ph)-1)];d.qpos[:]=r['qpos'];mj.mj_kinematics(m,d)
            im=Image.new('RGB',(352,384),(243,245,244)); g=ImageDraw.Draw(im)
            for y in np.arange(-1.5,1.6,.25):g.line([at((i,y,0)) for i in [-2,3]],fill='lightgray')
            for b in range(2,m.nbody):
                parent=m.body_parentid[b]
                if parent>0:g.line([at(d.xpos[parent]),at(d.xpos[b])],fill='#273e54',width=5)
            f=np.asarray(r['force']);p=at(d.xpos[wr]); assert np.linalg.norm(f)==0
            g.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill='teal')
            g.rectangle((0,0,352,80),fill='#1c2933')
            for i,line in enumerate([f"{x['model']}  |  native 00  |  {t*.005:.2f} s",f'Wrist 0.00 N  |  never connected', 'HARD STOP - end held' if t>=len(ph) else 'g=0 native continuation']):
                g.text((6,4+23*i),line,font=font,fill='white' if i<2 else 'orange')
            load=np.asarray(r['foot_normal_force_n']); g.text((6,352),'  '.join(f'{side} {n:.0f} N '+('ON' if n>=.05*weight else 'off') for side,n in zip('LR',load)),font=font,fill='#202020')
            frames.append(np.asarray(im))
        w.append_data(np.concatenate(frames,axis=1))
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(movie),'-f','null','-'],check=True)
(o/'video-manifest.json').write_text(json.dumps(dict(sources=ident,sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),physics_ticks=0,wall_s=time.monotonic()-start,fps=25,projection='fixed orthographic skeleton FK from archived qpos, wrist marker and zero-length force vector'),indent=2))
