"""Full synchronized kinetic replay, no dynamics, same scale across panels."""
from pathlib import Path
import os,sys,gzip,pickle,subprocess,json,hashlib,time
import numpy as np,mujoco as mj
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
s=Path(sys.argv[1]);o=s/'videos';o.mkdir(exist_ok=False);clock=time.monotonic()
m=mj.MjModel.from_xml_path(str(Path(os.environ['CTFT_PINNED_RUNTIME'])/'humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml'))
d=mj.MjData(m); weight=m.body_mass.sum()*np.linalg.norm(m.opt.gravity);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
case=json.loads((s/'lock/lock.json').read_text())['picks'][0]['source']['case'];movies=[]
for group in ['branches','prior','stop-and-no-force']:
    plan=[('reference',b,'') for b in ['continue','stop','no_force']] if group=='branches' else [('reference','continue',''),('d2','continue','-walk')]
    if group=='stop-and-no-force':plan=[(k,b,suffix) for b in ['stop','no_force'] for k,suffix in [('reference',''),('d2','-walk')]]
    paths=[s/k/f'{case:02d}-{b}{suffix}.gz' for k,b,suffix in plan]
    if not all(p.is_file() for p in paths):continue
    eps=[]
    for p in paths:
        with gzip.open(p,'rb') as f:eps.append(pickle.load(f))
    origin=eps[0]['initial_qpos'][:3]; wr=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'right_wrist_yaw_link')
    def proj(p):
        dx,dy=p[:2]-origin[:2];return np.array([80+135*(.96*dx-.28*dy),325-133*p[2]+18*(.28*dx+.96*dy)])
    def xy(p):return tuple(proj(np.asarray(p)).astype(int))
    dest=o/f'U-{case:02d}-{group}-1x.mp4'
    with imageio.get_writer(str(dest),fps=25,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-threads','2']) as w:
        for idx in range(0,max(len(x['physics']) for x in eps),8):
            panels=[]
            for (kind,b,_),x in zip(plan,eps):
                ph=x['physics'];r=ph[min(idx,len(ph)-1)];finished=idx>=len(ph);f=np.asarray(r['force']);mag=np.linalg.norm(f)
                d.qpos[:]=r['qpos'];mj.mj_kinematics(m,d);im=Image.new('RGB',(384,384),'#f1f4f3');g=ImageDraw.Draw(im)
                for j in range(2,m.nbody):
                    if m.body_parentid[j]:g.line([xy(d.xpos[m.body_parentid[j]]),xy(d.xpos[j])],fill='#2d4059',width=5)
                tip=np.array(r['wrist']);a=proj(tip);v=proj(tip+.06*f)-a;end=a+v
                g.ellipse([tuple(a-3),tuple(a+3)],fill='#c33721')
                if np.linalg.norm(v)>1:
                    g.line([tuple(a),tuple(end)],fill='#c33721',width=4)
                    unit=v/np.linalg.norm(v);orth=np.array([-unit[1],unit[0]])
                    g.polygon([tuple(end),tuple(end-8*unit+4*orth),tuple(end-8*unit-4*orth)],fill='#c33721')
                st=('NEVER LOADED' if b=='no_force' else ('CONNECTED' if r['active'] else 'RELEASED'))
                zz='walking z' if kind=='d2' and idx<200 else 'standing z'
                g.rectangle((0,0,384,70),fill='#1e2d3a')
                g.text((5,2),f'U | train {case:02d} | {b} | {zz}',fill='white',font=font)
                g.text((5,24),f'{idx*.005:.2f} s | {st} | {mag:.2f} N',fill='#ffce90',font=font)
                g.text((5,46),'END HELD'+(' - HARD STOP' if x['hard'] else '') if finished else '1x | red = applied force (0.06 m/N)',fill='white',font=font)
                g.text((4,355),'   '.join(f'{name}: {n:.0f} N '+('ON' if n>=.05*weight else 'off') for name,n in zip('LR',r['foot_normal_force_n'])),fill='#162524',font=font)
                panels.append(np.asarray(im))
            w.append_data(np.concatenate(panels,axis=1))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(dest),'-f','null','-'],check=True)
    movies.append(dict(name=dest.name,sha=hashlib.sha256(dest.read_bytes()).hexdigest(),sources=[str(p.relative_to(s)) for p in paths]))
(o/'manifest.json').write_text(json.dumps(dict(videos=movies,wall_s=time.monotonic()-clock),indent=2))
