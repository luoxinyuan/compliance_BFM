"""Render the same full episode for B0/U/T, without stepping physics."""
import gzip,hashlib,json,os,pickle,subprocess,sys,time
from pathlib import Path
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
    root=Path(os.environ["CTFT_PINNED_RUNTIME"]);folder=Path(sys.argv[1]);out=folder/"videos"
    out.mkdir(exist_ok=False)
    m=mujoco.MjModel.from_xml_path(str(root/"humanoidverse/data/robots/g1/scene_29dof_freebase_mujoco.xml"))
    d=mujoco.MjData(m);m.vis.global_.offwidth=480;m.vis.global_.offheight=480
    renderer=mujoco.Renderer(m,height=400,width=480);cam=mujoco.MjvCamera();cam.azimuth=118;cam.elevation=-12;cam.distance=2.75
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
    start=time.monotonic();manifest=[]
    for case,b in [(0,"continue"),(0,"stop"),(0,"no_force"),(23,"continue")]:
        parent=f"ctft-eval-parent-{case:02d}";eps=[];files={}
        for name in ("B0","U","T"):
            p=folder/f"{parent}-{name}-{b}.pkl.gz"
            with gzip.open(p,"rb") as stream:x=pickle.load(stream)
            eps.append(x);files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        camera=eps[0]["initial_qpos"][:3].copy();camera[0]+=.22;camera[2]=.80;cam.lookat=camera
        duration=max(len(x["physics"]) for x in eps)*.005
        movie=out/f"{parent}-{b}.mp4"
        with imageio.get_writer(str(movie),fps=25,codec="libx264",quality=8,macro_block_size=16,ffmpeg_params=["-threads","2","-pix_fmt","yuv420p"]) as writer:
            for t in np.arange(0,duration,.04):
                panels=[]
                for x in eps:
                    n=int(round(t/.005));end=n>=len(x["physics"])
                    r=x["physics"][min(n,len(x["physics"])-1)]
                    d.qpos[:]=r["qpos"];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam)
                    panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel)
                    draw.rectangle([0,0,480,63],fill=(32,32,32))
                    draw.text((6,3),f'{x["model"]} | {parent} | {b} | 1x {t:.2f}s',font=font,fill="white")
                    draw.text((6,24),f'Force {np.linalg.norm(r["force"]):.2f} N | root travel X {r["root"][0]-x["initial_qpos"][0]:+.2f} m',font=font,fill="white")
                    status="SAFETY STOP - final frame held" if end and x["hard"] else ("loaded" if r["active"] else "no applied load")
                    draw.text((6,42),status,font=font,fill="orange" if end else "white")
                    panels.append(np.asarray(panel))
                writer.append_data(np.concatenate(panels,axis=1))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-i",str(movie),"-f","null","-"],check=True)
        manifest.append(dict(path=movie.name,sources=files,duration=duration,fps=25,sha256=hashlib.sha256(movie.read_bytes()).hexdigest()))
    renderer.close();(out/"manifest.json").write_text(json.dumps(dict(videos=manifest,wall_s=time.monotonic()-start),indent=2))
if __name__=="__main__":main()
