"""Only saved actual source-prefix+suffix qpos, no rerun or reference animation."""
import os,json,gzip,copy,hashlib,subprocess,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--video-dir',default='videos-final32');a=p.parse_args();v=a.root/a.video_dir;v.mkdir();merged=v/'merged';merged.mkdir()
def merge(name,case):
 src=a.root/'final-01'/(name+'__'+case+'.json.gz');pre=src.with_name(src.name.replace('.json.gz','-source-prefix.json.gz'));ep=json.loads(gzip.decompress(src.read_bytes()));prefix=json.loads(gzip.decompress(pre.read_bytes()));ep['physics']=prefix['physics']+ep['physics'];ep['render_provenance']={'suffix':str(src),'suffix_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_prefix':str(pre),'source_sha256':hashlib.sha256(pre.read_bytes()).hexdigest(),'physics_rerun':False};dest=merged/src.name
 with gzip.open(dest,'wt') as f:json.dump(ep,f)
 return str(dest)
items=[]
for i,case,title in [(1,'new_near','New6.8cm'),(2,'new_long','New69cm'),(4,'zero_connected','Connected ZERO demand')]:
 references=[('native_standing','Frozen BFM STANDING')] if case=='zero_connected' else [('I','I frozen historical walking')]
 sides=[{'path':merge(name,case),'title':caption+' | '+title} for name,caption in references+[('C0','C0 frozen local / delayed walking'),('guided_last','NATIVE guided RAW head | FIXED FINAL32 | NOT dev-selected'),('coordinate_last','NATIVE guided COORD head | FIXED FINAL32 | NOT dev-selected')]]
 items.append({'name':f'{i:02d}-'+case,'grid4':True,'sides':sides})
items.append({'name':'03-guided-common-prefix-stop-continue','cross_condition_description':'SAME frozen guided fixed final32 model; exact same demand prefix until8.27s, then real partner braking/continuation; no condition-based policy selection','sides':[{'path':merge('coordinate_last',case),'title':'SAME native coordinate | '+label+' | after8.27s'} for case,label in [('new_stop','partner stops'),('new_continue','partner continues')]]})
(v/'video-pairs.json').write_text(json.dumps(items,indent=2));os.environ.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',PYOPENGL_PLATFORM='egl',M11_VIDEO_RUN=str(v),BFM_RECOVERY_OUTPUT=str(v),BFM_RENDER_ASSET_ROOT=str(a.repo),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
subprocess.run([str(Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python')),'-u',str(Path(__file__).with_name('native_movie.py'))],check=True,cwd=v)
manifest=json.loads((v/'manifest.json').read_text());models=json.loads((a.root/'final-01/models.json').read_text());rows=json.loads((a.root/'final-01/summaries.json').read_text());lines=['# Issue48 — Native capability contact learning videos','','仅4条关键MP4，真实qpos/1×/25fps；拼入同一实际回合保存的真实源前缀，非reference/q_PD动画。','01：相同新6.8cm需求，左上I、右上C0、左下原生引导raw-head末轮32、右下原生引导coordinate-head末轮32。','02：相同新69cm需求，同样四个模型；保留所有失败，未按任务换新模型。普通探索全矩阵保留在报告，视频固定展示两种同协方差的均值head。','03：同一coordinate固定末轮32checkpoint（开发未选中，不改原选择），共同前缀8.27s后真实伙伴停止/继续。每行固定世界视角＋俯视。','04：已连接零需求，左上换为原生Standing，其他三项保持同一冻结模型，展示无请求动作与牵拉；不同于关闭真实接触。','黄箭头实际力、红点anchor、青点接触点、蓝线root轨迹；release后旧anchor误差是非任务几何描述。','两条新策略均为特权Teacher，不是新Student或真机感知验证。', '本版本替换早期开发best可视化：三组best都为初始化，无法展示PPO学到的变化。这里统一用预先保存的末轮32，不按条件挑模型，不修改开发选模，早期视频在服务器/private分析目录完整保留。','模型源：新策略fresh高层网络，原生goal/motion continuation初始化＋真实PPO；I/C0仅冻结对照，不参与新策略初始化。','','## 模型身份（固定末轮用于学习变化诊断，不按封存重选）']
for k,vv in models.items():lines.append(f"- {k}: SHA256 {vv['sha256']}; {vv['path']}")
lines+=['','较短分支完整结束后保持末帧以同步展示；非慢放。若开发选择iteration0，它就是原生初始化，并不表示PPO胜出。','本对照同时改变协方差方向和总探索方差，不能单独归因为有方向性；概率几何/信赖域也随协方差改变。','','## 视频与实际全接触指标']
import imageio_ffmpeg
for m in manifest:
 path=Path(m['path']);assert path.stat().st_size>10000;subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],check=True,stdout=subprocess.DEVNULL);lines.append(f"- {path.name}: {m['duration']:.2f}s/{m['frames']}frames; SHA256 {m['sha256']}")
for r in rows:
 if r['model'] in ['I','C0','ordinary_best','guided_last','coordinate_last','native_standing'] and r['case'] in ['new_near','new_long','new_stop','new_continue','zero_connected']:lines.append(f"- {r['label']}: {r['error_cm']:.3f}cm / {r['force_rms']:.3f}N; boundary={r['boundary']}")
(v/'INDEX.md').write_text('\n'.join(lines)+'\n');(v/'validated.json').write_text(json.dumps({'status':'all_four_fully_decoded','manifest':manifest},indent=2))
