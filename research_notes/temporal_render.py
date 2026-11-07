"""Four predetermined actual-qpos videos, fixed final checkpoints, no physics rerun."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--repo', type=Path, required=True)
a = p.parse_args()
v = a.root/'videos'
v.mkdir()
(v/'merged').mkdir()


def merge(name, case):
    suffix = a.root/'final-01'/(name+'__'+case+'.json.gz')
    source = suffix.with_name(suffix.name.replace('.json.gz', '-source-prefix.json.gz'))
    ep = json.loads(gzip.decompress(suffix.read_bytes()))
    prefix = json.loads(gzip.decompress(source.read_bytes()))
    ep['physics'] = prefix['physics']+ep['physics']
    ep['render_provenance'] = dict(suffix=str(suffix), source=str(source),
                                   suffix_sha256=hashlib.sha256(suffix.read_bytes()).hexdigest(),
                                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                   physics_rerun=False)
    target = v/'merged'/suffix.name
    with gzip.open(target, 'wt') as f:
        json.dump(ep, f)
    return str(target)


names = ['A-4901_last', 'B-4901_last', 'A-4902_last', 'B-4902_last']
titles = ['A independent | seed4901 | FINAL', 'B continuous | seed4901 | FINAL',
          'A independent | seed4902 | FINAL', 'B continuous | seed4902 | FINAL']
items = []
for number, case in [(1, 'new_near'), (2, 'new_long')]:
    items.append(dict(name=f'{number:02d}-'+case, grid4=True,
                      sides=[dict(path=merge(n, case), title=t) for n, t in zip(names, titles)]))
items.append(dict(name='03-common-prefix-stop-continue', grid4=True,
                  cross_condition_description='Same fixed final policy per arm/seed4901; matched actual prefix8.76s then real stop/continue, no condition-based checkpoint selection',
                  sides=[dict(path=merge(n, c), title=t) for n, c, t in [
                      ('A-4901_last', 'new_stop', 'A independent | STOP | seed4901'),
                      ('A-4901_last', 'new_continue', 'A independent | CONTINUE | seed4901'),
                      ('B-4901_last', 'new_stop', 'B continuous | STOP | seed4901'),
                      ('B-4901_last', 'new_continue', 'B continuous | CONTINUE | seed4901')]]))
items.append(dict(name='04-moving-source', grid4=False,
                  sides=[dict(path=merge(n, 'new_moving'), title=t)
                         for n, t in zip(names[:2], titles[:2])]))
(v/'video-pairs.json').write_text(json.dumps(items, indent=2))
os.environ.update(CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                  PYOPENGL_PLATFORM='egl', M11_VIDEO_RUN=str(v), BFM_RECOVERY_OUTPUT=str(v),
                  BFM_RENDER_ASSET_ROOT=str(a.repo), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
python = '/mnt/data/temp_user/yi.gao/projects/compliance_BFM/.venv/bin/python'
subprocess.run([python, '-u', str(Path(__file__).with_name('temporal_movie.py'))], check=True, cwd=v)
manifest = json.loads((v/'manifest.json').read_text())
assert len(manifest) == 4
import imageio_ffmpeg
for row in manifest:
    path = Path(row['path'])
    assert path.stat().st_size > 10000
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
                    '-f', 'null', '-'], check=True, stdout=subprocess.DEVNULL)
models = json.loads((a.root/'final-01/models.json').read_text())
rows = json.loads((a.root/'final-01/summaries.json').read_text())
index = ['# Issue49 — Native temporal exploration videos', '',
         '仅4条MP4+INDEX；保存的真实qpos，25fps/1×，无新物理重跑或reference动画。',
         '01/02：新6.1cm/73.5cm需求，左上A4901、右上B4901、左下A4902、右下B4902。',
         'A独立探索、B时间连续探索，单步stationary噪声协方差相同；执行均为同网络确定性均值。',
         '03：上排A4901停止/继续，下排B4901停止/继续；8.76s前真实共同前缀，之后伙伴真实平滑改变。',
         '04：A/B4901从同native moving源状态接管。每行世界固定视角+俯视。',
         'NATIVE PREFIX/LEARNED POLICY标明真实源前缀与策略接管，不能把前缀行走归功于新策略。',
         '四条均用每个run预先要求保留的固定末轮，不按任务或终检重选；开发best结果在报告单列。',
         '黄箭头=真实力；红A=伙伴anchor、青C=接触点；蓝线root轨迹。释放后旧anchor误差仅非任务几何。',
         '当前均为特权Teacher，不是新Student/真机感知。所有失败和完整结束过程保留。', '', '## 模型身份']
for name in names:
    m = models[name]
    index.append(f"- {name}: iteration{m['iteration']}; SHA256 {m['sha256']}; runtime {m['runtime']}; {m['path']}")
index += ['', '## 全接触指标（3D RMS cm / 实际力RMS N）']
for row in rows:
    if row['model'] in names and row['case'] in ['new_near', 'new_long', 'new_stop', 'new_moving', 'zero_connected']:
        index.append(f"- {row['label']}: {row['error_cm']:.3f}cm / {row['force_rms']:.3f}N; cap {100*row['cap_fraction']:.1f}%; boundary={row['boundary']}")
index += ['', '## 文件验证']
for row in manifest:
    index.append(f"- {Path(row['path']).name}: {row['duration']:.2f}s/{row['frames']}frames/25fps; SHA256 {row['sha256']}")
(v/'INDEX.md').write_text('\n'.join(index)+'\n')
(v/'validated.json').write_text(json.dumps(dict(status='all_four_fully_decoded', manifest=manifest), indent=2))
