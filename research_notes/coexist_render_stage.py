import os,time,json,pathlib,subprocess,hashlib,signal
P=pathlib.Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM');R=P/'bridge-v2-owned/runs/feedback-coexistence-20261105';W=P/'bridge-v2-owned/trees/feedback-coexistence/repo';O=R/'videos';GPU='2';start=time.time()
for name in ['final-learned','final-reference','final-FR']:
 assert json.loads((R/(name+'-ledger.json')).read_text())['exit_code']==0
assert not O.exists();O.mkdir()
items=[]
for index,task,description in [(1,'holdout_local6','New6cm local request'),(2,'holdout_long72','New72cm long request')]:
 sides=[]
 for folder,model,title in [('final-reference','I','I | original reliable walking'),('final-reference','C0','C0 | local positive / long failure'),('final-learned','seed4601_last','RL4601 | last32 | NOT selected'),('final-learned','seed4602_last','RL4602 | last32 | NOT selected')]:
  sides.append(dict(path=str(R/folder/(model+'__'+task+'.json.gz')),title=title+' | '+description))
 items.append(dict(name=f'{index:02d}-'+task,grid4=True,sides=sides))
items.append(dict(name='03-common-prefix-stop-continue',cross_condition_description='Same frozen seed4601 last32, prefix identical until7.37s. No condition-based policy selection.',sides=[dict(path=str(R/'final-learned'/('seed4601_last__holdout_'+k+'.json.gz')),title='SAME RL4601 last32 | '+k+' after7.37s | research negative') for k in ['stop','continue']]))
(O/'video-pairs.json').write_text(json.dumps(items,indent=2))
before=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert int(before.split(',')[0])<100 and int(before.split(',')[1])==0,before
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='2',PYOPENGL_PLATFORM='egl',M11_VIDEO_RUN=str(O),BFM_RECOVERY_OUTPUT=str(O),BFM_RENDER_ASSET_ROOT=str(W),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',PYTHONPATH=str(W));os.sched_setaffinity(0,{16,17,18,19});t=time.monotonic()
f=open(O/'stdout.log','w');c=subprocess.Popen([str(P/'.venv/bin/python'),'-u',str(W/'research_notes/adapter_render.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=O)
l=dict(status='running',pid=c.pid,supervisor=os.getpid(),gpu=GPU,before=before,started=time.time(),runtime=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),allocated_wall=300,scope='offline render actual saved physics qpos; zero new physics or training');ledger=R/'video-ledger.json';ledger.write_text(json.dumps(l,indent=2))
try:c.wait(timeout=290)
except subprocess.TimeoutExpired:
 os.killpg(c.pid,signal.SIGTERM);c.wait(timeout=10)
finally:
 l.update(status='finished',exit_code=c.returncode,job_wall=time.monotonic()-t);ledger.write_text(json.dumps(l,indent=2));f.close()
assert c.returncode==0
manifest=json.loads((O/'manifest.json').read_text());assert len(manifest)==3
for row in manifest:
 p=pathlib.Path(row['path']);assert p.stat().st_size>10000
 # Full decoding is additional validation, not inferred from renderer success.
 subprocess.run([__import__('imageio_ffmpeg').get_ffmpeg_exe(),'-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
models=json.loads((R/'final-frozen-models.json').read_text())['all']
lines=['# Issue46: real-feedback coexistence videos','', '仅3条关键视频。真实保存qpos、1×/25fps；不是reference或q_PD动画。未按任务拼接策略。','', '01：新6cm。左上I、右上C0、左下RL4601末轮、右下RL4602末轮。','02：新72cm，同样四个冻结模型；保留C0长距失败。','03：同一个RL4601末轮，共同物理前缀到7.37s后，上方伙伴停止、下方继续。每行左侧固定世界视角、右侧俯视。','', '两个RL末轮是预定负结果对照，不是开发选定新基线；两seed开发均选择初始化I。新测试后没有补训或重选。','完整接触与释放恢复保留。释放后的旧anchor几何误差只是描述，不是任务误差。黄箭头真实力、红点anchor、青点实际接触点、蓝线root路径。','', '模型身份：']
for k,v in models.items():lines += [f"- {k}: {v['sha256']}; {v['path']}"]
for row in manifest:lines += [f"- {row['name']}.mp4: {row['duration']:.2f}s; SHA256 {row['sha256']}"]
summary=[]
for name in ['final-learned','final-reference','final-FR']:summary += json.loads((R/name/'summaries.json').read_text())
lines+=['','视频条件实际全接触指标（cm/N）：']
for x in summary:
 if any(x['label']==pathlib.Path(side['path']).name.removesuffix('.json.gz') for item in items for side in item['sides']):lines.append(f"- {x['label']}: {x['error_cm']:.3f}cm / {x['force_rms']:.3f}N; boundary={x['boundary']}")
lines+=['','完整结果：Issue46 / COEXIST_RESULTS.md。所有模型是特权Teacher；不是新增Student或实机验证。']
(O/'INDEX.md').write_text('\n'.join(lines)+'\n');(O/'validated.json').write_text(json.dumps(dict(status='all_three_rendered_and_fully_decoded',manifest=manifest)))
