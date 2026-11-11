import json,hashlib,shutil,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);S=p.parse_args().stage; R=S/'publication/research_notes'
f=R/'F1_STATE_AND_SEARCH_COVERAGE.md';t=f.read_text();t=t.replace('# F1状态与信息覆盖（L0续验完成）','# F1状态与信息覆盖（L0–L2完成）');t=t.replace('L1/L2/L3为0。','该句原指L0报告时点；当前L1/L2实际覆盖见下节，L3仍为0。');t=t.replace('旧C-P因果预测模块未在本L0完成新F1实现/资格。','L0时尚无新C-P资格；L2现已实现当前状态常速预测并完成8次未来spec干预和真实共同前缀核验。');t+='''

## L1/L2最终覆盖

L1使用锁定主清单全部24状态/12父组，每状态16个连续3s控制，共384分支。保留所有成功、失败及各自runtime；两次恢复错误与剩余状态续跑见F1_ERRATUM。16候选的精确节点、phase定位和来源见F1_L1_L2_METHODS及f1_l1_l2_protocol.json，不按成功结果换池。三原生参考仅供源/恢复资格，不计接触成功率。

L2固定使用P1-1、P1-13、tail-sustained-5c67、tail-sustained-97ff、tail-hold-97ff、near-sustained-5f62、stop-sustained-5f63，最后一项加同前缀continue分支：六父组、七初态、八窗口。每窗3s，固定8候选（原16池索引0/2/4/5/6/7/9/10）、2s预测，每100ms执行首动作后重规划，240查询。C-S为冻结BC500真实执行，C-P为因果当前特权搜索；均独立恢复同一完整初态。L1为真实未来程序O-F，不能与L2作单一信息变量的因果比较。

别名组未构造，C-S planner缺格；没有借真实隐藏状态模拟器冒充受限预测器。真实stop/continue分叉前100个5ms tick、5个高层命令逐值一致；分叉后真实force/anchor于5.505s起不同，C-S受限输入与命令于5.6s开始不同。兄弟分支只计同一父组。权限明细见f1_info_access_audit.jsonl，实际首命令/首BFM子步240次对齐见f1_query_alignment.json。

旧state manifest及source plan内容/hash完全不改，未读封存。L1/L2快照、proposal全序列、预测/真实执行、所有失败保留于独立stage；L3未执行，3h未挪用。
''';f.write_text(t)
manifest=json.loads((S/'runs/video-01/manifest.json').read_text());clean=[]
for x in manifest:
 clean.append({k:v for k,v in x.items() if k not in ('path','sides','condition')}|{'filename':Path(x['path']).name,'sides':[{'title':y['title'],'artifact':Path(y['path']).name,'sha256':y['sha256']} for y in x['sides']]})
(R/'f1_video_manifest.json').write_text(json.dumps(clean,indent=2)+'\n')
(R/'F1_VIDEO_INDEX.md').write_text('''# F1真实执行视频

Mac：`~/Projects/compliance-bfm-diagnostics/videos/f1-l1-l2-20261111/`。三条均已同步、校验SHA，完整解码；25fps、75帧、1×真实qpos，完整3s诊断窗。没有将候选预测、reference或q_PD当真实运动；不声称完整回合或实时计算。每次100ms控制的离线搜索中位耗时17.75s。

|文件|上下画面|解释|
|---|---|---|
|f1-p1-13-feedback.mp4|固定BC500 / 因果特权重规划|同状态同条件，跟随和力小幅改善，yaw略增；两者均过窗口参考|
|f1-stop-continue.mp4|停止需求 / 继续需求，均同一因果planner|同前缀后真实分叉；停止有效，继续分支跟随与滑移失配。两行不是相同需求，不作为同条件优劣比较|
|f1-tail-hold-release.mp4|固定BC500 / 因果特权重规划|同状态同条件，减力明显，但yaw/滑移代价保留；覆盖保持到释放，释放后不考核旧anchor|

每行自身左右视角为同一条实际轨迹，字幕含时刻、阶段、真实XYZ力、接触误差及pelvis/torso yaw；黄箭头为力。中帧图已人工查看。SHA与源轨迹身份见[f1_video_manifest](f1_video_manifest.json)。不剪除窗口内失败，不拼不同checkpoint为统一策略。
''')
files=['f1_phase_reduction.py','f1_query_reduction.py','f1_slip_reduction.py','f1_render_job.py','f1_assemble.py','f1_scientific_tables.py','f1_narrative.py','f1_publication_finish.py']
hashes={}
for n in files:
 src=S/'prep'/n;shutil.copy2(src,R/n);hashes[n]=hashlib.sha256(src.read_bytes()).hexdigest()
shutil.copy2(S/'prep/report_f1.py',R/'f1_report_reduction.py');hashes['f1_report_reduction.py']=hashlib.sha256((S/'prep/report_f1.py').read_bytes()).hexdigest()
hashes['archived_l1_slip_reducer']=hashlib.sha256((S/'prep/f1_slip_reduction_l1_archived.py').read_bytes()).hexdigest()
(R/'f1_postprocess_identities.json').write_text(json.dumps(hashes,indent=2)+'\n')
