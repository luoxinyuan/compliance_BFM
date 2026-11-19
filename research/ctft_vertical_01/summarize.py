import csv,json,statistics as st
from pathlib import Path
rows=list(csv.DictReader(open('paired.csv')));cases=[0,1,2,4,5,6,7,8]
data={(r['group'],int(r['case']),r['branch']):r for r in rows}
med=st.median
def v(g,i,b,k):return float(data[g,i,b][k])
def diff(g,i,k,b='continue'):return v(g,i,b,k)-v(g,i,'no_force',k)
def arr(g,k,b):return [v(g,i,b,k) for i in cases]
def col(k,b,unit=1):return ' | '.join(f'{unit*med(arr(g,k,b)):.2f}' for g in ['reference','soft'])
def delta(k,unit=1):return ' | '.join(f'{unit*med([diff(g,i,k) for i in cases]):.2f}' for g in ['reference','soft'])
def count(k,b):return ' | '.join(str(sum(data[g,i,b][k]=='True' for i in cases))+'/8' for g in ['reference','soft'])
lines=['# 冻结 U：伙伴竖直顺从性配对诊断','', '8 个已暴露训练祖先。A 为逐 tick 验证过的原伙伴；B 仅把限幅前世界 Z 弹簧和阻尼合力乘 0.25。固定 standing 命令，同一路径、模型、PD、水平 K/D 与 10 N 总上限。各表按祖先先计算，再取中位数。动作和物理均无更新训练。','','|持续牵引实际提示/身体量|原伙伴|竖直软化|','|---|---:|---:|']
for k,name,u in [('early_fwd','前 2 s 引导轴平均力/N',1),('early_lat','前 2 s 横向平均力/N',1),('early_up','前 2 s 世界竖直平均力/N',1),('early_frac','前 2 s 正向主导占有载 tick/%',100),('moving_frac','移动期正向主导占有载 tick/%',100),('early_height','前 2 s 腕高于 anchor/cm',100),('early_sat','前 2 s 总力饱和/%',100),('force_rms','全接触窗实际力 RMS/N',1),('lag_rms','全接触窗三维滞后 RMS/cm',100),('two_root','2 s root 沿轴推进/cm',100),('two_wrist','2 s 腕沿轴推进/cm',100),('two_p95','2 s 最差承重段材料点 p95/cm/s',100),('two_path','2 s 最大承重材料点路程/cm',100),('alternating_support_changes','接触期左右支撑转换次数',1)]:
 lines.append(f'|{name}|{col(k,"continue",u)}|')
lines+=['','|配对腕肩净响应（有力减同源无力）|原伙伴|竖直软化|','|---|---:|---:|']
for k,n,u in [('half_reach','0.5s 腕相对肩前伸/cm',100),('two_reach','2s 腕相对肩前伸/cm',100),('two_elbow','2s 右肘角差/°',1),('two_shoulder','2s 右肩 pitch 差/°',1),('two_heightdiff','2s 左右腕高度差的有力增量/cm',100)]:
 lines.append(f'|{n}|{delta(k,u)}|')
lines+=['','|停止与释放（提前停止分支）|原伙伴|竖直软化|','|---|---:|---:|',f"|停止通过|{count('stopped','stop')}|",f"|恢复路程/cm|{col('release_root_path','stop',100)}|",f"|恢复偏航/°|{col('release_yaw','stop')}|"]
for g in ['reference','soft']:
 inds={k:sum(x>.5 for x in arr(g,k,'continue')) for k in ['early_frac','moving_frac']}
 lines.append(f"\n{g}：正向主导祖先数（≥1 N 的 tick 中正前分量过半占优），头两秒 {inds['early_frac']}/8、移动期 {inds['moving_frac']}/8；24 条中机械硬停 {sum(r['hard']=='True' for r in rows if r['group']==g)}；原全回合足底门通过 {sum(r['slip_result']=='PASS' for r in rows if r['group']==g)}/24。")
lines+=['','完整无力动作、物理记录须逐值通过，提前停止和继续的共同前缀须一致。冻结控制器的有力/无力差同时包含力直接作用和策略反馈。腕肩前伸按躯干前轴投影；力与 root/腕推进按实际伙伴方向。三维力减小受总力重新分配影响，单独降力不能证明策略改善。','', '逐源时序与身份见 paired.csv、arm_curves.csv.gz、checks.json、soft/model-identities.json 和 source.lock。']
Path('RESULTS.md').write_text('\n'.join(lines)+'\n')
