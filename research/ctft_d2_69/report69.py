from pathlib import Path
import csv,json,statistics as st
s=Path.cwd();r=list(csv.DictReader((s/'pair24.csv').open()));assert len(r)==24
a=[v for v in r if v['series'].endswith('continue')];stop=[v for v in r if v['series'].endswith('-stop')];nf=[v for v in r if v['series'].endswith('no_force')];matches=[v for v in a if v['matched']=='True']
num=lambda v,k:float(v[k])
def values(v,k):return [num(v,i+'_'+k) for i in 'AB']
def med(group,k):return [st.median(g) for g in zip(*(values(v,k) for v in group))]
def pp(vals,f=1):return ' / '.join(f'{x*f:.3f}' for x in vals)
def count(rs,k):return [sum(v[i+'_'+k]=='True' for v in rs) for i in 'AB']
def better(v):return num(v,'B_two_s_p95')<=num(v,'A_two_s_p95') and num(v,'B_two_s_path')<=num(v,'A_two_s_path')
def improved(v):
    u,j=values(v,'two_s_p95');p,q=values(v,'two_s_path')
    return q<=p and (j<=.7*u or ((u>.05 or p>.02) and j<=.05 and q<=.02))
def d(v,k):
    x,y=values(v,k);return y-x
good=sum(better(v) for v in matches);better_count=sum(improved(v) for v in matches)
fmean=lambda k:st.median(num(v,k) for v in matches)
def no_worse(k):
    x,y=med(matches,k);return y<=1.1*x
passed=len(matches)>=6 and good*6>=5*len(matches) and better_count*6>=4*len(matches) and all(no_worse(k) for k in ['two_s_force','two_s_lag_rms_m'])
status='UNMATCHED/UNDERPOWERED' if len(matches)<6 else 'MOTION_STRUCTURE_CANDIDATE' if passed else 'NO_MATCHED_QUALITY_GAIN'
g=json.loads((s/'gradient.json').read_bytes());cost=json.loads((s/'d2/cost.json').read_bytes())
c=json.loads((s/'d2/command.json').read_bytes())
no=[v['series'] for v in nf if d(v,'root_path')>.05 or d(v,'yaw_abs_deg')>10]
rel=count(stop,'release_pass');ss=count(stop,'stopped')
oldgate=[sum(v[i+'_slip_result']=='PASS' for v in r) for i in 'AB']
hdr=f"""# Issue 69 冻结 U，单次小幅命令对照

**{status}。** 8 个独立训练祖先、24 个条件对，standing 对照 A 使用 #68 已逐 tick 复验的 24 条原样参考，本轮 B 使用同一冻结 U，首 1 s 给定合法球面比例 0.15，随后回 standing。实际大圆全角 {c['full_angle_rad']*180/3.14159265:.2f}°，干预角为其 15%，输出范数 {c['norm']:.6f}。同一干预也给从未施力分支。人工给命令只能检验运动结构和需求撤销风险。

## D0 存档审计

|同一时基的证据|实际结果|可辨范围|
|---|---|---|
|8 个持续分支前 2 s，每源先求时间均值再取中位|引导轴 −1.37 N，竖直 −5.23 N；腕高于 anchor 11.6 cm；8/8 非前向主导|按 ≥1 N tick，正向轴分量为最大绝对分量的 tick 须过半才记前向主导。固定水平 anchor 路径产生三维反馈。|
|首次不合格承重段 / 合格段|不合格段 8/8 非前向主导，合格段 7/8。旧证据中 5/8 首败由累计路程触发；6/8 坏段原生滑移代价较高|同状态比较没有充分证明代价符号错误。stance-support.csv 和按原始 tick 索引的 npz 保留双脚载荷、各次承重出入、动作、qPD、q/dq 与逐项回报。|
|仍连接的停止 / 已释放 / 从未施力|停止分支保持段引导轴 +1.60 N、竖直 −6.38 N（跨源中位）；真正释放与无力为 0 N|仅在连接时计算任务 lag；释放后用身体路径、偏航。|
|g=0 原生保持|8 个相同源中 U 在 00、02、08 三个原生回合硬停；逐源身份及原因见 g0-native.json|独立原生继续执行，不能并入 g=1 的无力或释放得分。|
|终点 U 在同批历史真实训练访问状态的三个加权损失梯度，冻结均值、无更新|32 行，动作范数 FB/CPR/aux = {pp(g['action']['norm'])}；Actor 参数范数 = {pp(g['parameters']['norm'])}。参数余弦 FB–CPR / FB–aux / CPR–aux = {pp([g['parameters']['cosine'][0][1],g['parameters']['cosine'][0][2],g['parameters']['cosine'][1][2]])}|固定真实历史训练输入、当前均值处负价值的局部导数。不能还原某次随机训练梯度或唯一拆分聚合 aux 的滑移梯度。非优化步骤；批与源 SHA 已存。|

## D1 同进度承重质量

匹配 {len(matches)}/8（最低 6）。匹配组两指标同时不劣 {good}/{len(matches)}，累计路程不劣且切速至少降30%或真正转合格 {better_count}/{len(matches)}。全程硬停 A/B = {count(r,'hard')}；按 #67 原整回合足底质量门通过 A/B = {oldgate} / 各24。下表是全部持续回合头2秒，不能用全组中位替代匹配祖先判读。

|祖先|root A/B cm|腕 A/B cm|最差材料点 p95 A/B m/s|最大累计路程 A/B cm|推进匹配|
|---|---:|---:|---:|---:|---|
"""
for v in a:
    hdr+=f"|{v['series'][:2]}|{pp(values(v,'two_s_root'),100)}|{pp(values(v,'two_s_wrist'),100)}|{pp(values(v,'two_s_p95'))}|{pp(values(v,'two_s_path'),100)}|{v['matched']}|\n"
hdr+=f"""
## 需求条件化与限制

持续牵引全接触窗力 RMS 中位 A/B {pp(med(a,'force_rms'))} N，滞后 {pp(med(a,'lag_rms'),100)} cm。提前停止 2 s 收步通过 {ss[0]}/8 与 {ss[1]}/8；释放4 s 恢复通过 {rel[0]}/8 与 {rel[1]}/8。停止分支恢复路程中位 {pp(med(stop,'release_root_path'),100)} cm、偏航 {pp(med(stop,'release_yaw'))}°。全程无力额外路径 >5 cm 或最大连续偏航额外 >10° 的祖先 {len(no)}/8，具体为 {no}。无力路径中位 {pp(med(nf,'root_path'),100)} cm。逐源逐分支的数据在 pair24.csv，含腕高、三轴力及旧门；完整阶段和承重段在 stance-phase.json。每一组以祖先为推论单位。

## 账本与复核

新执行 {cost['branches']}、全流程 {cost['ticks']} 个 5 ms tick；主作业入口计时 {cost['seconds']:.1f} s。模型、命令、输入、源码、轨迹、共同前缀及分项成本见伴随身份清单。没有参数更新、优化器步骤、额外取角或封存评价。梯度批与物理支撑、不可公开的完整轨迹留在独立运行目录。
"""
(s/'CTFT_D2_69_RESULTS.md').write_text(hdr)
