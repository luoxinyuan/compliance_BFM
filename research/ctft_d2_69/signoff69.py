from pathlib import Path
import json,statistics,csv
s=Path.cwd();p=s/'CTFT_D2_69_RESULTS.md';text=p.read_text()
rows=list(csv.DictReader((s/'pair24.csv').open()))
locked=json.loads((s/'denominator.json').read_text());outcome=text.split('**')[1]
if len(locked)<6:
    decision='优先另案验证伙伴竖直约束：在同一冻结 U、同一站立命令、同一水平路径和完整真实起点下，减小伙伴对竖直误差的约束能否产生持续向前的实际提示，同时保住停止和无力稳定性？需独立授权，本轮没有改变接触物理。'
elif outcome=='MOTION_STRUCTURE_CANDIDATE':
    decision='建议另案仅验证冻结 Actor 的一维需求调节及撤销。物理结果限于本伙伴几何，不能宣称已学到自主触发。'
else:
    decision='停止本 α 配方；优先另案检验同一冻结 U 在改变伙伴竖直顺从性后，实际前向提示与稳步的关系。没有充分的代理错排证据支持直接改滑移奖励。'
ledger=json.loads((s/'ledger69.json').read_text())
tag='## 唯一下一项决策'
assert tag not in text
text+=f"""
{tag}

{decision}

D0 的逐时力与姿态结果中，首两秒 8 个原 standing 持续分支在总力 ≥1 N 时，正引导轴分量为三轴中最大绝对分量的 tick 比例均为零。各源时间平均竖直力有 5 组向下、3 组向上；非前向主导不等于都向下。本指标只描述已执行的输入与后果。

物理计数包含初始化，实际动作只在训练来源的冻结评价中执行。原封存保持未读。原始文件与运行源码 SHA、共同祖先逐值前缀和无力实际为零的核对在 ledger69.json 和 verification.json。D0 CPU 主体 {ledger['D0_archive_body_s']:.1f}s，梯度主体 {ledger['D0_grad_body_s']:.1f}s；验证/表格/视频子作业 {ledger['postprocess_subjob_s']:.1f}s。包含启动、辅助检查、等待与交付的保守累计包络 {ledger['total_job_upper_s']:.1f}s，输出盘点 {ledger['raw_output_bytes_at_inventory']/2**20:.1f} MiB，不含重复副本；本地与个人分支另留 256 MiB 包络。
"""
p.write_text(text)
