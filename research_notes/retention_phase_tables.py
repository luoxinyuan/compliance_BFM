"""Format saved phase metrics only; no simulation or model evaluation."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);e=json.loads(p.read_text());selected=e['selection']['selected']['iteration'];last=e['selection']['final_iteration']
# Evaluator role spelling is preserved in evidence; select new rows by checkpoint basename.
rows=e['retrospective_rows']+[r for r in e['new_development'] if Path(r['model_path']).name in {f'candidate-{selected}.pt',f'candidate-{last}.pt'}]
def vec(x):return '—' if x is None else '/'.join(f'{v:.3f}' for v in x)
s=['# 固定物理评价：完整分阶段表','', '全部旧模型为retrospective；5403为固定开发评价，不是新封存。XYZ均在世界系。误差只在实际接触有效时计算，释放后的旧anchor误差不作为任务指标。yaw为该段末pelvis角度；滑移是有载荷足点的运动代理，不是实机测量。q_PD峰值与真实5ms关节变化分别报告。','']
for case in ['short08','normal72','stop66','none']:
 s += [f'## {case}','', '|模型|阶段/时间s|XYZ误差RMS cm|有符号XYZ误差cm|XYZ力RMS N|末yaw°|滑移p95/峰 m/s|q_PD/实际q峰°|','|---|---|---|---|---|---|---|---|']
 for r in rows:
  if r['case']!=case:continue
  for ph,q in r['quality'].items():
   s.append(f"|{r['role']}|{ph} {q['start']:.2f}–{q['end']:.2f}|{vec(q['error_xyz_rms_cm'])}|{vec(q['error_xyz_signed_cm'])}|{vec(q['force_xyz_rms'])}|{q['yaw_deg_final']:.2f}|{q['slip_p95_mps']:.4f}/{q['slip_peak_mps']:.4f}|{q['qpd_step_peak_deg']:.2f}/{q['actual_joint_5ms_peak_deg']:.2f}|")
 s.append('')
Path(sys.argv[2]).write_text('\n'.join(s)+'\n')
