"""Generate the fixed-endpoint report from complete paired tables and the resource ledger."""
from pathlib import Path
import json,sys,csv
import numpy as np
J=lambda p:json.loads(p.read_bytes())
folder=Path(sys.argv[1]);stage=folder.parents[1]
s=J(folder/"paired-summary.json");g=J(folder/"endpoint-gates-and-retention.json")
manifest=J(stage/"locks/final-evaluation-manifest-v2.json")
ps=J(folder/"progress.json")
allmetrics=[J(f) for f in folder.glob("*.pkl.gz.metrics.json")];assert len(allmetrics)==252
by={(r["parent"],r["model"],r["branch"]):r for r in allmetrics}
strata=["early_static","transition_near","post_transition_actual_moving"]
r=["# BFM-CTFT U/T 固定终点评价","", "单 seed；24 个适配留出祖先，每祖先 continue/stop/no-force；另含 12 个训练谱系原生回归。原始 B0 和两个固定终点均按确定性均值执行。共 252 个完整评价条目，遇真实硬停止保留至失败 tick。","", "## 持续牵引的完整条件","", "力与滞后列为各父组主动窗口 RMS 的中位数，硬停止回合只报告实际观察窗口。身体前进按引导轴投影。通过数同时要求完整恢复及所有任务、滑移、支撑与安全条件。","", "|源/模型|通过/父组|力 N|滞后 cm|身体前进 cm|停止达标|滑移达标|滑移缺测|硬停|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for st in ["all",*strata]:
    for a in ("B0","U","T"):
        x=s["families"][f"{st}/{a}/continue"];r.append(f'|{st} / {a}|{x["demonstrated"]}/{x["n"]}|{x["median_force"]:.2f}|{100*x["median_lag"]:.1f}|{100*x["median_root"]:.1f}|{x["stopped"]}|{x["slip_pass"]}|{x["unmeasured"]}|{x["hard"]}|')
r+=["","## 停止和原生回归","","|模型|stop 两秒内减速并维持|stop 释放恢复路程和偏航均达标|stop 硬停|原生硬停|相对 B0 新增硬停父组|共同完整运动参考窗|模型/B0 关节跟踪 RMSE 中位数(rad)|","|---|---:|---:|---:|---:|---|---:|---:|"]
for a in ("B0","U","T"):
    q=g["branch_component_pass"][f"{a}/stop"];x=g["native"][a];hard=s["families"][f"all/{a}/stop"]["hard"]
    r.append(f'|{a}|{q["stopped"]}/24|{q["release"]}/24|{hard}|{x["hard"]}/12|{", ".join(x["new_hard"]) or "无"}|{x["tracking_complete_pairs"]}|{x["paired_window_median_rmse"]:.4f} / {x["paired_b0_median_rmse"]:.4f}|')
r+=["","原生跟踪采用原生 motion 实际持续段对应的 50 Hz 参考关节角，仅在双方均有完整持续段的父组比较。进入、退出或恢复发生的硬终止仍在全回合失败计数。12 个源均来自已有训练运动谱系。","", "## 同初态接触效应与配对不确定度","","下表为 12 个 early_static 父组的持续牵引及同起点无力分支，硬停另列；不把无力时自身前移归给接触。","", "|模型|有力/无力 root 前进中位数 cm|有力/无力硬停|","|---|---:|---:|"]
for a in ("B0","U","T"):
    c,n=(s["families"][f"early_static/{a}/{b}"] for b in ("continue","no_force"))
    r.append(f'|{a}|{100*c["median_root"]:.1f} / {100*n["median_root"]:.1f}|{c["hard"]} / {n["hard"]}|')
r+=["","按 24 个祖先配对 bootstrap（10,000 次，表中为均值差及 95% 区间）。误差与力按各自实测窗口计算，硬终止率单列。","","|对比|力差 N|滞后差 cm|联合成功率差 pp|","|---|---:|---:|---:|"]
for k in ("U-B0","T-B0","T-U"):
    vals=[]
    for m,scale in [("force_rms",1),("lag_rms",100),("demonstrated_joint_success",100)]:
        x=s["paired"][f"all/{k}"][f"continue/{m}"]
        vals.append(f'{scale*x["mean"]:.2f} [{scale*x["ci95"][0]:.2f}, {scale*x["ci95"][1]:.2f}]')
    r.append("|"+k+"|"+"|".join(vals)+"|")
r+=["","## 奖励与真实任务质量","","比较双方都完成的父组。先检查存档物理量是否能重构正式 contact reward，再统计移动阶段腕速成本下降的配对。双降指整段接触的实际力与 RMS 滞后都不升。停止、足底与原生质量仍按各自物理门判。","","|比较|分支|可比父组|腕速成本改善|其中实际力和滞后双降|取舍父组|","|---|---|---:|---:|---:|---|"]
for k in ("U-B0","T-B0","T-U"):
    for b in ("continue","stop"):
        x=s["reward"][f"all/{k}/{b}"];r.append(f'|{k}|{b}|{x["eligible"]}|{x["improved_response"]}|{x["also_better_burden_lag"]}|{", ".join(x["tradeoffs"]) or "无"}|')
r+=["","## 正确性与限制","","正式端点各 131,072 行、1,024 组更新；两臂共享 8,192 行只计一次。首轮评价的 113 个产物全部隔离。修正版 B0 在全部 12 条原生轨迹的 qpos、qvel、腕力、足法向力与终止时刻逐值复现。实际 stop/continue 共同前缀逐值校验，全部失败保留。","","这些配对只支持当前腕部水平引导、当前伙伴速度与源分布。奖励项重构和物理门判断可描述取舍；均未操纵 standing z、力上限、其他奖励权重或梯度，因此不能确定这些因素造成了差异。",""]
(folder/"BFM_CTFT_FIXED_ENDPOINT_RESULTS.md").write_text("\n".join(r))
