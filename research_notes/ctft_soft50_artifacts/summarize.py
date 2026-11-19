"""Read completed actual execution only; no policy calls."""
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
read = lambda name: json.loads((run / name).read_text())
base = read("baseline-measured.json")
soft = read("soft-measured.json") if (run / "soft-measured.json").exists() else None
rows = []
for phase, key, description in [
    ("early","v95","前0.5秒水平腕速p95 m/s"),
    ("early","force","前0.5秒力 RMS N"),
    ("early","err","前0.5秒误差 RMS cm"),
    ("early","wrist_lead_cm","前0.5秒最大前向超前 cm"),
    ("recoil","world","首次单支撑前手腕世界坐标最大回撤 cm"),
    ("recoil","relative","首次单支撑前腕相对肩最大回撤 cm"),
    ("move","root_cm","移动段 root 前进 cm"),
    ("move","wrist_cm","移动段腕前进 cm"),
    ("move","err","移动段误差 RMS cm"),
    ("move","force","移动段力 RMS N"),
    ("hold","err","保持段误差 RMS cm"),
    ("hold","force","保持段力 RMS N"),
    ("release","root_path_cm","释放后4秒 root 路程 cm"),
    ("release","yaw_change","释放后4秒骨盆净转角 deg")]:
    vals = [x.get(phase, {}).get(key) if x else None for x in (base, soft)]
    rows.append([description, *vals])
for key, description in [("first_single","首次持续单支撑距 onset s"),("exchange","推进中持续相反单支撑距 onset s")]:
    rows.append([description,base.get(key),soft.get(key) if soft else None])
for k, description in [("p95","承重段最大 p95 m/s"),("path","承重段最大材料点行程 m")]:
    rows.append([description,*[max((v[k] for v in x["stances"] if v.get(k) is not None),default=None) if x else None for x in (base,soft)]])
fmt = lambda x: "—" if x is None else f"{x:.4f}"
print("| 实际物理量 | 原版 | 前五条半幅 |\n|---|---:|---:|")
for label, a, b in rows:
    print(f"| {label} | {fmt(a)} | {fmt(b)} |")
if soft:
    tests = {
        "早期腕速降低":soft["early"]["v95"]<base["early"]["v95"],
        "早期作用力降低":soft["early"]["force"]<base["early"]["force"],
        "移动段推进保持":all(abs(soft["move"][k]-base["move"][k])<=5 for k in ("root_cm","wrist_cm")),
        "持续支撑交换":soft["exchange"] is not None,
        "承重质量":soft["slip_pass"],"无硬界":soft["hard"] is None,
        "保持段负担":soft["hold"]["force"]<=base["hold"]["force"]+.5 and soft["hold"]["err"]<=base["hold"]["err"]+1}
    print("\n"+"；".join(k+" "+("通过" if v else "未通过") for k,v in tests.items()))
