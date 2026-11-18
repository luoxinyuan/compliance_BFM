"""Produce group-level numerical tables only after the locked 252 rollout files exist."""
import csv,json,sys
from pathlib import Path
import numpy as np
folder=Path(sys.argv[1])
summary=json.loads((folder/"paired-summary.json").read_bytes())
metrics={}
for p in folder.glob("*.metrics.json"):
    r=json.loads(p.read_bytes())
    metrics[r["parent"],r["model"],r["branch"]]=r
assert len(metrics)==252
parents=sorted({p for p,a,b in metrics if b=="continue"})
def interval(arr):
    a=np.asarray(arr);idx=np.random.default_rng(681117).integers(0,len(a),(10000,len(a)))
    return [float(a.mean()),*np.quantile(a[idx].mean(1),[.025,.975]).tolist()]
details={};gates={}
for a in ("B0","U","T"):
    for b in ("stop","continue","no_force"):
        rs=[metrics[p,a,b] for p in parents]
        flags=[dict(complete=r["complete_release"] and not r["hard"],
             wrist_progress=r["wrist_progress_ratio"]>=.8,force=r["force_rms"]<=7,lag=r["lag_rms"]<=.2,
             body=r["root_forward"]>=.2,steps=r["alternating_support_changes"]>=2,
             stopped=r["stopped"],release=(r["complete_release"] and r["release_yaw"]<=15 and r["release_root_path"]<=.15),
             slip=r["slip_result"]=="PASS") for r in rs]
        gates[a,b]={k:sum(f[k] for f in flags) for k in flags[0]}
        for p,r,f in zip(parents,rs,flags):
            details[p,a,b]=dict(**f,all_pass=all(f.values()),phys_hard=r["hard"],slip_status=r["slip_result"])
def speeds_compare(new,base):
    stable=0;pareto=0;important=[]
    for p in parents:
        l=metrics[p,base,"continue"];r=metrics[p,new,"continue"]
        if r["complete_release"] and l["complete_release"]:
            stable+=1
            if r["lag_rms"]<l["lag_rms"] and r["force_rms"]<l["force_rms"]:pareto+=1
        if (r["hard"],r["slip_result"],r["stopped"])!=(l["hard"],l["slip_result"],l["stopped"]):important.append(p)
    return dict(complete_pairs=stable,both_lag_and_force_improved=pareto,quality_differences=important)
quality={f"{new}-{base}":speeds_compare(new,base) for base,new in [("B0","U"),("B0","T"),("U","T")]}
native=json.loads((folder/"native-motion-retention.json").read_bytes())["rows"]
native_report={}
for a in ("B0","U","T"):
    rows=[r for r in native if r["model"]==a]
    base={r["parent"]:r for r in native if r["model"]=="B0"}
    common=[r for r in rows if r["complete"] and base[r["parent"]]["complete"]]
    native_report[a]=dict(hard=sum(r["hard"] for r in rows),new_hard=[r["parent"] for r in rows if r["hard"] and not base[r["parent"]]["hard"]],tracking_complete_pairs=len(common),paired_window_median_rmse=float(np.median([r["reference_q_rmse_rad"] for r in common])),paired_b0_median_rmse=float(np.median([base[r["parent"]]["reference_q_rmse_rad"] for r in common])))
# All task quality values and denominators are kept even when a gate fails.
write=folder/"endpoint-gates-and-retention.json"
write.write_text(json.dumps(dict(branch_component_pass={f"{a}/{b}":v for (a,b),v in gates.items()},native=native_report,paired=quality),indent=2))
with (folder/"endpoint-parent-gates.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=["parent","model","branch",*next(iter(details.values())).keys()]);w.writeheader()
    for (p,a,b),r in details.items():w.writerow(dict(parent=p,model=a,branch=b,**r))
