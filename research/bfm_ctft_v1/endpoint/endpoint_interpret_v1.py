"""Summarize mechanisms after complete independent fixed endpoint evaluation."""
import csv,json,sys,statistics
from pathlib import Path
import numpy as np
folder=Path(sys.argv[1]);root=folder.parents[1]
J=lambda p:json.loads(Path(p).read_bytes())
parents=J(root/"locks/final-evaluation-manifest-v2.json")["parents"]
table={};regressions={}; splits={};matched=[]
for p in parents:
    group=p["parent_id"]; stratum=p["source"]["selected_stratum"];K=p["branches"][0]["partner"]["stiffness"]
    for name in ["B0","U","T"]:
        load=lambda b:J(folder/f"{group}-{name}-{b}.pkl.gz.metrics.json")
        long,stop,none=map(load,["continue","stop","no_force"])
        table[group,name]=dict(source=stratum,K=K,**{f"{b}_{k}":r[k] for b,r in [("long",long),("stop",stop),("none",none)] for k in ["hard","slip_result","complete_release","root_forward","wrist_progress_ratio","lag_rms","force_rms","stopped","release_yaw","release_root_path","alternating_support_changes"]})
        for b in ["continue","stop"]:
            wr=J(folder/f"{group}-{name}-{b}.pkl.gz.reward.json")
            table[group,name][f"{b}_max_reward_reconstruction_difference"]=max((v["max_reward_reconstruction_difference"] for v in wr.values()),default=None)
            for phase,v in wr.items():table[group,name][f"{b}_{phase}_wristcost"]=v["mean_response_cost"]
for i,n in [("U","B0"),("T","B0"),("T","U")]:
    for g in ["all","early_static","transition_near","post_transition_actual_moving","K40","K80"]:
        pairs=[(table[p["parent_id"],i],table[p["parent_id"],n]) for p in parents if g=="all" or p["source"]["selected_stratum"]==g or f'K{int(p["branches"][0]["partner"]["stiffness"])}'==g]
        vals={}
        for k in ["long_root_forward","none_root_forward","long_lag_rms","long_force_rms","stop_release_yaw","stop_release_root_path"]:
            differences=[a[k]-b[k] for a,b in pairs if a[k] is not None and b[k] is not None and all(r[("stop" if k.startswith("stop") else "none" if k.startswith("none") else "long")+"_complete_release"] for r in [a,b])]
            if differences:vals[k]=dict(n=len(differences),median_change=float(np.median(differences)),mean_change=float(np.mean(differences)))
        splits[f"{i}-{n}/{g}"]=vals
for p in parents:
    for name in ["B0","U","T"]:
        row=table[p["parent_id"],name]
        matched.append(dict(parent=p["parent_id"],model=name,**row))
with (folder/"paired-mechanism-and-reward.csv").open("w") as f:
    keys=sorted(set().union(*(r.keys() for r in matched)));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(matched)
(folder/"mechanism-summary.json").write_text(json.dumps(splits,indent=2))
