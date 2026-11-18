"""Aggregate immutable physical endpoints with parent-paired uncertainty."""
import csv,json,sys,time
from pathlib import Path
import numpy as np
from score_endpoint_v1 import ROOT,write

def interval(x):
    a=np.asarray(x,dtype=float)
    rng=np.random.default_rng(681117)
    boots=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())

def main(folder):
    manifest=json.loads((ROOT.parent/"locks/final-evaluation-manifest-v2.json").read_bytes())
    strata={p["parent_id"]:p["source"]["selected_stratum"] for p in manifest["parents"]}
    ms=sorted(folder.glob("*.pkl.gz.metrics.json"));assert len(ms)==252
    allrows=[];lookup={}
    for p in ms:
        r=json.loads(p.read_bytes())
        key=(r["parent"],r["model"],r["branch"])
        assert key not in lookup
        lookup[key]=r
        c=r["branch"]!="native_continuation"
        r["stratum"]=strata.get(r["parent"],"native_regression")
        if c:
            r["task_met"]=bool(r["complete_release"] and not r["hard"] and r["wrist_progress_ratio"]>=.8 and r["force_rms"]<=7 and r["lag_rms"]<=.20 and r["root_forward"]>=.2 and r["alternating_support_changes"]>=2 and r["stopped"] and r["release_root_path"]<=.15 and r["release_yaw"]<=15)
            r["demonstrated_joint_success"]=r["task_met"] and r["slip_result"]=="PASS"
        allrows.append({k:v for k,v in r.items() if k not in ("stages","stances","hard_reason")})
    with (folder/"endpoint-conditions.csv").open("w") as f:
        keys=sorted(set().union(*(r.keys() for r in allrows)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
    summary={}; pairs={}; reward={}
    for stratum in ["all","early_static","transition_near","post_transition_actual_moving"]:
        parents=[p for p,s in strata.items() if stratum=="all" or s==stratum]
        for name in ["B0","U","T"]:
            for b in ["continue","stop","no_force"]:
                rs=[lookup[p,name,b] for p in parents]
                summary[f"{stratum}/{name}/{b}"]=dict(n=len(rs),hard=sum(r["hard"] for r in rs),demonstrated=sum(r["demonstrated_joint_success"] for r in rs),task_met=sum(r["task_met"] for r in rs),unmeasured=sum(r["slip_result"]=="UNMEASURED" for r in rs),slip_pass=sum(r["slip_result"]=="PASS" for r in rs),complete=sum(r["complete_release"] for r in rs),median_force=float(np.median([r["force_rms"] for r in rs])),median_lag=float(np.median([r["lag_rms"] for r in rs])),median_root=float(np.median([r["root_forward"] for r in rs])),median_wrist_ratio=float(np.median([r["wrist_progress_ratio"] for r in rs])),stopped=sum(r["stopped"] for r in rs))
        for base,new in [("B0","U"),("B0","T"),("U","T")]:
            def delta(metric,b):
                return [float(lookup[p,new,b][metric])-float(lookup[p,base,b][metric]) for p in parents]
            key=f"{stratum}/{new}-{base}"
            pairs[key]={f"{b}/{m}":interval(delta(m,b)) for b in ["continue","stop","no_force"] for m in ["force_rms","lag_rms","root_forward","demonstrated_joint_success"]}
            for b in ["continue","stop"]:
                counts=dict(eligible=0,improved_response=0,also_better_burden_lag=0,tradeoffs=[],excluded=[])
                for p in parents:
                    rs=[lookup[p,i,b] for i in (base,new)]
                    if not all(r["complete_release"] and not r["hard"] for r in rs):
                        counts["excluded"].append(p);continue
                    vals=[json.loads((folder/f"{p}-{i}-{b}.pkl.gz.reward.json").read_bytes())["moving"] for i in (base,new)]
                    if any(v["max_reward_reconstruction_difference"]>1e-6 for v in vals):counts["excluded"].append(p);continue
                    counts["eligible"]+=1
                    if vals[1]["mean_response_cost"]<vals[0]["mean_response_cost"]:
                        counts["improved_response"]+=1
                        if rs[1]["lag_rms"]<=rs[0]["lag_rms"] and rs[1]["force_rms"]<=rs[0]["force_rms"]:counts["also_better_burden_lag"]+=1
                        else:counts["tradeoffs"].append(p)
                reward[f"{key}/{b}"]=counts
    native={a:dict(hard=[p for p,n,b in lookup if b=="native_continuation" and n==a and lookup[p,n,b]["hard"]]) for a in ("B0","U","T")}
    write(folder/"paired-summary.json",dict(families=summary,paired=pairs,native=native,reward=reward,partial_episodes="Force and lag include observed suffix only. Hard stops fail; report separately."))
if __name__=="__main__":main(Path(sys.argv[1]))
