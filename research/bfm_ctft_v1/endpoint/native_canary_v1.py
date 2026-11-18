import gzip,json,pickle,sys,time
from pathlib import Path
import numpy as np
from score_endpoint_v1 import ROOT,write
from source_evidence_readonly import read
stage=ROOT.parent;out=Path(sys.argv[1]);start=time.monotonic();records=[]
for cell in json.loads((stage/"locks/final-evaluation-manifest-v2.json").read_bytes())["native_g0_evaluation"]:
    with gzip.open(out/(cell["parent_id"]+"-B0-native_continuation.pkl.gz"),"rb") as f: actual=pickle.load(f)["physics"]
    exp=read(stage/"runs/s0-native-sources-03"/cell["source"]["evidence_file"])["physics"][4*cell["source_control"]:]
    n=min(len(exp),len(actual));diff={}
    for key in ("qpos","qvel","foot_normal_force_n","force"):
        v=np.array([r[key] for r in actual[:n]])-np.array([r[key] for r in exp[:n]])
        diff[key]=float(abs(v).max())
    records.append(dict(parent=cell["parent_id"],actual=len(actual),archived=len(exp),diff=diff))
write(out/"native-complete-archive-check.json",dict(records=records,wall_s=time.monotonic()-start))
