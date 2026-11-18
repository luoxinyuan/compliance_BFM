"""Descriptive chronology only. Actual training/evaluation gates remain fixed."""
import json,gzip,pickle,time,sys
from pathlib import Path
import numpy as np
folder=Path(sys.argv[1]);rows=[];start=time.monotonic()
for file in sorted(folder.glob("ctft-eval-*-continue.pkl.gz")):
    with gzip.open(file,"rb") as f:x=pickle.load(f)
    met=json.loads((folder/(file.name+".metrics.json")).read_bytes())
    bad=[s for s in met["stances"] if s["measured"] and (s["p95"]>.05 or s["path"]>.02)]
    r=x["physics"];mag=np.linalg.norm([q["force"] for q in r],axis=1);cap=np.flatnonzero(mag>=10-1e-8)
    complete_t=min((s["end_exclusive"]*.005 for s in bad),default=None)
    rows.append(dict(parent=met["parent"],model=met["model"],earliest_failed_stance_end_s=complete_t,first_cap_tick_s=float(cap[0]*.005) if len(cap) else None))
(folder/"support-force-order.json").write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-start),indent=2))
