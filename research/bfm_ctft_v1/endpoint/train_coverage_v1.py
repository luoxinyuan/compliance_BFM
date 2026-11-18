import csv,json,sys,statistics
from pathlib import Path
from collections import Counter
stage=Path(sys.argv[1]);out=stage/"runs/endpoint-evaluation-02"
curves=[];coverage={}
for a in "UT":
    path=stage/f"runs/formal-{a}-v1a-01/blocks.json"; b=json.loads(path.read_bytes())
    for r in b:
        avg={k:statistics.mean(v[k] for v in r["metrics"]) for k in ("actor_loss","fb_loss","aux_critic_loss","disc_loss","Q_fb","Q_discriminator","Q_aux","mean_aux_reward") if k in r["metrics"][0]}
        curves.append(dict(arm=a,rows=r["collected"],updates=r["updates"],**avg))
    last=b[-1];eps=last["actual_episodes"]
    coverage[a]=dict(actual_phase_rows=last["actual_phase_rows"],actual_complete_episodes=sum(e["complete"] for e in eps),started_episodes=len(eps),distinct_source_cases=len({e["spec"]["source"]["case"] for e in eps}),actual_source_entry_components=dict(Counter(e["spec"]["source"]["reset_component"] for e in eps)),endpoints=json.loads((stage/f"runs/formal-{a}-v1a-01/store/round-131072/readback.json").read_bytes()))
with (out/"train-curves.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
(out/"train-coverage.json").write_text(json.dumps(coverage,indent=2))
