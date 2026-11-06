"""Publish small aggregate snapshots; never include handoff or raw trajectories."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

def read(p):
    return json.loads(p.read_text()) if p.exists() else None

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    out={'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'in_progress','training':{},'initial_identity':read(a.stage/'initial-identity-check.json')}
    for seed in [4701,4702]:
        d=a.stage/f'ppo{seed}-r1';curves=[]
        for line in (d/'training.jsonl').read_text().splitlines():
            if not line.strip():continue
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # last line can be being appended
            curves.append({k:v for k,v in r.items() if k not in ['episodes']})
        dev={}
        for x in sorted(d.glob('dev-*/summary.json')):
            score=read(x.parent/'score.json')
            if x.parent.name=='dev-0000':score={'score':1.,'selection':{'groups':dict(near=1.,mid=1.,far=1.,stop=1.)},'definition':'Each condition normalized to this frozen initial C0 cost'}
            dev[x.parent.name]={'score':score,'episodes':[{k:v for k,v in r.items() if k!='handoff'} for r in read(x)]}
        hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in d.glob('*.pt')}
        out['training'][str(seed)]={'curves':curves,'development':dev,'selection':read(d/'selection.json'),'result':read(d/'result.json'),'model_SHA256':hashes}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({'checked_utc':out['checked_utc'],'training':{k:{'iterations':len(v['curves']),'selection':v['selection'],'result':v['result']} for k,v in out['training'].items()}},indent=2))

if __name__=='__main__':main()
