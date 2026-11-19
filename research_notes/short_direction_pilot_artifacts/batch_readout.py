"""Report actual replay sampling, with minibatch draws distinct from new environment rows."""
import argparse
import ast
import collections
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('run',type=Path)
run=parser.parse_args().run
sampled=collections.defaultdict(collections.Counter)
per_batch=[]
for line in (run/'sampled.jsonl').read_text().splitlines():
    update=json.loads(line)
    stage='actor' if update['update']>=64 else 'warmup'
    eligible=0
    for label,count in update['counts'].items():
        origin,branch,phase,loaded=ast.literal_eval(label)
        sampled[stage]['draws']+=count
        if phase==1 and branch in (1,2,3) and loaded:
            sampled[stage][('left','front','right')[branch-1]]+=count
            eligible+=count
        if phase==3:sampled[stage]['release_draws']+=count
    per_batch.append((stage,eligible))
for stage,stats in sampled.items():
    nums=[n for name,n in per_batch if name==stage]
    stats.update(batches=len(nums),moving_loaded_min=min(nums),moving_loaded_max=max(nums))
(run/'sample-summary.json').write_text(json.dumps(dict(sampled),indent=2)+'\n')
