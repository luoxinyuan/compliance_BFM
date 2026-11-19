"""Summarize paired mean execution on the same two development ancestors."""
from pathlib import Path
import csv,json,sys,statistics
run=Path(sys.argv[1])
raw=list(csv.DictReader((run/'stage-readout.csv').open()))
nodes=sorted({int(r['node']) for r in raw})
out=[]
for n in nodes:
    subset=[r for r in raw if int(r['node'])==n]
    for phase in ('moving','hold','release','full'):
        for branches in [('left','front','right'),('fork_left','fork_right'),('zero',),('none',),('native',),('intermediate',)]:
            rows=[r for r in subset if r['branch'] in branches and r['phase']==phase]
            if not rows:continue
            stats=dict(node=n,phase=phase,branches='/'.join(branches),cells=len(rows))
            keys=['force_rms_N','connected_error_rms_cm','arm_x_cm','arm_y_cm','root_path_cm','material_speed_p95_mps_at_20N']
            for k in keys:
                xs=[float(r[k]) for r in rows if r[k]!='']
                stats[k]=statistics.mean(xs) if xs else None
            if 'left' in branches:
                by={(r['source'],r['branch']):float(r['arm_y_cm']) for r in rows}
                stats['left_minus_right_arm_y_cm']=statistics.mean(by[p,'left']-by[p,'right'] for p in sorted({r['source'] for r in rows}) if (p,'left') in by and (p,'right') in by) if len(rows)>=3 else None
            out.append(stats)
(run/'matched-evaluation.json').write_text(json.dumps(out,indent=2)+'\n')
