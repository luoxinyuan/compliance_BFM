from low2 import *
files={};eps={}
for n in names:
 p=path(n);q=p.with_name(n+'-source-prefix.json.gz');x=load(p);vid=old/'runs/video-input'/(n+'.json.gz')
 if not vid.exists():continue
 y=load(vid);l=len(x['physics']);assert y['physics'][-l:]==x['physics'];assert abs(y['physics'][0]['time']-3.005)<1e-7
 for f in [p,q,vid]:
  files[str(f.relative_to(old))]=hashlib.sha256(f.read_bytes()).hexdigest()
 eps[n]=dict(handoff=x['condition']['handoff_time'],frames=l,video_start=y['physics'][0]['time'],video_input_matches_suffix=True)
(s/'inputs.json').write_text(json.dumps(dict(files=files,episodes=eps),indent=2))
