from pathlib import Path
import json,hashlib
s=Path.cwd()
def rd(p):return json.loads(p.read_bytes())
jobs={k:rd(s/k/'cost.json') for k in ['invalid01','replay','soft']}
physical=sum(v['ticks'] for v in jobs.values());branches=sum(v['branches'] for v in jobs.values())
post=float((s/'post.wall').read_text())+float((s/'front.wall').read_text())
wall=sum(v['seconds'] for v in jobs.values())+post
files=[p for p in s.rglob('*') if p.is_file() and 'ipc' not in p.parts and p.name!='ledger.json']
entry=dict(jobs=jobs,physics_ticks=physical,executions=branches,scientific_episodes=24,optimizer_steps=0,post_s=post,timed_job_s=wall,additional_diagnostics_imports_io_and_coordination_upper_s=900,upper_total_s=wall+900,artifact_bytes=sum(p.stat().st_size for p in files),hashes={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
assert physical<=85000 and branches==28 and entry['upper_total_s']<=5400 and entry['artifact_bytes']+256*2**20<2*2**30
(s/'ledger.json').write_text(json.dumps(entry,indent=2))
