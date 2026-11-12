import json,hashlib,time,sys
from pathlib import Path
import torch,numpy as np
S=Path(sys.argv[1]);out={};started=time.time()
def read(p):return json.loads(p.read_text())
def exact(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
ctrl=read(S/'runs/control-04/rounds.json');res=read(S/'runs/resume-02/rounds.json')
out['round_fingerprints']={str(r['round']):r['fingerprints']==ctrl[r['round']-1]['fingerprints'] for r in res}
for r in (2,3):
 a=torch.load(S/f'runs/control-04/checkpoints/round-{r}/state.pt',map_location='cpu',weights_only=False)
 b=torch.load(S/f'runs/resume-02/checkpoints/round-{r}/state.pt',map_location='cpu',weights_only=False)
 out[f'round{r}']={k:exact(a[k],b[k]) for k in ('agent','rows','native','commands','actions','rng','task_rng','physics_rng','physical_stream','cursor','spec_id','episodes')}
 out[f'round{r}']['optimizers']={k:exact(a['agent']['optimizers'][k],b['agent']['optimizers'][k]) for k in a['agent']['optimizers']}
rows=a['rows'];out['data']={'rows':len(rows),'terminated':sum(bool(r['next']['terminated']) for r in rows),'duration_values':sorted(set(float(r['duration']) for r in rows)), 'row_keys':list(rows[0]), 'next_keys':list(rows[0]['next']), 'physical_stream_rows':len(a['physical_stream']), 'episodes':a['episodes']}
for k in ('truncated','discount'):
 if k in rows[0]['next']:out['data'][k]=[i for i,r in enumerate(rows) if bool(r['next'][k])][:20]
released=[r for r in rows if not bool(r['observation']['contact_privileged_v1'][0,21])]
out['data']['inactive_rows']=len(released);out['data']['inactive_bootstrap_nonterminal']=sum(not bool(r['next']['terminated']) for r in released)
out['throughput']={'round_seconds':[r['total_s'] for r in ctrl],'warm_transition_s':750/sum(r['total_s'] for r in ctrl),'warm_update_s':6/sum(r['total_s'] for r in ctrl),'update_seconds':sum(r['update_s'] for r in ctrl),'io_seconds':sum(r['io_s'] for r in ctrl),'cold_transition_s':750/read(S/'jobs/control-04.json')['wall_s']}
out['checkpoints']={str(p.relative_to(S)):read(p) for p in (S/'runs/control-04/checkpoints').glob('round-*/manifest.json')}
out['audit_wall_s']=time.time()-started
(S/'audit-results.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
