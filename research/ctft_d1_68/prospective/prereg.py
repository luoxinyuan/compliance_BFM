"""Freeze train-only parent, partner and one motion command before new physics."""
from pathlib import Path
import json,gzip,pickle,hashlib,io,sys,os
import numpy as np
import torch
base=Path(os.environ['CTFT_PINNED_RUNTIME']).parent
out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
def rd(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Archives are private trusted training assets. Read tensor storage on CPU.
torch.storage._load_from_bytes=lambda b: torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
srcs=rd(base/'locks/source-reset-domain-v1.json')['train_sources']
eps=rd(base/'runs/formal-U-v1a-01/train-only-check-131072.json')['source_episodes']
picks=[]; walking=None; why=[]
for src in sorted(srcs,key=lambda v:v['case']):
    case=src['case']; path=base/'runs/s0-native-sources-03'/src['evidence_file'];assert sha(path)==src['source_sha256']
    with gzip.open(path,'rb') as f:x=pickle.load(f)
    used=sorted({e['spec']['source']['control'] for e in eps if e['spec']['source']['case']==case and e['spec']['mode']==1})
    parents=[e for e in eps if e['spec']['source']['case']==case and e['spec']['branch']=='continue' and e['complete']]
    if not parents:why.append([case,'no completed historical continue']);continue
    def phase(i):
        r=x['physics'][i*4-1];guard=r['ctft_online_support'];return tuple(bool(k) for k in guard['loaded'])
    def speed(i):return float(np.linalg.norm(x['physics'][4*i-1]['qvel'][:2]))
    if walking is None:
        candidates=[i for i in range(76,min(len(x['controls']),375)) if .08<=speed(i)<=.22 and any(phase(i))]
        if not candidates:why.append([case,'no qualified first native command']);continue
        wi=min(candidates,key=lambda i:(abs(speed(i)-.15),i))
        z=x['controls'][wi]['z'].reshape(-1).numpy();assert abs(np.linalg.norm(z)-16)<1e-4
        np.save(out/'walking.npy',z)
        walking=dict(case=case,control=wi,phase=phase(wi),speed=speed(wi),z_file_sha256=sha(out/'walking.npy'),source_sha=src['source_sha256'])
    legal=[i for i in used if i>=10 and i<len(x['controls']) and phase(i)==tuple(walking['phase'])]
    if not legal:why.append([case,'no compatible archived training handoff']);continue
    idx=min(legal,key=lambda i:(abs(speed(i)-walking['speed']),i))
    pp=parents[0]['spec']['partner'].copy()
    short={**pp,'distance_m':pp['distance_m']*.55}
    picks.append(dict(source=src,control=idx,speed=speed(idx),phase=phase(idx),branches={'continue':pp,'stop':short,'no_force':pp},history_partner=parents[0]['spec']['parent']))
    if len(picks)==8:break
lock=dict(picks=picks,walking=walking,missing=why,first_event='earliest measured failing stance by start then foot. select control immediately before stance, clamp to handoff; normal same ordering; no outcome-based reselection',d1='continue failure, U/B0/T 20ms then U 2s total; no replacements',d2='at source handoff from same actual history; one 1s fixed walking z versus standing, all three branches through release+4s; phase matched by loaded feet',branch_cap=180,physics_cap=1000000,stage_cap_s=36000,output_cap_bytes=96*2**30,parent_cap=12)
(out/'lock.json').write_text(json.dumps(lock,indent=2));print(len(picks),sha(out/'lock.json'),walking)
