"""One authorized asset only; sequential, rate-limited, SHA-verified download."""
import hashlib,json,os,time,urllib.request
from pathlib import Path

REV='62b4206d68e026de5e5dc7efb1529bccfb95164c'
URL=f'https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/{REV}/new_model_for_training_code_inference/checkpoint/buffers/train/buffer.hdf5?download=true'
DEST=Path('/mnt/data-oss/users/yi.gao/compliance_BFM/native_reward_assets')/REV/'train'
OUT=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/issue21-reward-stage-20261030A')
SIZE=26675212288
SHA='46eac53f603e45921d25504fcf7d3eea861f2a9f8ea503046855e701493b546b'
RATE=20*1024**2
start=time.monotonic()
status={'pid':os.getpid(),'expected_bytes':SIZE,'expected_sha256':SHA,'rate_limit_bytes_s':RATE,'deadline_seconds':5400,'attempts':[]}

def save():
    (OUT/'download-status.json').write_text(json.dumps(status,indent=2))

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

final=DEST/'buffer.hdf5'
part=DEST/'buffer.hdf5.part'
if final.exists():
    assert final.stat().st_size==SIZE and digest(final)==SHA
    status.update(state='already-verified',sha256=SHA);save();raise SystemExit(0)
op=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}))
for attempt in range(1,5):
    offset=part.stat().st_size if part.exists() else 0
    if offset==SIZE:break
    assert offset<SIZE
    row={'attempt':attempt,'offset':offset};status['attempts'].append(row);status['state']='downloading';save()
    try:
        req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-{SIZE-1}'})
        with op.open(req,timeout=40) as r:
            cr=r.headers.get('Content-Range','')
            assert (r.status==206 and cr==f'bytes {offset}-{SIZE-1}/{SIZE}') or (offset==0 and r.status==200 and int(r.headers.get('Content-Length',0))==SIZE),(r.status,cr)
            row['http_status']=r.status;row['content_range']=cr
            received=offset;request_start=time.monotonic();last=0
            with part.open('ab' if offset else 'xb') as f:
                while received<SIZE:
                    if time.monotonic()-start>5400:raise TimeoutError('Download wall budget exhausted')
                    b=r.read(min(4*1024**2,SIZE-received))
                    if not b:raise EOFError('Premature EOF')
                    f.write(b);received+=len(b)
                    delay=(received-offset)/RATE-(time.monotonic()-request_start)
                    if delay>0:time.sleep(delay)
                    if time.monotonic()-last>30:
                        status.update(received_bytes=received,elapsed_seconds=time.monotonic()-start)
                        save();print('PROGRESS',received,SIZE,flush=True);last=time.monotonic()
        row['complete']=True;break
    except Exception as e:
        row['error']=type(e).__name__+': '+str(e).split('?')[0]
        print('DOWNLOAD_ERROR',json.dumps(row),flush=True);save()
        if time.monotonic()-start>5400:break
assert part.stat().st_size==SIZE,'Incomplete asset retained; not usable'
status['state']='hashing';save()
actual=digest(part);assert actual==SHA,('SHA mismatch',actual)
part.rename(final)
status.update(state='verified',sha256=actual,received_bytes=SIZE,elapsed_seconds=time.monotonic()-start,path=str(final));save()
print('VERIFIED',json.dumps(status),flush=True)
