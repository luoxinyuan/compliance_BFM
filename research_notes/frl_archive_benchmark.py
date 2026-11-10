"""CPU-only lossless archive benchmark on copied, already recorded Issue58 data."""
import argparse,cProfile,gzip,hashlib,json,os,pstats,time
from pathlib import Path

def write_archive(path, payload, mode):
    if mode == 'legacy':
        with gzip.open(path, 'wt') as f: json.dump(payload, f)
    else:
        with gzip.open(path, 'wb', compresslevel=1 if mode == 'block_level1' else 9) as f:
            f.write(json.dumps(payload).encode('utf-8'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.time();a.output.mkdir(parents=True,exist_ok=False)
    side=a.output/'job.json'
    def save(x):side.write_text(json.dumps(x,indent=2))
    row=dict(status='running',started=start,pid=os.getpid(),device='CPU only; CUDA_VISIBLE_DEVICES empty',category='F-RL-v2 engineering acceleration',hard_job_s=1200,RL_decisions=0,agent_updates=0,source=str(a.source))
    save(row);results=[]
    try:
        for name in ['physics-98.json.gz','native-controls-98.json.gz']:
            t=time.perf_counter()
            with gzip.open(a.source/name,'rt') as f:data=json.load(f)
            load_s=time.perf_counter()-t
            source_bytes=gzip.decompress((a.source/name).read_bytes());digest=hashlib.sha256(source_bytes).hexdigest()
            for repeat in range(3):
                for mode in (['legacy','block_level9','block_level1'] if repeat==0 else ['block_level1','block_level9','legacy']):
                    assert time.time()-start<1100,'engineering time budget'
                    path=a.output/(name+'.'+str(repeat)+'.'+mode)
                    pr=cProfile.Profile();t=time.perf_counter();
                    if repeat==0:pr.enable()
                    write_archive(path,data,mode)
                    if repeat==0:pr.disable()
                    elapsed=time.perf_counter()-t
                    rebuilt=gzip.decompress(path.read_bytes());assert hashlib.sha256(rebuilt).hexdigest()==digest,'Uncompressed archive changed'
                    top=[]
                    if repeat==0:
                        pr.dump_stats(str(path)+'.pstats')
                        stats=pstats.Stats(pr);top=sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:12]
                    results.append(dict(source=name,mode=mode,repeat=repeat,profiled=(repeat==0),rows=len(data),read_decode_s=load_s,write_s=elapsed,bytes=path.stat().st_size,uncompressed_bytes=len(source_bytes),uncompressed_sha256=digest,bytes_exact=True,cprofile_top_self=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_s=v[2],inclusive_s=v[3]) for k,v in top]))
                    (a.output/'measurements.json').write_text(json.dumps(results,indent=2));print(name,mode,repeat,elapsed,path.stat().st_size,flush=True)
        row.update(status='completed',measurements=len(results),elapsed_s=time.time()-start,all_archives_uncompressed_bytes_exact=True)
    except BaseException as exc:
        row.update(status='failed',error=repr(exc),elapsed_s=time.time()-start);save(row);raise
    save(row)
if __name__=='__main__':main()
