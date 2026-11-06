"""Frozen, post-selection real-physics evaluation. No training or checkpoint choice."""
import m11_ipc
import argparse, json, os, time, hashlib, subprocess, multiprocessing as mp
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CLEAN_KIND='direct',NARROW_DOMAIN='neighborhood',NARROW_SEED='4700',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    import torch,numpy as np
    from c0_engagement_worker import launch
    from humanoidverse.force_compliance.parallel import receive
    from clean_policy import Expert
    from geometry_policy import expand
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(__file__).resolve().parents[1];commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain'])
    cfg=json.loads(Path(__file__).with_name('c0_engagement_protocol.json').read_text())
    extra=json.loads(Path(__file__).with_name('c0_engagement_final_specs.json').read_text())
    schedule={**cfg['development'],**cfg['heldout'],**extra['specifications']}
    models={k:Path(cfg['models'][k]['path']) for k in ['C0','I','RL4601']}
    for k,p in models.items():assert sha(p)==cfg['models'][k]['sha256']
    selection={}
    for seed in [4701,4702]:
        d=a.stage/f'ppo{seed}-r1';result=json.loads((d/'result.json').read_text());assert result['status']=='completed' and result['iterations']==16
        s=json.loads((d/'selection.json').read_text());assert sha(d/'best.pt')==s['sha256'];selection[str(seed)]=s
        models[f'{seed}_selected']=d/'best.pt';models[f'{seed}_last']=d/'last.pt'
    assert not a.output.exists();a.output.mkdir()
    atomic(a.output/'manifest.json',dict(started_utc=time.time(),runtime=commit,protocol_sha256=sha(Path(__file__).with_name('c0_engagement_protocol.json')),extra_sha256=sha(Path(__file__).with_name('c0_engagement_final_specs.json')),selection=selection,models={k:dict(path=str(v),sha256=sha(v)) for k,v in models.items()},condition_order=list(schedule),primary_seed=4701,no_selection_from_final=True))
    ctx=mp.get_context('spawn');ps=[];cs=[];beg=time.monotonic();output={};fingerprints={}
    try:
        for i in range(3):
            c,x=ctx.Pipe();p=ctx.Process(target=launch,args=(x,a.gpu,i,a.output/f'worker-{i}'));p.start();x.close();cs.append(c);ps.append(p)
        ready=[receive(c,p,900) for c,p in zip(cs,ps)];assert all(x['input_dim']==1088 for x in ready)
        net=Expert(1088).cuda()
        for name,p in models.items():
            ck=torch.load(p,map_location='cuda',weights_only=True);expand(net,ck)
            weights={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
            h=hashlib.sha256()
            for k in sorted(weights):
                if k.startswith('actor.') or k in ['center','scale','log_std']:h.update(k.encode());h.update(weights[k].tobytes())
            digest=h.hexdigest()
            if digest in fingerprints:
                output[name]=dict(reused_actor=fingerprints[digest],actor_hash=digest);atomic(a.output/'index.json',output);continue
            fingerprints[digest]=name
            d=a.output/name;d.mkdir();items=[(k,v,'learned') for k,v in schedule.items()]
            for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=weights,schedule=items[i::3],output=str(d)))
            rows=[]
            for c,proc in zip(cs,ps):rows+=receive(c,proc,1800)['summaries']
            atomic(d/'summary-private.json',rows)
            public=[{k:v for k,v in r.items() if k!='handoff'} for r in rows];atomic(d/'summary.json',public)
            output[name]=dict(directory=str(d),actor_hash=digest,complete=len(rows),safety_boundaries=sum(bool(r['boundary']) for r in rows))
            atomic(a.output/'index.json',output);print('FROZEN_FINAL_MODEL',name,len(rows),flush=True)
    finally:
        for c in cs:
            try:c.send(dict(op='close'))
            except (OSError,EOFError):pass
        for p in ps:
            p.join(10)
            if p.is_alive():p.terminate();p.join(10)
        atomic(a.output/'resource.json',dict(job_wall_seconds=time.monotonic()-beg,new_RL_training_decisions=0,new_supervised_updates=0,completed=list(output),workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps]))
if __name__=='__main__':main()
