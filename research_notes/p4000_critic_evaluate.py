"""Matched fixed16 critic comparison, no reselection."""
import m11_ipc
import os,json,argparse,multiprocessing as mp,hashlib,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];S=ROOT.parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 os.environ.update(CUDA_VISIBLE_DEVICES='0',CLEAN_KIND='direct',P4000_SEED='5201',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 import torch
 from clean_policy import Expert
 from p4000_worker import launch
 from humanoidverse.force_compliance.parallel import receive
 torch.set_num_threads(1);torch.set_grad_enabled(False);plan=json.loads(Path(__file__).with_name('p4000_protocol.json').read_text());inputs=json.loads((S/'critic-final-models.json').read_text());plan=json.loads(Path(__file__).with_name('p4000_critic_protocol.json').read_text());ctx=mp.get_context('spawn');ps=[];cs=[];summary=[];status='initializing'
 try:
  for i in range(3):
   c,d=ctx.Pipe();p=ctx.Process(target=launch,args=(d,'0',i,a.output/f'worker-{i}'));p.start();d.close();ps.append(p);cs.append(c)
  ready=[receive(c,p,900) for c,p in zip(cs,ps)];(a.output/'workers.json').write_text(json.dumps(ready,indent=2));net=Expert(1088,'direct').cuda();identities=[]
  for name,path in inputs.items():
   p=Path(path);digest=hashlib.sha256(p.read_bytes()).hexdigest();ck=torch.load(p,map_location='cuda',weights_only=True);net.load_state_dict(ck['state_dict']);identities.append(dict(name=name,path=path,sha256=digest,iteration=ck.get('iteration'),runtime=ck['runtime']));w={k:v.cpu().numpy() for k,v in net.state_dict().items()}
   schedule=[(name+'__'+k,v,'mean') for k,v in plan['final'].items()]
   for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::len(cs)],output=str(a.output)))
   for c,p in zip(cs,ps):
    rr=receive(c,p,1200)['summaries']
    for r in rr:r['model_sha256']=digest
    summary+=rr
   (a.output/'summaries.json').write_text(json.dumps(summary,indent=2));(a.output/'models.json').write_text(json.dumps(identities,indent=2));print('P4000_FINAL',name,[(x['label'],x['all']['error_cm'],x['all']['force_rms']) for x in summary if x['label'].startswith(name+'__')],flush=True)
  status='completed'
 finally:
  for c in cs:
   try:c.send(dict(op='close'))
   except (BrokenPipeError,EOFError,OSError):pass
  for p in ps:
   p.join(timeout=10)
   if p.is_alive():p.terminate();p.join(timeout=10)
  (a.output/'result.json').write_text(json.dumps(dict(status=status,episodes=len(summary),RL_decisions=0,no_final_reselection=True,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps]),indent=2))
if __name__=='__main__':main()
