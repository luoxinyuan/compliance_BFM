"""Finite shared-GPU privileged expert acquisition; reuses force_compliance GAE/parallel contracts."""
import argparse,os,json,time,signal,multiprocessing as mp,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM/bridge-v2-owned/runs/m11-privileged-expert-20261101')
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['throughput','train','final'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--iterations',type=int,default=200);p.add_argument('--steps',type=int,default=256);p.add_argument('--resume',type=Path);p.add_argument('--max-seconds',type=float,default=14400);a=p.parse_args();assert not a.output.exists();a.output.mkdir()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch,numpy as np
    from humanoidverse.force_compliance.parallel import receive,merge_rollouts
    from m11_policy import Expert,update
    from m11_worker import launch
    plan=json.loads(Path(__file__).with_name('m11_protocol.json').read_text());torch.set_num_threads(1);torch.manual_seed(plan['seed']);torch.set_grad_enabled(False)
    ctx=mp.get_context('spawn');ps=[];cs=[];status='initializing';net=opt=None;decisions=iteration=updates=0;best=float('inf');started=time.monotonic();logs=[]
    def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,stop)
    def weights():return {k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
    def save(name):
        f=a.output/name;tmp=f.with_suffix('.tmp');torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=opt.state_dict(),input_dim=n,decisions=decisions,iteration=iteration,optimizer_updates=updates,best_score=best,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),prior_sha=plan['prior_sha256'],runtime_commit=commit,plan_sha=sha(Path(__file__).with_name('m11_protocol.json'))),tmp);tmp.replace(f);return f
    def evaluate(schedule,folder):
        folder.mkdir();w=weights()
        for i,c in enumerate(cs):c.send(dict(op='evaluate',policy=w,schedule=schedule[i::len(cs)],output=str(folder)))
        rows=[]
        for c,proc in zip(cs,ps):rows+=receive(c,proc,timeout=1200)['summaries']
        atomic(folder/'summary.json',rows);return rows
    def selection(rows):
        # Fixed before fitting; rank all dev families, penalize failures and force saturation.
        return float(np.mean([r['force_rms']+3*r['error_cm']/100+1*r['cap_fraction']+20*(r['boundary'] is not None) for r in rows]))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'])
    atomic(a.output/'launch.json',dict(pid=os.getpid(),commit=commit,argv={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},started=time.time()))
    try:
        for i in range(a.workers):
            parent,child=ctx.Pipe();proc=ctx.Process(target=launch,args=(child,a.gpu,i,a.output/f'worker-{i}'));proc.start();child.close();ps.append(proc);cs.append(parent)
        ready=[receive(c,p,timeout=900) for c,p in zip(cs,ps)];assert len({r['input_dim'] for r in ready})==1;n=ready[0]['input_dim'];atomic(a.output/'workers.json',ready)
        net=Expert(n).cuda();opt=torch.optim.Adam([dict(params=net.actor.parameters(),lr=3e-5),dict(params=[net.log_std],lr=1e-5),dict(params=net.critic.parameters(),lr=3e-4)])
        if a.resume:
            ck=torch.load(a.resume,map_location='cuda',weights_only=True);assert ck['prior_sha']==plan['prior_sha256'];net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);decisions=ck['decisions'];iteration=ck['iteration'];updates=ck['optimizer_updates'];best=ck['best_score'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        save('initial.pt')
        if a.stage=='final':
            rows=evaluate([(name+'__'+mode,spec,mode) for name,spec in plan['final'].items() for mode in ['prior','m10_expert','learned']],a.output/'evaluation');status='completed';atomic(a.output/'final-summary.json',rows)
        else:
            if a.stage=='train':
                initial=evaluate([(name+'__learned',spec,'learned') for name,spec in plan['dev'].items()],a.output/'dev-initial');best=min(best,selection(initial));save('best.pt');atomic(a.output/'selection.json',dict(score=best,path=str(a.output/'best.pt'),rule=plan['selection']))
            status='running';iterations=2 if a.stage=='throughput' else a.iterations
            for local_it in range(iterations):
                if time.monotonic()-started>a.max_seconds:status='time_budget';break
                t=time.monotonic();w=weights()
                for c in cs:c.send(dict(op='rollout',steps=a.steps,version=iteration,policy=w))
                results=[receive(c,p,timeout=1200) for c,p in zip(cs,ps)];sample_s=time.monotonic()-t;assert all(r['version']==iteration for r in results)
                batch=merge_rollouts(results,.95);batch={k:v.cuda() for k,v in batch.items()};batch['raw_adv_std']=(batch['returns']-batch['value']).std();t=time.monotonic();u=update(net,opt,batch)
                decisions+=len(batch['reward']);iteration+=1;updates+=u['optimizer_updates'];stats=[s for r in results for s in r['stats']];active=sum(s['active_steps'] for s in stats);phys=sum(s['physics_steps'] for s in stats)
                row=dict(iteration=iteration,rl_decisions=decisions,optimizer_updates=updates,sampling_seconds=sample_s,optimization_seconds=time.monotonic()-t,decisions_per_second=len(batch['reward'])/sample_s,physical_steps=phys,control_steps=sum(s['control_steps'] for s in stats),prefill_physics_steps=sum(r['prefill_physics_steps'] for r in results),force_rms=(sum(s['active_force_squared'] for s in stats)/max(active,1))**.5,cap_fraction=sum(s['cap_steps'] for s in stats)/max(active,1),reward=float(batch['reward'].mean()),reward_terms={k:float(np.mean([s['terms'][k] for s in stats])) for k in stats[0]['terms']},completed_episodes=[e for r in results for e in r['episodes']],optimization=u,elapsed_s=time.monotonic()-started)
                logs.append(row)
                with (a.output/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic(a.output/'status.json',dict(status=status,**row));print('M11_UPDATE',iteration,decisions,'force',round(row['force_rms'],3),'rate',round(row['decisions_per_second'],2),'KL',round(u['epochs'][-1]['joint_gaussian_KL'],5),'EV',round(u['updated_value_EV'],3),flush=True)
                if iteration%5==0 or a.stage=='throughput':save('latest.pt')
                if a.stage=='train' and iteration%plan['evaluate_every']==0:
                    rows=evaluate([(name+'__learned',spec,'learned') for name,spec in plan['dev'].items()],a.output/f'dev-{iteration:04d}');score=selection(rows)
                    if score<best:best=score;save('best.pt');atomic(a.output/'selection.json',dict(score=score,iteration=iteration,decisions=decisions,path=str(a.output/'best.pt'),rule=plan['selection']))
                    atomic(a.output/f'dev-{iteration:04d}/score.json',dict(score=score,best=best));print('M11_DEV',iteration,score,best,flush=True)
                if a.stage=='throughput' and local_it==0:torch.save({k:v.cpu() for k,v in batch.items()},a.output/'diagnostic-batch.pt')
            else:status='completed'
            save('last.pt')
            if a.stage=='throughput':atomic(a.output/'throughput.json',dict(samples=sum(len(r['batch']['reward']) for r in results),last_sampling_s=logs[-1]['sampling_seconds'],rate=logs[-1]['decisions_per_second'],total_decisions=decisions,physical_steps=sum(r['physical_steps'] for r in logs),updates=updates))
    except KeyboardInterrupt:
        status='interrupted'
        if net is not None:save('interrupted.pt')
    except BaseException:
        import traceback
        status='failed';(a.output/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        for c in cs:
            try:c.send(dict(op='close'))
            except (BrokenPipeError,EOFError,OSError):pass
        for p in ps:
            p.join(timeout=10)
            if p.is_alive():p.terminate();p.join(timeout=10)
        atomic(a.output/'result.json',dict(status=status,rl_decisions=decisions,optimizer_updates=updates,elapsed_s=time.monotonic()-started,workers=[dict(pid=p.pid,exitcode=p.exitcode) for p in ps],bfm_updated=False,student_updated=False))
if __name__=='__main__':main()
