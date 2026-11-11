"""Offline real-checkpoint actor/PD parity on archived actual controls; zero physics steps."""
import argparse,json,time,os,subprocess,fcntl,shutil
from pathlib import Path
from .provenance import file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    start=time.monotonic();a.output.mkdir(exist_ok=False,parents=True)
    runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report=dict(runtime=runtime,status='running',physics_steps=0,training_updates=0,model_forwards=0)
    lock=(a.project/'.dual-track-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert shutil.disk_usage(a.project).free>=100*(1<<30)
        assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        import torch
        from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
        from .executor import OriginalExecutor,direct_command
        torch.set_num_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        modeldir=a.project/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint'
        sha=file_sha(modeldir/'model/model.safetensors');assert sha=='33f410c190877a1348dc3fafa3f0e97b277ad0251b39615ff98e5bd26369e361'
        report.update(model_sha256=sha,model_bytes=(modeldir/'model/model.safetensors').stat().st_size)
        load_start=time.monotonic();model=load_model_from_checkpoint_dir(modeldir,device='cpu');model.eval();model.requires_grad_(False)
        # Same caller-explicit staging as qualified runtime; no monkeypatch or CFG mutation.
        model._actor.to('cuda');model._obs_normalizer.to('cuda');model.device='cuda'
        report['load_and_stage_s']=time.monotonic()-load_start
        cpath=a.project/'bridge-v2-owned/runs/issue21-causallatent-20261030A/attempt-01/outputs/action-contract.json'
        contract=json.loads(cpath.read_text());offset=torch.tensor(contract['q_PD_offset'],device='cuda');scale=torch.tensor(contract['q_PD_scale'],device='cuda')
        adapter=OriginalExecutor(model,offset,scale);checks=[]
        files=['new_stop-sustained-5f6380cdec75','new_stop-sustained-5f6380cdec75-continue','new_tail-hold-97ff10c4dcd9']
        for sid in files:
            path=a.project/'experiments/f1-l1-l2-20261111/runs/l3-02'/(sid+'-L3-OF.pt')
            trajectory=torch.load(path,map_location='cpu',weights_only=False)
            for j in (0,1,4,30,75,120,149):
                row=trajectory['controls'][j];obs={k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in row['bfm_obs'].items()};z=torch.tensor(row['executed_z'],device='cuda')
                a0=model.act(obs,z,mean=True);a1,q1=adapter.command({k:obs[k] for k in adapter.dimensions},z)
                archived_a=torch.tensor(row['action'],device='cuda')[None];archived_q=torch.tensor(row['q_PD'],device='cuda')[None]
                assert torch.equal(a0,a1),'filter changes actor'
                assert torch.equal(a1,archived_a),'archived action parity'
                assert torch.equal(q1,archived_q),'archived q_PD parity'
                checks.append(dict(state=sid,substep=j,actor_exact=True,PD_exact=True));report['model_forwards']+=2
        # Probe the legal input map only, not its physical consequences.
        z0=torch.tensor(trajectory['controls'][0]['executed_z'],device='cuda');raw=z0/16
        assert torch.equal(direct_command(raw,z0),model.project_z(raw))
        report.update(status='passed',checks=checks,actor_fields=list(adapter.dimensions),model_config=json.loads((modeldir/'model/config.json').read_text()),model_init_dimensions={k:list(v.shape) for k,v in model.obs_space.spaces.items()},action_contract=contract,action_contract_sha256=file_sha(cpath),cuda_peak_bytes=torch.cuda.max_memory_allocated(),after_weight_sha256=file_sha(modeldir/'model/model.safetensors'),source_kind='archived actual execution; no simulator created')
        assert report['after_weight_sha256']==sha
    except BaseException as exc:
        report.update(status='failed_preserved',error=repr(exc));raise
    finally:
        report['job_s']=time.monotonic()-start;(a.output/'result.json').write_text(json.dumps(report,indent=2));fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':main()
