"""Read-only same-source reconstruction of failed four-way numerical screen.

The first hardware trial did not archive the failed tensors before asserting.
This is explicitly a reconstruction, not an assertion that those lost tensor
bytes were recovered. No physics, learning, precision or tolerance changes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def main():
    import torch
    from agent import build
    from contact_replay import build_buffers
    from learning_checkpoint import digest
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    lock=json.loads((STAGE/'locks/common-aux-scale-v1.json').read_text())
    agent,identity=build(aux_scale=lock['value'],compile_execution=False)
    buffers,_,sources=build_buffers(ROOT,identity['original_training_config']['aux_rewards_scaling'])
    full=buffers['train'].get_full_buffer();indices=torch.tensor([1200,1724,0,600])
    obs={k:v[indices].to(agent.device) for k,v in full['observation'].items()};z=full['z'][indices].to(agent.device)
    before=digest(agent._model._actor.state_dict())
    traces={};order=[];handles=[]
    def hook(name):
        def record(module,args,output):
            if torch.is_tensor(output):
                if name not in traces:traces[name]=[];order.append(name)
                traces[name].append(output.detach().cpu().clone())
        return record
    for name,module in agent._model._actor.named_modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm)):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.no_grad():
        batched=agent.act(obs,z,mean=True)
        for h in handles:h.remove()
        batch_traces=traces;traces={};order=[];handles=[]
        for name,module in agent._model._actor.named_modules():
            if isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm)):
                handles.append(module.register_forward_hook(hook(name)))
        single=torch.cat([agent.act({k:v[i:i+1] for k,v in obs.items()},z[i:i+1],mean=True) for i in range(4)])
    for h in handles:h.remove()
    layers=[]
    for name in batch_traces:
        b=batch_traces[name][0];s=torch.cat(traces[name])
        layers.append(dict(module=name,max_abs=float((b-s).abs().max()),
            fails_original_cross_shape_tolerance=not torch.allclose(b,s,atol=1e-6,rtol=1e-5)))
    result=dict(status='FOUR_WAY_SAME_SOURCE_NUMERICAL_RECONSTRUCTION',
        original_failed_input_tensors_not_archived=True,
        input_identity='reconstructed from actual g1 force/no-force first rows and actual g0 case1/case2 first qualified rows',
        same_input_per_shape=True,source_rows=[1200,1724,0,600],sources=sources,
        action_max_abs=float((batched-single).abs().max()),
        action_within_original_tolerance=bool(torch.allclose(batched,single,atol=1e-6,rtol=1e-5)),
        first_observed_layer_difference=next((r for r in layers if r['max_abs']>0),None),layers=layers,
        float32_matmul_precision=torch.get_float32_matmul_precision(),tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
        no_precision_or_tolerance_changed=True,actor_parameters_unchanged=before==digest(agent._model._actor.state_dict()),
        updates=0,new_physics_ticks=0,wall_s=time.monotonic()-start,
        runtime=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    torch.save(dict(observation={k:v.cpu() for k,v in obs.items()},z=z.cpu(),batched=batched.cpu(),singles=single.cpu()),a.output/'actual-reconstruction.pt')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['layers','sources']}),flush=True)


if __name__=='__main__':main()
