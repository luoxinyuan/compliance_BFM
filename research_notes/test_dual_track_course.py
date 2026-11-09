"""CPU-only marginal/prefix/provenance regression; zero physical or RL steps."""
import argparse,json,tempfile
import numpy as np
import torch
from dual_track_task import Task,CFG
from humanoidverse.agents.buffers.transition import DictBuffer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    x=Task.__new__(Task);x.rng=np.random.default_rng(560205);x.pending_continue_spec=None
    counts={k:0 for k in CFG['request_probs']};pairs=0
    for i in range(12000):
        s=x.sample_spec();counts[s['demand_family']]+=1
        if s['demand_family']=='stop':
            t=x.pending_continue_spec;assert t is not None
            assert all(s[k]==t[k] for k in ('source','source_profile','source_until','onset','distance','ramp','parent_id','source_parent_id','common_prefix_parent_id'))
            assert 'stop_time' not in t and t['release']>s['release'];pairs+=1
    rates={k:v/12000 for k,v in counts.items()}
    assert all(abs(rates[k]-v)<.02 for k,v in CFG['request_probs'].items()),rates
    b=DictBuffer(capacity=8,device='cpu')
    data=dict(action=torch.ones(3,29),provenance=dict(
        parent_sha256=torch.arange(96,dtype=torch.uint8).reshape(3,32),time=torch.arange(3,dtype=torch.float64)[:,None]))
    b.extend(data)
    from pathlib import Path
    out=Path(a.output)
    with tempfile.TemporaryDirectory(dir=out) as folder:
        b.save(folder);c=DictBuffer.load(folder)
        for key in ('parent_sha256','time'):
            assert torch.equal(b.get_full_buffer()['provenance'][key],c.get_full_buffer()['provenance'][key])
    result=dict(episode_counts=counts,episode_fractions=rates,paired_parents=pairs,
        same_prefix_task_parameters_exact=True,replay_provenance_roundtrip_exact=True,physics_steps=0,optimizer_steps=0)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
