"""One bounded CUDA unit check on saved data; no new physics or RL decisions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from clean_policy import Expert, update as original_update
from p4000_critic import update, actor_adam, equal_tree
from humanoidverse.force_compliance.parallel import merge_rollouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    stage = Path(__file__).resolve().parents[2]
    source = stage/'runs/seed5201-32'
    checkpoint = torch.load(source/'initial.pt', map_location='cuda', weights_only=True)
    results = []
    for worker in range(3):
        with np.load(source/f'worker-{worker}/rollout-0000.npz') as data:
            results.append(dict(batch={k:data[k].copy() for k in data.files}))
    batch = {k:v.cuda() for k,v in merge_rollouts(results, .95).items()}
    batch['raw_adv_std'] = (batch['returns']-batch['value']).std()
    nets, opts = [], []
    for _ in range(2):
        net = Expert(1088, 'direct').cuda()
        net.load_state_dict(checkpoint['state_dict'])
        opt = torch.optim.Adam([dict(params=net.actor.parameters(),lr=3e-5),
                                dict(params=net.critic.parameters(),lr=3e-4)])
        nets.append(net); opts.append(opt)
    started = time.monotonic()
    torch.manual_seed(9701)
    old = original_update(nets[0], opts[0], batch)
    torch.manual_seed(9701)
    new = update(nets[1], opts[1], batch, critic_shuffle_seed=52010000)
    actor_equal = all(torch.equal(v, nets[1].actor.state_dict()[k]) for k,v in nets[0].actor.state_dict().items())
    adam_equal = equal_tree(actor_adam(opts[0],nets[0]), actor_adam(opts[1],nets[1]))
    assert actor_equal and adam_equal, 'The actor update changed on the same saved batch'
    assert old['optimizer_updates'] == new['optimizer_updates']
    assert new['critic_fit']['optimizer_updates'] == 18
    assert new['critic_fit']['post']['mse'] < new['critic_fit']['pre']['mse']
    assert old['pre_logprob_max_error'] < 3e-4 and new['pre_logprob_max_error'] < 3e-4
    assert old['epochs'][-1]['joint_gaussian_KL'] <= .02001
    assert new['epochs'][-1]['joint_gaussian_KL'] <= .02001
    result = dict(status='completed', check='same CUDA actor update and Adam state bitwise equal',
                  initial_checkpoint_sha256=hashlib.sha256((source/'initial.pt').read_bytes()).hexdigest(),
                  saved_rollout_scope='first batch before original critic warmup; implementation unit check only',
                  actor_equal=actor_equal, actor_adam_equal=adam_equal, old=old, new=new,
                  RL_decisions=0, new_physics_steps=0, elapsed_s=time.monotonic()-started,
                  limitation='Training target fit is not physical improvement or independent value validation')
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
