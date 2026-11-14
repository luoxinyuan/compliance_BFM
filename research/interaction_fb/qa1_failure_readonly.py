"""One permitted read-only QA1 first-failure diagnosis: no optimizer step/physics."""
import argparse,copy,hashlib,inspect,json,sys,time
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
P=ROOT.parents[2];S=P/'experiments/r2-g17-qa1-20261115'
from research.interaction_fb.approved_journal import load_state
from research.interaction_fb.journal import read
from diagnostic_fb_resume import tree_digest
started=time.monotonic();out=S/'prep/failure-readonly-01';out.mkdir(exist_ok=False)
lock=json.loads((ROOT/'research_notes/r2_ei_qa1_schedule_lock.json').read_text());source=Path(lock['source']);s=load_state(source)
run=S/'runs/fixed-replay-02'
def packet(n):
    p=run/f'U-Q{n}.ptz';m=json.loads(p.with_suffix('.manifest.json').read_text());return read(p,m)
a=packet(56)['agent'];b=packet(512)['agent']
assert tree_digest(s['agent'])==lock['source_agent_digest']==tree_digest(a)
def frozen(ss):
    ss=copy.copy(ss)
    for k in ('audit','updates','q_steps'):ss.pop(k)
    for k in ('nets','targets'):ss[k]={n:v for n,v in ss[k].items() if not n.startswith(('q1.','q2.'))}
    ss['optimizers']={n:v for n,v in ss['optimizers'].items() if n not in ('q1','q2')}
    return tree_digest(ss)
assert frozen(a)==frozen(b)
alias=[]
for name in ('q1','q2'):
    optstate=copy.deepcopy(a['optimizers'][name]);first=optstate['param_groups'][0]['params'][0]
    # A single actual first-parameter state suffices to inspect ownership. No step/backward/mutation.
    actual=optstate['state'][first];par=torch.nn.Parameter(torch.zeros_like(actual['exp_avg']))
    slim=copy.deepcopy(optstate);slim['param_groups'][0]['params']=[first];slim['state']={first:actual}
    optimizer=torch.optim.Adam([par],lr=.0001);optimizer.load_state_dict(slim)
    loaded=optimizer.state[par]
    alias.append(dict(name=name,actual_first_parameter_id=first,step56=float(a['optimizers'][name]['state'][first]['step']),
        step512=float(b['optimizers'][name]['state'][first]['step']),device=str(loaded['step'].device),
        same_step_storage_pointer=loaded['step'].data_ptr()==actual['step'].data_ptr(),
        torch_policy_source=inspect.getsource(torch.optim.Optimizer._process_value_according_to_param_policy)))
    assert alias[-1]['same_step_storage_pointer']
summary=[]
for n in (56,128,256,512):
    p=run/('node-56.json' if n==56 else f'U-node-{n}.json');r=json.loads(p.read_text());means=[x for x in r['scores'] if x['candidate']=='mean']
    summary.append(dict(node=n,correct=r['correct'],n=r['n'],accuracy=r['accuracy'],parent_balanced=r['parent_balanced_accuracy'],
        unique=r.get('unique',1838),by_round=r['by_round'],mean_values=means))
ans=dict(status='STOP_AWAIT_DECISION',read_only=True,optimizer_updates=0,physics_ticks=0,cuda_initialized=torch.cuda.is_initialized(),
    source_agent_digest=tree_digest(s['agent']),source_manifest_unchanged=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()==lock['source_manifest_sha256'],
    U56_equals_original=True,U512_nonQ_exact=True,U_Q_groups=456,C_Q_groups=0,alias_evidence=alias,summary=summary,
    mechanism='PyTorch load_state_dict retains CPU Adam step tensors when capturable/fused false; reused in-memory source state is not an immutable deep snapshot. U updates its own valid Adam step and the aliased source-cache step. Subsequent same-source C load fails exact digest before its first update. Disk source remains unchanged.',
    correction_proposed_not_applied='Use independent deep-copied complete initial agent state on every load, and assert complete initial optimizer/target/model hashes per branch. Do not reuse mutable CPU Adam state as source. New isolated regression/update experiment requires decision under QA1 stop rule.',
    wall_s=time.monotonic()-started)
(out/'diagnostic.json').write_text(json.dumps(ans,indent=2));print(json.dumps(ans,indent=2))
