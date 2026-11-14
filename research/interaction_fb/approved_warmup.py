"""Four-block real Q-only entry, immutable lossless IO and new-process continuation.
Requires prior frozen mean/sampled qualification. No Actor/U update.
"""
import argparse,json,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
P=ROOT.parents[2]

def study(e,m,c,out):
 import torch,numpy as np
 from native_env import install
 from m11_runtime_speed import install as speed
 from rolling_backend import install as backend
 from research.interaction_fb.env import Environment
 from research.interaction_fb.approved_learning import ApprovedLearner
 from research.interaction_fb.approved_journal import Journal,load_state
 from research.interaction_fb.checkpoint import restore
 from diagnostic_fb_resume import tree_digest
 from dual_track_task import global_rng
 from f1_mjdata_contract import install_restore_guard
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 install();speed(e);(out/'backend-prefix').mkdir();backend(e,m,out/'backend-prefix');install_restore_guard()
 stage=ROOT.parent
 if ARGS.blocks>4:
  entry=json.loads((stage/'prep/q-entry-complete.json').read_text())
  assert entry['E_resume_exact'] and entry['I_resume_exact'] and entry['physical_pair_exact']
  for proof in entry['proofs']:
   proof_path=Path(proof['path'])
   assert hashlib.sha256(proof_path.read_bytes()).hexdigest()==proof['sha256']
   result=json.loads(proof_path.read_text())
   assert result['status']=='q_only_entry_complete' and result['study']['resume_exact']
  assert entry['warmup_only_projection_bytes']<12*2**30
 qualified=P/'experiments/r2-v11-20261114/runs/v11-remaining-sample-05/qualification.json'
 assert json.loads(qualified.read_text())['status']=='qualification_passed'
 mean=P/'experiments/r2-approved-20261114/runs/fresh-mean-qualification-01/qualification.json'
 assert json.loads(mean.read_text())['status']=='qualification_passed'
 lock=json.loads((ROOT/'research_notes/r2_ei_v11_entry_lock.json').read_text())
 data=json.loads((ROOT/'research_notes/r2_splits/train.json').read_text())
 assert hashlib.sha256((ROOT/'research_notes/r2_splits/train.json').read_bytes()).hexdigest()==lock['train_file_sha256']
 cases=[dict(sp,split='train') for p in data['parents'] for sp in p['specs']]
 from research.interaction_fb.warmup_guard import GuardedEnvironment
 env=GuardedEnvironment(e,m,c,cases);env.guard_output=out;env.reset(0);obs=env.proxy.enrich(env.task.extended_observation())
 agent=ApprovedLearner(m,obs,e.standing,torch.tensor(c['q_PD_scale'],device='cuda'),ARGS.arm,lock);env.proxy.agent=agent
 # Initial source generation precedes the fixed training random stream.
 torch.manual_seed(636411);np.random.seed(636411)
 base_digest=tree_digest(m._actor.state_dict());tail_digest=tree_digest(agent.tail.state_dict());u_digest=tree_digest(agent.nets['u'].state_dict())
 records=[];restored=0
 journal=Journal(out/'checkpoints')
 if ARGS.resume:
  restored=restore(ARGS.resume,agent,env)
  assert tree_digest(agent.state())==tree_digest(load_state(ARGS.resume)['agent'])
  if restored<ARGS.blocks:assert journal.adopt(ARGS.resume)==restored
 else:
  journal.save(0,agent,env,RUNTIME,'fresh_initial')
 for block in range(restored+1,ARGS.blocks+1):
  start=time.monotonic()
  with torch.no_grad():
   for _ in range(50):env.collect_call()
  sampled=time.monotonic()
  with torch.enable_grad():updates=agent.update_block_q_only(env.rows,block)
  torch.cuda.synchronize();updated=time.monotonic()
  meta=journal.save(block,agent,env,RUNTIME)
  rec=dict(block=block,transitions=env.next_cursor,collect_s=sampled-start,update_s=updated-sampled,
      io_s=time.monotonic()-updated,core_bytes=meta['files']['core.delta']['bytes'],data_bytes=meta['files']['data.ptz']['bytes'],
      q_steps=agent.q_steps,representation_steps=agent.representation_steps,actor_steps=agent.actor_steps,updates=updates)
  assert tree_digest(agent.tail.state_dict())==tail_digest and tree_digest(agent.nets['u'].state_dict())==u_digest
  records.append(rec);(out/'blocks.json').write_text(json.dumps(records,indent=2));print('Q_ONLY_COMMITTED',json.dumps({k:v for k,v in rec.items() if k!='updates'}),flush=True)
  from research.interaction_fb import benchmark
  if benchmark.STOP_REQUESTED:return dict(status='stopped_after_complete_block',blocks=records)
 # Independent counterfactual continuation after saved block4; never promoted into rollout.
 before=len(env.physical_stream)
 with torch.no_grad():env.collect_call()
 with torch.enable_grad():metric=agent.q_update(env.rows)
 torch.cuda.synchronize()
 proof=dict(agent=tree_digest(agent.state()),rng=tree_digest(global_rng()),physics=tree_digest(env.physical_stream[before:]),rows=tree_digest(env.rows[-5:]),metric=metric)
 if ARGS.resume and restored==ARGS.blocks:
  ref=json.loads((ARGS.resume.parent.parent/'continuation-proof.json').read_text());assert proof==ref,'Full resumed continuation/update differs'
 (out/'continuation-proof.json').write_text(json.dumps(proof,indent=2))
 assert tree_digest(m._actor.state_dict())==base_digest
 return dict(status='q_only_entry_complete',arm=ARGS.arm,blocks=records,checkpoint_transitions=ARGS.blocks*250,
     continuation_test_only_transitions=5,actor_steps=0,resume_exact=bool(ARGS.resume and restored==ARGS.blocks),resumed_from_round=restored,peak_cuda_bytes=torch.cuda.max_memory_allocated())

def main():
 global ARGS,RUNTIME
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=['E','I'],required=True);p.add_argument('--resume',type=Path);p.add_argument('--blocks',type=int,choices=[4,14,20],default=4);ARGS=p.parse_args()
 RUNTIME=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 from research.interaction_fb import benchmark
 benchmark.study=study;sys.argv=[__file__,'--output',str(ARGS.output),'--arm','A'];benchmark.main()
if __name__=='__main__':main()
