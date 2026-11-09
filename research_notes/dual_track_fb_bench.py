"""Full original FBcprAuxAgent.update benchmark on newly collected real transitions.
Disposable optimization only. No task/contact extension or learning claim.
"""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import torch,numpy as np
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
from humanoidverse.agents.base_model import migrate_legacy_aux_critic_config
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.buffers.transition import DictBuffer
import safetensors.torch
P=Path('/mnt/data/temp_user/yi.gao/projects/compliance_BFM')
MODEL=P/'bridge-v2-owned/runs/issue19-recovery-20261028T045520Z/model-view/checkpoint/model'
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sync():torch.cuda.synchronize()
def main():
 p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
 torch.set_num_threads(1);torch.manual_seed(5502);torch.set_float32_matmul_precision('high')
 cfg=FBcprAuxAgentConfig(model=FBcprAuxModelConfig(**migrate_legacy_aux_critic_config(json.loads((MODEL/'config.json').read_text()))),compile=False,cudagraphs=False)
 cfg.train.batch_size=1024
 kw=json.loads((MODEL/'init_kwargs.json').read_text());space=json_to_space(kw['obs_space'])
 t=time.monotonic();agent=cfg.build(space,kw['action_dim']);sync();init_s=time.monotonic()-t
 t=time.monotonic();safetensors.torch.load_model(agent._model,MODEL/'model.safetensors',strict=True,device='cuda');sync();load_s=time.monotonic()-t
 assert all(len(o.state)==0 for o in [agent.actor_optimizer,agent.backward_optimizer,agent.forward_optimizer,agent.critic_optimizer,agent.discriminator_optimizer,agent.aux_critic_optimizer])
 train=DictBuffer(capacity=8192,device='cpu');assert len(train)==0
 c=torch.load(a.controls,map_location='cpu',weights_only=False)
 keys=list(kw['obs_space']['spaces']);pairs=[(x,y) for x,y in zip(c,c[1:]) if abs(y['time']-x['time']-.02)<1e-6]
 obs={k:torch.tensor(np.array([x['bfm_obs'][k][0] for x,y in pairs]),dtype=torch.float32) for k in keys};nobs={k:torch.tensor(np.array([y['bfm_obs'][k][0] for x,y in pairs]),dtype=torch.float32) for k in keys}
 data=dict(observation=obs,action=torch.tensor(np.array([x['action'] for x,y in pairs]),dtype=torch.float32),z=torch.tensor(np.array([x['executed_z'][0] for x,y in pairs]),dtype=torch.float32),next=dict(observation=nobs,terminated=torch.zeros(len(pairs),1,dtype=torch.bool)))
 train.extend(data)
 # Benchmark-only sequence provider: 8 contiguous actual controls, never across a reset.
 blocks=[list(range(i,i+8)) for i in range(len(pairs)-7) if all(abs(pairs[i+j][0]['time']-pairs[i][0]['time']-.02*j)<1e-6 for j in range(8))]
 class Sequences:
  def sample(self,batch_size):
   assert batch_size%8==0
   starts=torch.randint(len(blocks),(batch_size//8,));ix=torch.tensor([j for i in starts.tolist() for j in blocks[i]])
   return dict(observation={k:v[ix] for k,v in obs.items()},next=dict(observation={k:v[ix] for k,v in nobs.items()}))
 buffers=dict(train=train,expert_slicer=Sequences());write(a.output/'warm-start.json',dict(weights_only=True,source=str(MODEL),optimizer_empty=True,replay_initially_empty=True,counts_initial=dict(env_steps=0,agent_updates=0),new_replay_transitions=len(train),new_real_sequence_blocks=len(blocks),expert_role='new real BC500 execution sequences for throughput only; not qualified motion expert training dataset',init_s=init_s,weights_load_s=load_s,config=cfg.model_dump()))
 results=[]
 for segment in range(2):
  # Both segments use actual full update: discriminator,F/B,critic,aux critic,actor and target updates.
  # Segment2 compiles existing update functions; no numerical loss/optimizer change.
  ct=time.monotonic()
  if segment==1:
   agent.cfg.compile=True;agent.setup_compile()
  compile_registration_s=time.monotonic()-ct
  times=[];torch.cuda.reset_peak_memory_stats();sync();t0=time.monotonic()
  for i in range(128):
   sync();t=time.monotonic();metrics=agent.update(buffers,segment*128+i);sync();dt=time.monotonic()-t;times.append(dt)
   assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
   if i%8==7:print('FULL_FB_UPDATE',segment,i+1,dt,flush=True)
   write(a.output/f'segment-{segment}-progress.json',dict(updates=i+1,times=times))
  elapsed=time.monotonic()-t0
  result=dict(segment=segment,updates=128,wall_s=elapsed,per_update_s=times,median_s=float(np.median(times)),steady_median_excluding_first8_s=float(np.median(times[8:])),first_update_s=times[0],compile_enabled=segment==1,compile_registration_s=compile_registration_s,compile_first_call_included=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),replay_transitions=len(train),optimizer_names=list(agent.optimizer_dict))
  results.append(result);write(a.output/'segments.json',results)
 # Save/read full model+optimizers once; do not overwrite source archive.
 t=time.monotonic();agent.save(str(a.output/'disposable-checkpoint'));sync();save_s=time.monotonic()-t
 t=time.monotonic();saved=torch.load(a.output/'disposable-checkpoint/optimizers.pth',map_location='cpu',weights_only=False);read_s=time.monotonic()-t
 assert saved.keys()==agent.optimizer_dict.keys()
 write(a.output/'result.json',dict(status='completed',full_agent_updates=256,formal_training=False,model_source_unchanged=True,segments=results,checkpoint_save_s=save_s,optimizer_read_s=read_s,model_read_s_initial=load_s,compiled_model_readback='model safetensors identity checked separately; no formal resume'))
if __name__=='__main__':main()
