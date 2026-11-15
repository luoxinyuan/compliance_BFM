"""Parent-balanced finite actual successor windows, current policy phase only.
No source/reference/candidate rows or cross-episode futures enter this buffer.
"""
import numpy as np
import torch
from .ei_v2_agent import SCALES

def outcome(ph):
    assert ph
    a,b=ph[0],ph[-1];R=np.asarray(a['pelvis_matrix']).T
    w=np.asarray(b['wrist'])-a['wrist'];r=np.asarray(b['root'])-a['root']
    yaw=np.unwrap([np.arctan2(p['pelvis_matrix'][1][0],p['pelvis_matrix'][0][0]) for p in ph]);active=[p for p in ph if p['active']]
    y=np.r_[R@w,R@r,R@(w-r),R@np.mean([p['force'] for p in ph],axis=0),R@np.asarray(b['contacts'][0]['velocity']),R@np.asarray(b['qvel'][:3]),yaw[-1]-yaw[0],max(p['stage_loaded_slip_mps'] for p in ph),max(p['tilt_deg'] for p in ph),np.mean([any(p['support']) for p in ph]),np.mean([np.linalg.norm(p['force'])>=9.999 for p in active]) if active else 0.,max(p['joint_limit_excess_max_rad'] for p in ph)]
    assert y.shape==(24,) and np.isfinite(y).all()
    return y.astype(np.float32)

class ActualSuccessors:
    def __init__(self,env,phase):self.env=env;self.phase=phase;self.last_n=-1;self.items={}
    def prepare(self):
        env=self.env
        if self.last_n==len(env.rows):return
        self.last_n=len(env.rows);rows=[r for r in env.rows if r.get('eligible_for_joint_PPO') and r['behavior_phase']==self.phase][-8192:];self.items={}
        by_episode={}
        for r in rows:by_episode.setdefault(r['episode'],[]).append(r)
        for rr in by_episode.values():
            end=rr[-1]['physics_end']
            for r in rr:
                begin=r['physics_begin'];ys=[];masks=[]
                for n in (100,200,400):
                    valid=begin+n<=end
                    ph=env.physical_stream[begin:begin+n] if valid else None
                    if valid:assert len(ph)==n and abs(ph[-1]['force_time']-ph[0]['force_time']-(n-1)*.005)<1e-7
                    ys.append(outcome(ph)/np.asarray(SCALES,dtype=np.float32) if valid else np.zeros(24,dtype=np.float32));masks.append(valid)
                if any(masks):self.items.setdefault(bytes(r['provenance']['parent_sha256'][0].tolist()).hex(),[]).append((r,np.stack(ys),masks))
    def __call__(self):
        self.prepare()
        if not self.items:return None
        keys=sorted(self.items);data=[]
        for _ in range(64):
            k=keys[int(torch.randint(len(keys),(1,)))];v=self.items[k];data.append(v[int(torch.randint(len(v),(1,)))])
        rows=[v[0] for v in data];dev=self.env.agent.device
        o={k:torch.cat([r['observation'][k] for r in rows]).to(dev) for k in rows[0]['observation']}
        return o,torch.cat([r['z'] for r in rows]).to(dev),torch.tensor([r['option'] for r in rows],device=dev),torch.tensor(np.stack([v[1] for v in data]),device=dev,dtype=torch.float32),torch.tensor(np.asarray([v[2] for v in data]),device=dev,dtype=torch.bool)
