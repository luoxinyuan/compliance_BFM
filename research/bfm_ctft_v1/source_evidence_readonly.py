"""CPU archive-only qualification failure extraction; never a restore reader.

Tensor payloads are intentionally omitted because only plain physical rows are
analyzed. This reader cannot qualify model/optimizer/state restore equivalence.
"""
import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np


def omitted(*a,**kw):return None


class PhysicsOnly(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='torch.storage' and name=='_load_from_bytes':return omitted
        if module=='torch._utils' and name.startswith('_rebuild_'):return omitted
        return super().find_class(module,name)


def read(path):
    with gzip.open(path,'rb') as f:return PhysicsOnly(f).load()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--compare',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);begun=time.monotonic()
    x=read(a.input);ph=x['physics'];b=x['boundary'];last=ph[-1]
    tail=ph[-11:]
    result=dict(status='ARCHIVED_PHYSICS_ONLY_DIAGNOSTIC',input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        case=x['case'],physics_rows=len(ph),controls=len(x['controls']),boundary_kind=None if b is None else b['kind'],
        reason=last.get('ctft_stop_reason'),last_time_s=last['time'],
        source_time_since3s=last['time']-3.,source_exit_starts_relative_s=7.5,
        joint_excess_max_rad=max(r['joint_limit_excess_max_rad'] for r in ph),
        max_force_n=max(float(np.linalg.norm(r['force'])) for r in ph),
        last_55ms_normal_force_n=[r['foot_normal_force_n'] for r in tail],
        last_55ms_geometric_contact=[r['support'] for r in tail],
        last_55ms_causal_guard=[r.get('ctft_online_support') for r in tail],
        final_root_speed_mps=float(np.linalg.norm(last['qvel'][:2])),
        final_root_height_m=last['height'],final_tilt_deg=last['tilt_deg'],
        key_boundary='55ms insufficient current load during native exit, not delayed .10s stance qualification',
        no_checkpoint_restore_claim=True,new_physics_ticks=0,new_updates=0)
    if a.compare:
        y=read(a.compare);other=y['physics'];n=min(len(ph),len(other))
        keys=['qpos','qvel','root','wrist','force','foot_normal_force_n']
        error={k:float(np.max(np.abs(np.asarray([r[k] for r in ph[:n]])-np.asarray([r[k] for r in other[:n]])))) for k in keys}
        result['comparison']=dict(file_sha256=hashlib.sha256(a.compare.read_bytes()).hexdigest(),
            common_physics_rows=n,both_lengths_equal=len(ph)==len(other),field_max_abs_difference=error,
            same_physics=all(v==0 for v in error.values()))
    result['wall_s']=time.monotonic()-begun
    (a.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
