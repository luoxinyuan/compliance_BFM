"""CPU-only conversion of actual immutable CAS generations, no new learning."""
import argparse
import json
from pathlib import Path
import time
from cas_checkpoint import TensorStore
from packed_checkpoint import PackedTensorStore
from learning_checkpoint import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=TensorStore(a.source);target=PackedTensorStore(a.output/'store');base=None;rows=[]
    for name in ['round-2','round-3']:
        x=source.load(name);before=digest(x)
        packed,stats=target.save(name,x,base=base,anchor=name=='round-2')
        assert digest(target.load(name))==before
        if name=='round-2':base=packed
        rows.append(dict(generation=name,full_state_exact=True,**stats));del x
        (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'result.json').write_text(json.dumps(dict(status='PACKED_TWO_GENERATIONS_BYTE_VALUES_EXACT',
        cases=rows,updates=0,new_physics_ticks=0,wall_s=time.monotonic()-start,
        actual_gpu_task_and_next_update_qualification='PENDING_SEPARATE_NEW_PROCESS'),indent=2)+'\n')


if __name__=='__main__':main()
