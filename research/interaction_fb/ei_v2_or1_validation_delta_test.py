"""Real trusted signature roundtrip including a one-bit failure, CPU only."""
import time
START=time.monotonic()
from pathlib import Path
import argparse,io,zlib,json,copy,hashlib,ast,sys
import torch
torch.set_num_threads(1)
from .ei_v2_or1_validation_delta import save
from . import delta_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    contract=Path(__file__).resolve().parents[2]/'research_notes/f1_mjdata_contract.py';f=next(n for n in ast.parse(contract.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='exact');ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(contract),'exec'),ns);exact=ns['exact']
    b=a.fixture.read_bytes();meta=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));expected=torch.load(io.BytesIO(zlib.decompress(b)),map_location='cpu',weights_only=False)
    good=a.out/'equal';good.mkdir();m=save(good,expected,expected,a.fixture,meta,exact);manifest=json.loads((good/m['path']).read_text());restored=delta_tree.load(good/manifest['actual_delta']['path'],manifest['actual_delta'],expected);assert exact(restored,expected)
    bad=copy.deepcopy(expected);name=next(k for k,v in bad['agent']['modules'].items()if torch.is_tensor(v)and v.dtype==torch.float32 and v.numel());t=bad['agent']['modules'][name];bits=t.reshape(-1).view(torch.int32);bits[0]^=1;assert not exact(bad,expected)
    diff=a.out/'one-bit';diff.mkdir();md=save(diff,bad,expected,a.fixture,meta,exact);mm=json.loads((diff/md['path']).read_text());again=delta_tree.load(diff/mm['actual_delta']['path'],mm['actual_delta'],expected);assert exact(again,bad)and not exact(again,expected)
    corrupt=dict(mm['actual_delta']);corrupt['sha256']='0'*64
    try:delta_tree.load(diff/mm['actual_delta']['path'],corrupt,expected);raise RuntimeError('bad delta hash passed')
    except AssertionError:pass
    result=dict(status='REAL_ACTOR_ADAM_PHYSICS_SIGNATURE_LOSSLESS_CPU_PASS_NOT_GPU_RESTORE',all_equal_fields_exact=True,one_bit_mismatch_preserved=True,corrupt_hash_rejected=True,base_bytes=meta['bytes'],equal_delta_bytes=m['actual_delta_bytes'],one_bit_delta_bytes=md['actual_delta_bytes'],no_new_physics=True,wall_s=time.monotonic()-START)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
