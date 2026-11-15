"""Real accepted Actor/Adam/physics checkpoint reference equivalence, CPU only."""
import time
START=time.monotonic()
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research_notes'))
import torch
torch.set_num_threads(1)
from f1_mjdata_contract import exact
from .ei_v2_or1_expected import read,write
from .ei_v2_or1_validation_delta import save
from . import delta_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 old=json.loads((a.source/'result.json').read_text());old=old.get('study',old);legacy=read(a.source,old['expected']);meta=write(a.out,a.source/'checkpoints/block-2',legacy['metrics'],legacy,exact);rebuilt=read(a.out,meta);assert exact(legacy,rebuilt)
 v=a.out/'validation';v.mkdir();md=save(v,legacy,rebuilt,a.out/meta['path'],{k:meta[k]for k in ('sha256','bytes')},exact);manifest=json.loads((v/md['path']).read_text());actual=delta_tree.load(v/manifest['actual_delta']['path'],manifest['actual_delta'],read(a.out,meta));assert exact(actual,legacy)
 metrics_bytes=(a.out/'expected-last-update.metrics.ptz').stat().st_size
 result=dict(status='REAL_COMPLETE_CHECKPOINT_TO_ORIGINAL_SIGNATURE_BITWISE_CPU_PASS_NOT_GPU_RECOVERY',original_and_reference_agent_Adam_rng_physics_metrics_lastrows_exact=True,validation_delta_against_reference_exact=True,legacy_expected_bytes=old['expected']['bytes'],new_reference_bytes=meta['bytes'],new_metrics_bytes=metrics_bytes,new_actual_delta_bytes=md['actual_delta_bytes'],existing_checkpoint_referenced_not_copied=True,no_new_physics=True,wall_s=time.monotonic()-START)
 (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
