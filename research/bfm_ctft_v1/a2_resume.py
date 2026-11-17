"""Explicit fresh-process A2 restore entry; no physics or formal training."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes'),str(Path(__file__).parent)]
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from learning_checkpoint import resume_and_compare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);a=p.parse_args()
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');torch.manual_seed(681117)
    resume_and_compare(a.package,a.output,reference_path=a.reference)
