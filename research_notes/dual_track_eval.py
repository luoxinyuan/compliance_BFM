"""Original isolated retention evaluation, with explicit new-backend cache identity."""
import os
from pathlib import Path
import torch
import rolling_backend
original=rolling_backend.install

def current_backend(env,model,folder):
    prefix=Path(os.environ['DUAL_TRACK_PREFIX'])
    assert (prefix/'backend-standing-prefix.json').exists()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return original(env,model,prefix)
rolling_backend.install=current_backend
import retention_eval
if __name__=='__main__':retention_eval.main()
