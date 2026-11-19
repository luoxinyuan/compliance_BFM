#!/bin/bash
set -e -o pipefail
cd "$(dirname "$0")"
base=/mnt/data/temp_user/yi.gao/projects/compliance_BFM
source "$base/env.sh"
export CTFT_PINNED_RUNTIME="$base/experiments/bfm_ctft_v1/runtime-v1a" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$CTFT_PINNED_RUNTIME:$CTFT_PINNED_RUNTIME/research/bfm_ctft_v1:$PWD"
while [[ ! -f reference/complete.json ]]; do
    kill -0 "$(python -c 'import json;print(json.load(open("reference/launch.json"))["pid"])')" || exit 2
    sleep 10
done
while kill -0 "$(python -c 'import json;print(json.load(open("reference/launch.json"))["pid"])')" 2>/dev/null;do sleep 2;done
CUDA_VISIBLE_DEVICES="" python qualify.py "$PWD"
bash run.sh replay 3500 >replay.log 2>&1
bash run.sh d1 3000 >d1.log 2>&1
bash run.sh d2 7000 >d2.log 2>&1
CUDA_VISIBLE_DEVICES="" python compare.py "$PWD"
CUDA_VISIBLE_DEVICES="" python movie.py "$PWD"
