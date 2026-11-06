#!/usr/bin/env bash
set -euo pipefail
P=/mnt/data/temp_user/yi.gao/projects/compliance_BFM
R=$P/bridge-v2-owned/trees/native-contact-learning/repo
O=$P/bridge-v2-owned/runs/native-contact-20261106
source /mnt/data/temp_user/yi.gao/maintenance/storage_env.sh
source "$P/env.sh"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$R:$R/research_notes:$PYTHONPATH"
python "$R/research_notes/native_job.py" --ledger "$O/ledger.json" --name native-initialization --limit 600 -- python "$R/research_notes/native_initialize.py" --root "$O"
for ARM in ordinary guided; do
 python "$R/research_notes/native_job.py" --ledger "$O/ledger.json" --name "ppo-$ARM" --limit 6500 -- python "$R/research_notes/native_ppo.py" --kind direct --exploration "$ARM" --stage "native-$ARM" --output "$O/ppo-$ARM" --gpu 0 --iterations 32 --seed 4801 --resume "$O/native-initial.pt" --max-seconds 6200
 done
