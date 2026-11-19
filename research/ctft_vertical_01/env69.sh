base=/mnt/data/temp_user/yi.gao/projects/compliance_BFM
source "$base/env.sh"
export CTFT_PINNED_RUNTIME="$base/experiments/bfm_ctft_v1/runtime-v1a" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH="$CTFT_PINNED_RUNTIME:$CTFT_PINNED_RUNTIME/research_notes:$CTFT_PINNED_RUNTIME/research/bfm_ctft_v1:$PWD"
