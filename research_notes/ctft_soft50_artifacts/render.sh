set -eu
base=/mnt/data/temp_user/yi.gao/projects/compliance_BFM
stage="$base/experiments/ctft_soft50_01"
cd "$stage"
source "$base/experiments/ctft_quiet_01/env69.sh"
set -u
export BFM_ROLLING_VIDEO_RUN="$stage/video"
export BFM_RECOVERY_OUTPUT="$stage/video"
export BFM_RENDER_ASSET_ROOT="$base/experiments/native-rolling-h100-20261107/repo"
mkdir -p video
python - <<'PY'
from pathlib import Path
import json
root=Path.cwd()
arms=[("baseline","Original #50 rolling feedback"),("soft","50% command angle during first 0.5s")]
item={"name":"soft-start-72cm","sides":[{"title":title,"role":"online feedback","path":str(root/"paired02"/(arm+".json.gz"))} for arm,title in arms]}
(root/"video/video-pairs.json").write_text(json.dumps([item],indent=2))
PY
python "$base/maintenance/research-handoff-20261102/research_notes/rolling_movie.py"
