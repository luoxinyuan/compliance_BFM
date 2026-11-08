"""Render existing real qpos only, using previously verified renderer."""
import argparse,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
r=Path(__file__).resolve().parents[1];s=r.parent
os.environ.update(BFM_RECOVERY_OUTPUT=str(a.output),BFM_ROLLING_VIDEO_RUN=str(s/'runs/video-final-input'),BFM_RENDER_ASSET_ROOT=str(r))
import rolling_movie
rolling_movie.main()
