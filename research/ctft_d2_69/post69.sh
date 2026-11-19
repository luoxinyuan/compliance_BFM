set -eu
cd "$(dirname "$0")"
source env69.sh
test -s d2/complete.json
test -s d2/result.json
test "$(df --output=avail -B1 . | tail -1)" -ge 107374182400
timed() {
    local dest=$1
    shift
    TIMEFORMAT="%R"
    { time "$@" 2>&3; } 3>&2 2>"$dest"
}
timed verification.wall python verify69.py
timed pairs.wall python pairs69.py "$PWD"
python report69.py
export REVIEW_OUT="$PWD/videos"
timed video.wall flock -n "$base/gpu0-experiment.lock" python render69.py
