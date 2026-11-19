set -eu
cd "$(dirname "$0")"
source env69.sh
test -s audit/identity_and_projection.json
test -s gradient.json
export LOCK_SHA=4b49d8c4e79d298b6e8ccb34eeda90eaeed647fbe868c2323175c21c28f6fcb3
export DIAG_MODE=d2 DIAG_TIMEOUT=11500
test "$(df --output=avail -B1 . | tail -1)" -ge 107374182400
exec timeout 11700 flock -n "$base/gpu0-experiment.lock" "$base/.venv/bin/python" -u run_alpha.py
