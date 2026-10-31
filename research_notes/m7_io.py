"""Atomic owned ledger snapshots: concurrent readers must never see truncate/write gaps."""
import json, os
from pathlib import Path

def write(p, value):
    p = Path(p)
    tmp = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, p)
