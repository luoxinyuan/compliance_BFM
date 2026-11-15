"""Counter IO only: serialize atomic rename with readers, never mask stale counts.
No changes to reservation quantum, physics, policy, RNG or scientific gates.
"""
import contextlib,fcntl,json,os
from pathlib import Path

@contextlib.contextmanager
def locked(path):
    path=Path(path)
    with path.with_suffix('.io-lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)

def atomic_write(path,data):
    path=Path(path)
    with locked(path):
        tmp=path.with_suffix('.pending')
        tmp.write_text(json.dumps(data,indent=2)+'\n')
        os.replace(tmp,path)
