"""Fail-closed supervision utilities; no learning or physics behavior changes."""
import os
import signal
import stat
import time
from pathlib import Path


def tree_bytes(roots, retries=4):
    """Restart the entire scan on atomic rename; never report a partial scan.

    Missing roots are empty only at scan entry. Other IO errors propagate and
    must stop/reap the owned child. Hardlinks are counted once per snapshot.
    """
    for attempt in range(retries):
        try:
            seen = set()
            total = 0
            pending = [Path(p) for p in roots if Path(p).exists()]
            while pending:
                with os.scandir(pending.pop()) as entries:
                    for entry in entries:
                        st = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(st.st_mode):
                            pending.append(Path(entry.path))
                        elif stat.S_ISREG(st.st_mode):
                            key = (st.st_dev, st.st_ino)
                            if key not in seen:
                                seen.add(key)
                                total += st.st_size
            return total
        except FileNotFoundError:
            if attempt == retries - 1:
                raise
            time.sleep(.02)


def stop_owned_group(process, grace=90):
    """Only for our Popen(start_new_session=True) handle; always reap it."""
    if process.poll() is None:
        try:
            assert os.getpgid(process.pid) == process.pid
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except __import__('subprocess').TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return process.wait()
