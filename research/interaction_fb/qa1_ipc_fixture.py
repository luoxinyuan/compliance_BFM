"""Read-only startup regression: existing project socket policy in forkserver children."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research_notes')]
import m11_ipc
if __name__=='__main__':
    import multiprocessing as mp
    mp.set_start_method('forkserver',force=True)
    with mp.Manager() as manager:
        q=manager.Queue();q.put('qa1-readonly');assert q.get()=='qa1-readonly'
    print('PASS existing abstract socket forkserver; physics=0 optimizer=0')
