"""Bounded CPU-only exact-byte compression comparison; no model update."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from packed_checkpoint import PackedTensorStore
from lossless_chunks import encode, decode


def shuffled(raw):
    assert len(raw) % 4 == 0
    return np.frombuffer(raw, np.uint8).reshape(-1, 4).T.copy().tobytes()


def unshuffled(raw):
    return np.frombuffer(raw, np.uint8).reshape(4, -1).T.copy().tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    start = time.monotonic()
    store = PackedTensorStore(a.store)
    meta = store.metadata(a.name)
    entries = []
    def visit(v, path=''):
        if isinstance(v, dict) and v.get('__ctft_tensor__'):
            if v['chunks']:
                # First chunk per tensor stratifies model, targets and Adam.
                r = v['chunks'][0]
                if r['n'] % 4 == 0:
                    entries.append((path, r))
        elif isinstance(v, dict):
            for k, x in v.items(): visit(x, path + '/' + str(k))
        elif isinstance(v, (tuple, list)):
            for i, x in enumerate(v): visit(x, path + '/' + str(i))
    visit(meta)
    idx = np.linspace(0, len(entries)-1, min(96, len(entries)), dtype=int)
    rows = []
    for i in idx:
        key, ref = entries[i]
        raw = store.raw(ref)
        t = time.monotonic()
        _, plain = encode(raw)
        _, packed = encode(shuffled(raw))
        assert unshuffled(decode(packed, len(raw))) == raw
        baseline = None
        old_n, new_n = len(plain), len(packed)
        if ref['base']:
            baseline = store.raw(ref['base'])
            delta = np.bitwise_xor(np.frombuffer(raw, np.uint8),
                                   np.frombuffer(baseline, np.uint8)).tobytes()
            _, pd = encode(delta)
            _, sd = encode(shuffled(delta))
            assert unshuffled(decode(sd, len(delta))) == delta
            old_n, new_n = min(old_n, len(pd)), min(new_n, len(sd))
        rows.append(dict(path=key, raw_bytes=len(raw), plain_bytes=len(plain),
                         shuffle_bytes=len(packed), plain_best_bytes=old_n,
                         shuffle_best_bytes=new_n, had_xor_base=baseline is not None,
                         exact=True, cpu_s=time.monotonic()-t))
    out = dict(status='PASS_EXACT_SAMPLE', rows=rows, independent_training=False,
               raw_bytes=sum(x['raw_bytes'] for x in rows),
               plain_bytes=sum(x['plain_bytes'] for x in rows),
               shuffle_bytes=sum(x['shuffle_bytes'] for x in rows),
               plain_best_bytes=sum(x['plain_best_bytes'] for x in rows),
               shuffle_best_bytes=sum(x['shuffle_best_bytes'] for x in rows),
               elapsed_s=time.monotonic()-start,
               boundary='Stratified tensor first-chunk sample; not a full checkpoint or future guarantee')
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}), flush=True)


if __name__ == '__main__': main()
