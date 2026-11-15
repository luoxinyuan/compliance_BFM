"""Lossless storage for restoration validation, referencing immutable expected data.

Preserves every actual tensor/field, including any failed comparison. It changes
storage only; it does not replace individual checks with a whole-model average.
"""
from pathlib import Path
import json,hashlib,os
from . import delta_tree

def save(out,actual,expected,base_file,base_meta,exact):
    out=Path(out);base_file=Path(base_file)
    raw=base_file.read_bytes();assert len(raw)==base_meta['bytes'] and hashlib.sha256(raw).hexdigest()==base_meta['sha256']
    path=out/'actual-restored-update.delta'
    meta,decoded=delta_tree.dump(path,actual,expected)
    assert exact(actual,decoded),'lossless encoding changed actual validation state'
    reread=delta_tree.load(path,meta,expected)
    assert exact(actual,reread),'lossless validation readback changed actual state'
    manifest=dict(format='OR1-validation-lossless-reference-v1',complete=True,base=dict(path=str(base_file),**base_meta),actual_delta=dict(path=path.name,**meta),all_fields_preserved=True,physics_or_update_changed=False)
    mp=out/'actual-restored-update.manifest.json';assert not mp.exists()
    with mp.open('x')as f:f.write(json.dumps(manifest,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    return dict(path=mp.name,sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),bytes=mp.stat().st_size,actual_delta_bytes=meta['bytes'],all_fields_lossless=True)
