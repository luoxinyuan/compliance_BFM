"""Exact expected-validation signatures from immutable complete checkpoints."""
from pathlib import Path
import json,hashlib,os

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(root,meta):
    from .journal import read as read_payload
    root=Path(root)
    if meta.get('format')!='OR1-expected-checkpoint-reference-v1':
        return read_payload(root/meta.get('path','expected-last-update.ptz'),meta)
    ref_file=root/meta['path'];assert ref_file.stat().st_size==meta['bytes'] and digest(ref_file)==meta['sha256']
    ref=json.loads(ref_file.read_text());assert ref['format']==meta['format'] and ref['complete']
    cp=Path(ref['checkpoint']['path']);assert digest(cp/'manifest.json')==ref['checkpoint']['manifest_sha256']
    from .ei_v2_checkpoint import load
    from f1_mjdata_contract import canonical
    state=load(cp)
    metrics=read_payload(root/ref['metrics']['path'],ref['metrics'])
    return dict(agent=state['agent'],physics=canonical(state['physical'],state['mj_metadata']),rng=state['rng'],metrics=metrics,last_rows=state['rows'][-12:])

def write(out,checkpoint_dir,metrics,baseline,exact):
    from .journal import dump
    out=Path(out);cp=Path(checkpoint_dir)
    mm=dump(out/'expected-last-update.metrics.ptz',metrics)
    ref=dict(format='OR1-expected-checkpoint-reference-v1',complete=True,checkpoint=dict(path=str(cp),manifest_sha256=digest(cp/'manifest.json')),metrics=dict(path='expected-last-update.metrics.ptz',**mm),all_expected_fields_reconstructed=True,no_physics_or_update_changed=True)
    path=out/'expected-last-update.ref.json';assert not path.exists()
    with path.open('x')as f:f.write(json.dumps(ref,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    meta=dict(format=ref['format'],path=path.name,sha256=digest(path),bytes=path.stat().st_size)
    actual=read(out,meta)
    if not exact(actual,baseline):
        dump(out/'failed-original-full-expected-signature.ptz',baseline)
        raise AssertionError('checkpoint reference did not reconstruct original full signature exactly')
    return meta
