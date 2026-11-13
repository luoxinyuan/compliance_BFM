"""CPU-only Return pilot archive qualification; never constructs a simulator/agent."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_read(path, expected):
    assert path.stat().st_size == expected['bytes'], str(path)
    assert sha(path) == expected['sha256'], str(path)
    return torch.load(io.BytesIO(zlib.decompress(path.read_bytes())),
                      map_location='cpu', weights_only=False)


def load_core(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['complete']
    return manifest, checked_read(path / 'core.ptz', manifest['files']['core.ptz'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    assert not args.output.exists(), 'Immutable output already exists'
    assert sha(args.draft) == '224fb0d39b2adccbbc8c58e6a11c7dcd66736d50d7a142ad15d0855f42bb71a9'
    draft = json.loads(args.draft.read_text())
    root = args.project / 'experiments/interaction-pilot-v3-20261112/runs'
    b19 = root / 'pilot-B-v3/checkpoints/round-19'
    manifest, core = load_core(b19)
    assert sha(b19 / 'core.ptz') == '0cd3e3684d764cad42caaa83773b885868caeed0557dba1780ec36a3d03a3984'
    assert core['agent']['updates'] == 38
    assert core['agent']['exploration'] == dict(std=.01, clip=.06)
    lists = {k: [] for k in ('rows', 'native')}
    for seg in manifest['segments']:
        part = checked_read(b19.parent / seg['path'], seg)
        for key in lists:
            assert len(part[key]) == seg['counts'][key]
            lists[key].extend(part[key])
    assert all(len(lists[k]) == manifest['lengths'][k] for k in lists)
    specs = core['task_specs']
    train = {p['parent_id'] for p in draft['parents'] if p['role'] == 'return_train'}
    train.add('7f3a7da65fb1b4b3d41944dcb3d9d623cb2249a42503eb9a5969909888075b1d')
    dev = {p['parent_id'] for p in draft['parents'] if p['role'] != 'return_train'}
    assert not train & dev
    rows = lists['rows']
    replay = [i for i, r in enumerate(rows)
              if specs[int(r['provenance']['spec_index'])]['parent_id'] in train]
    assert replay
    # Native rows have no parent_id. Check source blocks against the actual
    # replay episode order before inheriting group membership.
    episodes = []
    for r in rows:
        key = (int(r['provenance']['online_episode']), int(r['provenance']['spec_index']))
        if not episodes or episodes[-1] != key:
            episodes.append(key)
    assert len({x[0] for x in episodes}) == len(episodes)
    native = lists['native']
    blocks = []
    for i, row in enumerate(native):
        if not blocks or row['time'] <= native[blocks[-1][-1]]['time']:
            blocks.append([])
        blocks[-1].append(i)
    assert len(blocks) == len(episodes), 'Cannot infer native parent mapping'
    native_indices, mapping = [], []
    for block, (episode, sid) in zip(blocks, episodes):
        spec = specs[sid]
        assert {native[i]['source'] for i in block} == {spec['source']}
        assert all(int(native[i]['observation']['_slot']) == 0 for i in block)
        assert all(not bool(native[i]['observation']['contact_privileged_v1'][0, 21]) for i in block)
        times = [float(native[i]['time']) for i in block]
        assert all(abs((b - a) - .02) < 1e-7 for a, b in zip(times, times[1:])), 'Native block not contiguous'
        selected = spec['parent_id'] in train
        if selected:
            native_indices.extend(block)
        mapping.append(dict(episode=episode, spec_index=sid, parent_id=spec['parent_id'],
                            count=len(block), time_range=[times[0], times[-1]], selected=selected))
    assert native_indices
    archives = []
    for item in draft['parents']:
        arm = 'A' if item['role'] == 'none_dev' else 'B'
        path = root / f'pilot-{arm}-v3/checkpoints' / f"round-{item['archived_round']}"
        m, c = load_core(path)
        spec = c['task_specs'][c['spec_id']]
        assert spec['parent_id'] == item['parent_id']
        assert item['time'] + 3 <= spec['end'] + 1e-8
        # This is archive availability, not a physics restoration pass.
        archives.append(dict(**item, core_sha256=sha(path / 'core.ptz'),
                             manifest_sha256=sha(path / 'manifest.json'), source_sha256=c['source_sha'],
                             archived_commands=len(c['commands']), archived_actions=len(c['actions']),
                             original_end=spec['end'], status='AVAILABLE_NOT_PHYSICALLY_RECONSTRUCTED'))
    result = dict(status='CPU_ARCHIVE_CHECKS_PASSED_GPU_QUALIFICATION_PENDING',
                  draft_sha256=sha(args.draft), parent_model_core_sha256=sha(b19 / 'core.ptz'),
                  original_replay_rows=len(rows), selected_replay_rows=len(replay),
                  original_native_rows=len(native), selected_native_rows=len(native_indices),
                  replay_indices=replay, native_indices=native_indices, native_parent_mapping=mapping,
                  training_parent_ids=sorted(train), development_parent_ids=sorted(dev),
                  archives=archives, new_physics_ticks=0, new_optimizer_steps=0,
                  cuda_initialized=torch.cuda.is_initialized(), wall_s=time.monotonic() - start)
    assert not result['cuda_initialized']
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('replay_indices', 'native_indices', 'archives', 'native_parent_mapping')}, indent=2))


if __name__ == '__main__':
    main()
