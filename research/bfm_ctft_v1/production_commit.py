"""Round storage policy only; no change to native learning or physics.

Every generation is a full logical state. XOR references use only a permanent
milestone. Retention can retire successful intermediate packs of THIS store;
failed/pending packages and replay segments are never removed.
"""
import json
from pathlib import Path
from packed_checkpoint import PackedTensorStore, durable, sha


class RoundCommitter:
    def __init__(self, root, *, total_rows, milestone_rows, create=True):
        self.root = Path(root)
        self.store = PackedTensorStore(self.root)
        self.policy = dict(version='CTFT_ROUND_RETENTION_V1', total_rows=total_rows,
                           milestone_rows=sorted(set(milestone_rows)), latest=2,
                           only_this_store=True, preserve_all_failure_and_replay=True)
        assert 0 in self.policy['milestone_rows']
        assert total_rows in self.policy['milestone_rows']
        path = self.root / 'policy.json'
        if create:
            assert not path.exists()
            durable(path, json.dumps(self.policy, sort_keys=True).encode())
            self.completed = []
            self.permanent = []
        else:
            assert json.loads(path.read_text()) == self.policy
            latest = json.loads((self.root / 'latest.json').read_text())
            last_rows = int(latest['package'].split('-')[1])
            assert latest['manifest_sha256'] == sha((self.root / latest['package'] / 'manifest.json').read_bytes())
            # latest is the sole atomic authority. A crash after its publish but
            # before the advisory index cannot lose a committed full state.
            complete = []
            for p in self.root.glob('round-*'):
                if not p.name.split('-')[-1].isdigit() or int(p.name.split('-')[-1]) > last_rows:
                    continue
                m = json.loads((p / 'manifest.json').read_text())
                assert m['complete']
                complete.append((int(p.name.split('-')[-1]), p.name, m['anchor']))
            complete.sort()
            self.completed = [x[1] for x in complete]
            self.permanent = [x[1] for x in complete if x[2]]
            assert self.completed[-1] == latest['package']
            for name in self.completed[-2:]:
                self.store.metadata(name)

    def commit(self, rows, full_state):
        assert 0 <= rows <= self.policy['total_rows']
        name = f'round-{rows}'
        assert name not in self.completed
        if self.completed:
            assert rows > int(self.completed[-1].split('-')[1])
        else:
            assert rows == 0
        anchor = rows in self.policy['milestone_rows']
        # Never reference the preceding transient generation. All dependencies
        # remain permanent, bounding storage independently of round count.
        base = self.store.metadata(self.permanent[-1]) if self.permanent else None
        _, measured = self.store.save(name, full_state, base=base, anchor=anchor)
        self.completed.append(name)
        if anchor:
            self.permanent.append(name)
        retired = self.store.retain(self.completed, self.permanent) if len(self.completed) >= 2 else []
        record = dict(policy_sha256=sha((self.root / 'policy.json').read_bytes()),
                      completed=self.completed, permanent=self.permanent,
                      latest=name, last_retired=retired)
        durable(self.root / 'commits.json', json.dumps(record, sort_keys=True).encode())
        return dict(measured, retired=retired, permanent=list(self.permanent))

    def load_latest(self):
        return self.store.load(self.completed[-1])


def qualify_fixture(output):
    """Small storage fixture; explicitly not physical/learner qualification."""
    import torch
    import time
    from learning_checkpoint import digest
    output = Path(output)
    c = RoundCommitter(output, total_rows=6144, milestone_rows=[0, 2048, 4096, 6144])
    original = torch.arange(40000, dtype=torch.float32)
    records = []
    for rows in [0, 1024, 2048, 3072, 4096, 5120]:
        state = dict(payload=original + rows, optimizer=dict(step=rows),
                     fixture_only=True, not_a_training_checkpoint=True)
        records.append(dict(rows=rows, result=c.commit(rows, state)))
        assert digest(c.load_latest()) == digest(state)
    # Advisory index loss is an interrupted administrative publication, not a
    # reason to roll back a verified latest Task/learner checkpoint.
    (output / 'commits.json').unlink()
    resumed = RoundCommitter(output, total_rows=6144, milestone_rows=[0, 2048, 4096, 6144], create=False)
    assert digest(resumed.load_latest()) == digest(state)
    assert (output / 'round-1024/retired.json').exists()
    assert not (output / 'round-1024/data.bin').exists()
    assert (output / 'round-0/data.bin').exists()
    assert (output / 'round-2048/data.bin').exists()
    assert (output / 'round-4096/data.bin').exists()
    return dict(status='STORAGE_FIXTURE_PASS', records=records,
                physical_qualification=False, learning_qualification=False)
