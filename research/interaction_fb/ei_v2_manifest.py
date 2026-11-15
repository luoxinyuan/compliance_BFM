"""Signed v2 P0 recipe/ancestry audit; never reads or executes sealed cases."""
import argparse, collections, copy, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT.parent
SIGNED_SHA = '69f10a9143932351c740d5208d561f81f9aeeafe69da49c3efc9da773ac5c3b9'

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def prepare():
    notes = ROOT / 'research_notes'
    assert hashlib.sha256((notes / 'CONTACT_GOAL_EI_TRAIN_CONTRACT_V2_APPROVED.md').read_bytes()).hexdigest() == SIGNED_SHA
    audit = json.loads((notes / 'r2_splits/audit.json').read_text())
    out = STAGE / 'locks'; out.mkdir(exist_ok=True)
    expanded = {}; original = {}; counters = {}
    modes = ['FOLLOW', 'HOLD', 'FOLLOW_TO_HOLD', 'HOLD_TO_FOLLOW']
    offsets = {'near': 0, 'mid': 1, 'stop': 2}
    for split, expected in [('train', (34, 44)), ('dev', (16, 20))]:
        path = notes / f'r2_splits/{split}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['splits'][split]['file_sha256']
        parents = json.loads(path.read_text())['parents']; count = collections.Counter(); rows = []
        assert (len(parents), sum(len(p['specs']) for p in parents)) == expected
        for parent in parents:
            family = parent['family']; i = count[family]; count[family] += 1
            K = 80 if i % 2 == 0 else 40
            source = 'goal_arm5' if family in offsets and i < 2 else 'standing'
            mode = modes[(i + offsets[family]) % 4] if family in offsets else 'HOLD'
            row = copy.deepcopy(parent)
            row.update(split=split, source=source, stiffness=K, goal_stream=mode,
                       goal_switch_after_handoff_s=1.5,
                       identity='signed_v2_recipe_not_physical_independence_proof')
            for sp in row['specs']:
                sp.update(source=source, h0_stiffness=float(K))
            row['derived_recipe_sha256'] = digest(row)
            rows.append(row)
        expanded[split] = rows; original[split] = parents
        counters[split] = dict(parents=len(rows), episodes=sum(len(p['specs']) for p in rows),
            families=dict(count), source_counts=dict(collections.Counter(p['source'] for p in rows)),
            stiffness_counts=dict(collections.Counter(p['stiffness'] for p in rows)),
            one_complete_sweep_50Hz=sum(round((s['end'] - s['source_until']) * 50) for p in rows for s in p['specs']))
        (out / f'expanded-{split}.json').write_text(json.dumps(dict(parents=rows), indent=2) + '\n')
    for key in ['parent_id', 'source_parent_id', 'partner_parent_id']:
        assert not ({p[key] for p in original['train']} & {p[key] for p in original['dev']})
    assert counters['train']['one_complete_sweep_50Hz'] <= 32000
    assert counters['dev']['source_counts'] == {'standing': 10, 'goal_arm5': 6}
    assert counters['dev']['stiffness_counts'] == {80: 8, 40: 8}
    # This is exactly the information read by the existing goal_arm5 branch.
    # Pulse fraction/duration, phase offset/rate and K are not used before contact.
    def effective_arm(p):
        s = p['specs'][0]
        return dict(native_asset='goal_arm5', arm_fraction=s['source_profile']['arm_fraction'],
                    time_law='smoothstep(clamp((t-3)/1.5,0,1))',
                    source_until=s['source_until'], reset='m8_stage.reset_common(seed=0)')
    pairs = []
    for t in expanded['train']:
        if t['source'] != 'goal_arm5': continue
        for d in expanded['dev']:
            if d['source'] == 'goal_arm5' and effective_arm(t) == effective_arm(d):
                pairs.append(dict(train_parent=t['parent_id'], dev_parent=d['parent_id'],
                    train_source_parent=t['source_parent_id'], dev_source_parent=d['source_parent_id'],
                    effective_recipe=effective_arm(t)))
    train_arm = [p for p in expanded['train'] if p['source'] == 'goal_arm5']
    short = min(train_arm, key=lambda p: p['specs'][0]['source_until'])
    long = next(p for p in train_arm if p['stiffness'] == short['stiffness'] and p['specs'][0]['source_until'] >= short['specs'][0]['source_until'] + .4)
    result = dict(status='P0_RECIPE_AUDIT_SOURCE_OVERLAP_SUSPECT', signed_sha256=SIGNED_SHA,
        counts=counters, original_structural_id_audit_pass=True,
        original_physical_independence_note=audit['physical_state_independence'],
        exact_effective_recipe_cross_split_pairs=pairs,
        common_native_arm_curve_count=1,
        proposed_arm_recipe_parent_counts={'train':len(train_arm), 'dev':sum(p['source']=='goal_arm5' for p in expanded['dev'])},
        sealed_cases_read=False, sealed_summary_only=audit['splits']['sealed'],
        physical_check=dict(short_parent=short['parent_id'], long_parent=long['parent_id'],
            short_end=short['specs'][0]['source_until'], long_end=long['specs'][0]['source_until'],
            scope='two TRAIN-only native prefixes; no Dev physics, no partner activation, no policy update'),
        future_gate='G01 must use actual native trajectory ancestry, not ignored profile hashes')
    (out/'source-recipe-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
