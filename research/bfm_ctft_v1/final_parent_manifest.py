"""Freeze the already-qualified source identities with exogenous partners.

Reads identities and source-split audit only; no new or archived learned-policy
outcomes, no old sealed sets. Siblings remain one independent ancestor.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def create(stage, output):
    stage = Path(stage)
    output = Path(output)
    assert not output.exists()
    source_path = stage / 'locks/final-source-state-manifest-v2.json'
    sources = json.loads(source_path.read_text())
    reset = json.loads((stage / 'locks/source-reset-domain-v1.json').read_text())
    train = {x['ancestor'] for x in reset['train_sources']}
    states = sources['states']
    assert len(states) == len({x['ancestor'] for x in states}) == 24
    assert train.isdisjoint(x['ancestor'] for x in states)
    parents = []
    for i, state in enumerate(states):
        # Fixed balanced factorial coverage; no score or trajectory outcome is
        # inspected to assign a partner to a source.
        speed = [.08, .12, .16][i % 3]
        distance = [.60, .75, .90][(i // 3) % 3]
        short = [.30, .375, .45][i % 3]
        common = dict(plateau_speed_mps=speed, accel_s=.5, hold_s=3.,
                      stiffness=[40., 80.][(i // 2) % 2], damping=8., cap_n=10.,
                      direction_deg=[-15., 0., 15.][(i // 3) % 3], recovery_s=4.)
        branches = []
        for name, force, length in [('continue', True, distance), ('stop', True, short),
                                    ('no_force', False, distance)]:
            partner = dict(common, distance_m=length)
            branches.append(dict(branch=name, mode=1, apply_force=force, partner=partner))
        # Check real constant-speed/ramp implementation: exact common prefix
        # lasts until the short branch starts decelerating, not until release.
        v = speed
        common_end = short / v
        r = .5
        def position(L, t):
            if t < r:
                u = t / r
                return v * r * (u**3 - .5*u**4), v*(3*u*u-2*u**3)
            if t < L/v:
                return .5*v*r+v*(t-r), v
            u = (t-L/v)/r
            return L-.5*v*r+v*r*(u-u**3+.5*u**4), v*(1-3*u*u+2*u**3)
        for t in np.arange(0., common_end, .005):
            assert position(short, t) == position(distance, t)
        parents.append(dict(parent_id=f'ctft-eval-parent-{i:02d}', source=copy.deepcopy(state),
                            common_prefix_s=common_end, branches=branches))
    result = dict(version='CTFT_FINAL_PARENTS_V1', status='LOCKED_NOT_EVALUATED', seed=681117,
                  source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  independent_source_ancestors=24, parents=parents,
                  contact_episodes_per_model=72, models=['B0', 'U', 'T'],
                  same_complete_source_rng_for_all_siblings_and_models=True,
                  no_source_motion_id_or_partner_future_in_actor=True,
                  initial_anchor='actual same source wrist; no mid-episode reanchor',
                  native_g0_evaluation='Separate12 qualified native cases still require final identity lock',
                  source_labels_are_not_contact_success=True,
                  unseen_claim='held out from this adaptation, not necessarily original BFM pretraining')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def add_native_regressions(stage, contact_manifest, output):
    stage, contact_manifest, output = Path(stage), Path(contact_manifest), Path(output)
    assert not output.exists()
    result = json.loads(contact_manifest.read_text())
    reset = json.loads((stage / 'locks/source-reset-domain-v1.json').read_text())
    # All12 original source ancestors, not a post-score subset. They were used
    # in adaptation source preparation; this is explicitly native regression,
    # not12 additional independent held-out contact parents.
    result['native_g0_evaluation'] = [dict(
        parent_id=f'ctft-native-regression-{i:02d}', source=source,
        source_control=25, mode=0, force=False,
        command='Original saved native motion z sequence at its real source phase; source schedule then returns to standing',
        horizon='Through original13s source schedule; preserve original failed tails as evaluation failures',
        source_starting_state_qualification='Requires actual complete restore, never qpos-only',
        prior_exposure='Training source ancestor, native capability regression only')
        for i, source in enumerate(reset['train_sources'])]
    result['version'] = 'CTFT_FINAL_PARENTS_V2'
    result['previous_manifest_sha256'] = hashlib.sha256(contact_manifest.read_bytes()).hexdigest()
    result['native_regression_ancestors'] = 12
    result['native_small_turn_coverage'] = 'Actual native responses must be reported; motion names do not establish directional-turn qualification'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
