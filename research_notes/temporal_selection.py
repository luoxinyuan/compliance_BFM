"""Preregistered absolute physical development rule; no near-zero ratios."""
from m39a_metrics import cost


def assess(rows, initial):
    old = {r['label']: r for r in initial}
    reasons = []
    for r in rows:
        base = old[r['label']]
        if r['boundary']:
            reasons.append(dict(label=r['label'], reason='existing physical safety boundary'))
        if not r['condition']['enabled']:
            if r['root_max_displacement'] > max(.05, base['root_max_displacement']+.03):
                reasons.append(dict(label=r['label'], reason='no-contact root drift>5cm or +3cm'))
            if r.get('torso_excursion_deg', 0) > base.get('torso_excursion_deg', 0)+10:
                reasons.append(dict(label=r['label'], reason='no-contact torso excursion +10deg'))
        else:
            family = r['label'].split('__')[0]
            allowance = max(2., .2*base['error_cm']) if family in ['near', 'mid', 'arm'] else max(5., .15*base['error_cm'])
            if r['error_cm'] > base['error_cm']+allowance:
                reasons.append(dict(label=r['label'], reason=f'active tracking increase>{allowance:.3f}cm'))
            if r['force_rms'] > base['force_rms']+1:
                reasons.append(dict(label=r['label'], reason='active force RMS increase>1N'))
            if r['cap_fraction'] > base['cap_fraction']+.15:
                reasons.append(dict(label=r['label'], reason='cap duration increase>15 percentage points'))
    values = [cost(r) for r in rows]
    return dict(score=sum(values)/len(values)+.25*max(values),
                admissible=not reasons, reasons=reasons,
                per_case_cost={r['label']: cost(r) for r in rows},
                interpretation='physical-unit relative deterioration allowances, not scientific success thresholds')
