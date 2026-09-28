#!/usr/bin/env python3
"""Co-occurring-domain composition and benchmark comparisons from aggregates.

Inputs are anonymous aggregates only: the joint counts of 16 domain
combinations by five recorded values for each cutoff scheme, and aggregate
benchmark and characteristic outputs. No source record is read.

For a given index domain, a record has a co-occurring domain when at least one
*other* domain meets its criterion. This equals multidomain status for records
meeting the index criterion and treats records below it symmetrically.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'analysis'
OUT = AN / 'benchmark_outputs'
EN = ['Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse']
KO = ['방임', '정서학대', '신체학대', '성학대']
SHORT = ['N', 'E', 'P', 'S']
SCHEMES = ['baseline', 'common6', 'common7']
B, SEED = 9999, 20260914  # identical draws to sample_composition_uncertainty.py
PROFILE = np.arange(80) // 5
LABEL = np.arange(80) % 5
POP = np.array([bin(p).count('1') for p in PROFILE])


def masks(d):
    score = (PROFILE & (1 << d)) != 0
    typ = LABEL == d
    other = (PROFILE & ~(1 << d) & 15) != 0
    return score, typ, other


def share_ci(counts, draws, num, den):
    point = 100 * counts[num].sum() / counts[den].sum()
    boot = 100 * draws[:, num].sum(axis=1) / draws[:, den].sum(axis=1)
    return point, boot


def composition():
    joint = pd.read_csv(AN / 'composition_ci_outputs/joint_counts.csv')
    archived = pd.read_csv(AN / 'composition_ci_outputs/sample_composition_ci.csv')
    rows, components = [], []
    for scheme in SCHEMES:
        a = joint[joint.scheme == scheme].sort_values(['profile', 'recorded_type'])
        assert (a.profile.to_numpy() * 5 + a.recorded_type.to_numpy() == np.arange(80)).all()
        counts = a['count'].to_numpy()
        assert counts.sum() == 3236
        draws = np.random.default_rng(SEED).multinomial(3236, counts / 3236, size=B)
        multi = POP >= 2
        for d, name in enumerate(EN):
            score, typ, other = masks(d)
            # Reproduce the archived multidomain contrast with the same draws.
            sm, sb = share_ci(counts, draws, score & multi, score)
            tm, tb = share_ci(counts, draws, typ & multi, typ)
            ref = archived[(archived.scheme == scheme) & (archived.domain == name)].iloc[0]
            lo, hi = np.quantile(sb - tb, [.025, .975])
            assert np.isclose(sm - tm, ref.difference_pp) and np.isclose(lo, ref.ci_lower_pp) \
                and np.isclose(hi, ref.ci_upper_pp), (scheme, name)
            # Symmetric co-occurring-domain contrast.
            so, sob = share_ci(counts, draws, score & other, score)
            to, tob = share_ci(counts, draws, typ & other, typ)
            lo, hi = np.quantile(sob - tob, [.025, .975])
            rows.append(dict(scheme=scheme, domain=name, score_n=int(counts[score].sum()),
                             type_n=int(counts[typ].sum()), shared=int(counts[score & typ].sum()),
                             score_cooccurring=int(counts[score & other].sum()),
                             type_cooccurring=int(counts[typ & other].sum()),
                             score_percent=so, type_percent=to, difference_pp=so - to,
                             ci_lower_pp=lo, ci_upper_pp=hi,
                             multidomain_difference_pp=sm - tm))
            if scheme == 'baseline':
                for part, m in (('shared', score & typ), ('score_only', score & ~typ),
                                ('type_only', typ & ~score)):
                    components.append(dict(domain=name, component=part, n=int(counts[m].sum()),
                                           cooccurring=int(counts[m & other].sum()),
                                           multidomain=int(counts[m & multi].sum())))
    return pd.DataFrame(rows), pd.DataFrame(components)


def benchmarks(comp):
    joint = pd.read_csv(AN / 'composition_ci_outputs/joint_counts.csv')
    a = joint[joint.scheme == 'baseline'].sort_values(['profile', 'recorded_type'])
    counts = a['count'].to_numpy()
    multi = POP >= 2
    rank = pd.read_csv(AN / 'anchor_counselor_outputs/highest_rank_benchmarks.csv')
    uniform_ref = pd.read_csv(AN / 'contract_outputs/uniform_choice.csv')
    uniform_ref = uniform_ref[(uniform_ref.scope == 'multi') & (~uniform_ref.exclude_none)]
    rows = []
    for d, name in enumerate(EN):
        score, typ, other = masks(d)
        n_multi = counts[score & multi].sum()
        observed = counts[score & typ & multi].sum()
        uniform = (counts[score & multi] / POP[score & multi]).sum()
        assert np.isclose(uniform, uniform_ref.set_index('domain').loc[name, 'expected_matches'])
        # Records with 0-1 criterion-meeting domains keep their observed type.
        keep = typ & (POP <= 1)
        n0, t1 = counts[keep].sum(), counts[keep & other].sum()
        score_share = 100 * counts[score & other].sum() / counts[score].sum()
        expected = dict(observed=observed, uniform=uniform)
        for rule in ('numeric', 'floor', 'ceiling'):
            r = rank[(rank.rule == rule) & (rank.domain == name)].iloc[0]
            assert r.multidomain_n == n_multi
            expected[rule] = r.expected_matches
        obs_row = comp[(comp.scheme == 'baseline') & (comp.domain == name)].iloc[0]
        for allocation, e in expected.items():
            # Observed multidomain records naming d include one below d's criterion.
            in_type = counts[typ & multi].sum() if allocation == 'observed' else e
            type_share = 100 * (in_type + t1) / (n0 + in_type)
            if allocation == 'observed':
                assert np.isclose(type_share, obs_row.type_percent)
            rows.append(dict(domain=name, allocation=allocation, multidomain_n=int(n_multi),
                             expected_matches=e,
                             multidomain_nonrepresentation_percent=100 * (1 - e / n_multi),
                             type_percent=type_share, score_percent=score_share,
                             difference_pp=score_share - type_share))
    return pd.DataFrame(rows)


def load_tables():
    joint = pd.read_csv(AN / 'composition_ci_outputs/joint_counts.csv')
    a = joint[joint.scheme == 'baseline'].sort_values(['profile', 'recorded_type'])
    counts = a['count'].to_numpy()
    load, destination = [], []
    for t, name in enumerate(EN):
        score_t, typ_t, other_t = masks(t)
        n = counts[typ_t].sum()
        row = dict(type=name, n=int(n), cooccurring=int(counts[typ_t & other_t].sum()))
        for e, other_name in enumerate(EN):
            row[f'meets|{other_name}'] = int(counts[typ_t & ((PROFILE & (1 << e)) != 0)].sum())
        load.append(row)
        score_only = score_t & ~typ_t
        named_other = score_only & (LABEL < 4) & np.array(
            [bool(p & (1 << l)) if l < 4 else False for p, l in zip(PROFILE, LABEL)])
        row = dict(domain=name, score_only=int(counts[score_only].sum()),
                   another_criterion_meeting_type=int(counts[named_other].sum()))
        for l, label in enumerate(EN + ['None']):
            row[f'recorded|{label}'] = int(counts[score_only & (LABEL == l)].sum())
        destination.append(row)
    return pd.DataFrame(load), pd.DataFrame(destination)


def main():
    OUT.mkdir(exist_ok=True)
    comp, parts = composition()
    bench = benchmarks(comp)
    load, dest = load_tables()
    comp.to_csv(OUT / 'cooccurrence_share_ci.csv', index=False)
    parts.to_csv(OUT / 'cooccurrence_components.csv', index=False)
    bench.to_csv(OUT / 'benchmark_contrasts.csv', index=False)
    load.to_csv(OUT / 'type_defined_cooccurrence_load.csv', index=False)
    dest.to_csv(OUT / 'score_only_recorded_types.csv', index=False)
    (OUT / 'metadata.json').write_text(json.dumps(dict(
        replicates=B, seed=SEED,
        method='whole-record paired multinomial bootstrap of the 16x5 joint table; percentile 95% CI',
        indicator='at least one criterion-meeting domain other than the index domain',
        archived_multidomain_contrasts_reproduced=True,
        benchmark_rule='multidomain records reallocated; records with 0-1 criterion-meeting domains keep observed types',
        inputs=['composition_ci_outputs/joint_counts.csv', 'anchor_counselor_outputs/highest_rank_benchmarks.csv',
                'contract_outputs/uniform_choice.csv', 'anchor_counselor_outputs/stage_any_concern_agreement.csv',
                'anchor_counselor_outputs/definition_characteristics.csv'],
        population_claim=False), indent=2))
    pd.set_option('display.width', 200)
    print(comp.round(2).to_string(index=False))
    print(bench.round(2).to_string(index=False))
    print(load.to_string(index=False))
    print(dest.to_string(index=False))


if __name__ == '__main__':
    main()
