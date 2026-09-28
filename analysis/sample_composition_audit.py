#!/usr/bin/env python3
"""Describe sample composition using previously verified aggregate counts.

No individual records are read or reconstructed. This adds descriptive
comparisons, not a missing-data correction or a new inferential model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
KO = ['방임', '정서학대', '신체학대', '성학대']
EN = ['Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse']
SHORT = ['N', 'E', 'P', 'S']
OUT = ROOT/'analysis/composition_outputs'


def calculate():
    joint = pd.read_csv(ROOT/'analysis/sensitivity_outputs/combination_record_matrix.csv')
    scores = pd.read_csv(ROOT/'analysis/selection_outputs/score_distribution.csv')
    scores = scores[scores.scope.eq('all_records')]
    rows, restricted = [], []
    for d, e in zip(KO, EN):
        active = joint.profile.str.split('+').apply(lambda x: d in x)
        score_n = int(joint.loc[active, 'profile_n'].sum())
        named_n = int(joint[d].sum())
        score_multi = int(joint.loc[active & joint.n_domains.ge(2), 'profile_n'].sum())
        named_multi = int(joint.loc[joint.n_domains.ge(2), d].sum())
        rows.append(dict(domain=e, score_n=score_n, named_n=named_n,
                         score_multi=score_multi, named_multi=named_multi,
                         score_multi_percent=100*score_multi/score_n,
                         named_multi_percent=100*named_multi/named_n))
        none = int(joint.loc[active, '해당 없음'].sum())
        same = int(joint.loc[active, d].sum())
        restricted.append(dict(domain=e, excluded_explicit_none=none,
                               denominator=score_n-none, other_named_type=score_n-none-same,
                               percent=100*(score_n-none-same)/(score_n-none)))
    composition = pd.DataFrame(rows)
    assert composition[['score_multi', 'named_multi']].values.tolist() == [[350,102],[648,124],[487,365],[92,81]]
    thresholds = []
    for t in range(3, 8):
        for e in EN:
            sub = scores[scores.domain.eq(e) & scores.score.ge(t)]
            n, u = int(sub.total.sum()), int(sub.not_same.sum())
            thresholds.append(dict(cutoff=t, domain=e, denominator=n, unrepresented=u, percent=100*u/n))
    threshold = pd.DataFrame(thresholds)
    old = pd.read_csv(ROOT/'analysis/sensitivity_outputs/threshold_sensitivity.csv')
    for row in threshold.itertuples():
        ref = old[(old.scheme=='common_all_domains') & (old.varied_threshold==row.cutoff)].iloc[0]
        suffix = dict(zip(EN,['neglect','emotional','physical','sexual']))[row.domain]
        assert row.denominator == int(ref['n_'+suffix])
        assert abs(row.percent-ref['reduction_'+suffix]) < 1e-6
    positive = joint[joint.n_domains.gt(0)]
    unmatched_records = unmatched_occurrences = matches = 0
    for row in positive.to_dict('records'):
        matched = sum(row[d] for d in row['profile'].split('+'))
        unmatched = row['profile_n']-matched
        matches += matched
        unmatched_records += unmatched
        unmatched_occurrences += unmatched*row['n_domains']
    assert (matches, unmatched_records, unmatched_occurrences) == (1331,148,189)
    OUT.mkdir(exist_ok=True)
    composition.to_csv(OUT/'sample_composition.csv',index=False)
    pd.DataFrame(restricted).to_csv(OUT/'named_type_restriction.csv',index=False)
    threshold.to_csv(OUT/'threshold_main_table.csv',index=False)
    (OUT/'decomposition.json').write_text(json.dumps(dict(total_occurrences=2348, matches=matches,
        unrepresented=1017, minimum=869, additional=148, unmatched_records=148,
        unmatched_occurrences=189, minimum_in_unmatched_records=41),indent=2)+'\n')
    return joint, composition, threshold


if __name__=='__main__':
    joint, composition, threshold=calculate()
    print(composition.to_string(index=False))
    print('Verified decomposition: 1017 = 869 + 148; unmatched records contain 189 occurrences.')
