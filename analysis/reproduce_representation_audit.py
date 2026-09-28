#!/usr/bin/env python3
"""Reproduce RQ1/RQ2 from the profile-by-recorded-type aggregate table.

No source records or reconstructed participant rows are used. Multinomial
resampling of the joint cells is distributionally identical to drawing whole
records with replacement for statistics that depend only on these cells.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

DOMAINS = ('방임', '정서학대', '신체학대', '성학대')
ENGLISH = ('Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse')


def read_joint_table(path):
    table = pd.read_csv(path, encoding='utf-8-sig')
    cells, domains, matched = [], [], []
    for _, row in table.iterrows():
        active = np.array([d in row['profile'].split('+') for d in DOMAINS], dtype=int)
        if active.sum() != row['n_domains']:
            raise ValueError('Domain count does not agree with profile')
        if sum(row[d] for d in (*DOMAINS, '해당 없음')) != row['profile_n']:
            raise ValueError('Joint cells do not sum to the profile total')
        for j, summary in enumerate((*DOMAINS, '해당 없음')):
            cells.append(int(row[summary]))
            domains.append(active)
            matched.append(active * (np.arange(4) == j))
    counts = np.array(cells, dtype=int)
    if np.any(counts < 0):
        raise ValueError('Negative counts')
    return counts, np.array(domains), np.array(matched)


def statistics(weights, domains, matched):
    weights = np.atleast_2d(weights)
    k = domains.sum(axis=1)
    positive = weights @ (k > 0)
    occurrences = weights @ k
    matches = weights @ matched.sum(axis=1)
    denominator = weights @ domains
    nonrepresentation = 1 - (weights @ matched) / denominator
    represented_fraction = np.divide(matched.sum(axis=1), k,
                                     out=np.zeros(len(k)), where=k > 0)
    case_fraction = np.sum(weights * represented_fraction, axis=1)
    metrics = np.column_stack((
        (weights @ (k > 1)) / positive,
        matches / positive,
        matches / occurrences,
        positive / occurrences,
        case_fraction / positive,
    ))
    return nonrepresentation, metrics


def run(input_path, out_dir, repetitions=9999, seed=20260907, tex_dir=None):
    counts, domains, matched = read_joint_table(input_path)
    if (counts.sum(), counts @ (domains.sum(axis=1) > 0),
        counts @ domains.sum(axis=1), counts @ matched.sum(axis=1)) != (3236, 1479, 2348, 1331):
        raise ValueError('Input does not reproduce frozen corpus totals')
    observed, metrics = statistics(counts, domains, matched)
    observed, metrics = observed[0], metrics[0]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(counts.sum(), counts / counts.sum(), size=repetitions)
    bootstrap, boot_metrics = statistics(draws, domains, matched)
    if not np.isfinite(bootstrap).all() or not np.isfinite(boot_metrics).all():
        raise ValueError('Undefined bootstrap statistics; do not silently drop replicates')
    covariance = np.cov(bootstrap, rowvar=False, ddof=1)
    contrast = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]])
    delta = contrast @ observed
    wald = float(delta @ np.linalg.solve(contrast @ covariance @ contrast.T, delta))
    global_p = float(chi2.sf(wald, 3))
    rows = []
    for i, j in itertools.combinations(range(4), 2):
        difference = observed[i] - observed[j]
        boot_difference = bootstrap[:, i] - bootstrap[:, j]
        p = (1 + np.count_nonzero(np.abs(boot_difference - difference) >= abs(difference))) / (repetitions + 1)
        lo, hi = np.quantile(100 * boot_difference, [.025, .975])
        rows.append(dict(first=ENGLISH[i], second=ENGLISH[j], difference_pp=100*difference,
                         ci_low=lo, ci_high=hi, p_bootstrap=p))
    adjusted = multipletests([row['p_bootstrap'] for row in rows], method='holm')[1]
    for row, p in zip(rows, adjusted):
        row['p_holm'] = float(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir/'audit_pairwise_comparisons.csv', index=False, float_format='%.9f')
    domain_rows = []
    for i, domain in enumerate(ENGLISH):
        lo, hi = np.quantile(100*bootstrap[:, i], [.025, .975])
        domain_rows.append(dict(domain=domain, positive=int(counts@domains[:, i]),
                                same_type=int(counts@matched[:, i]),
                                not_same_percent=100*observed[i], ci_low=lo, ci_high=hi))
    pd.DataFrame(domain_rows).to_csv(out_dir/'audit_domain_estimates.csv', index=False, float_format='%.9f')
    names = ('multidomain', 'record_match', 'occurrence_match', 'capacity_maximum', 'case_average')
    metric_rows=[]
    for j, name in enumerate(names):
        lo, hi=np.quantile(100*boot_metrics[:, j], [.025,.975])
        metric_rows.append(dict(metric=name, percent=100*metrics[j], ci_low=lo, ci_high=hi))
    pd.DataFrame(metric_rows).to_csv(out_dir/'audit_record_estimates.csv', index=False, float_format='%.9f')
    report = dict(input_file=input_path.name, input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  repetitions=repetitions, seed=seed, successful_replicates=repetitions, failed_replicates=0,
                  resampling='fixed-N multinomial over profile-by-recorded-type cells; whole-record bootstrap law',
                  intervals='2.5th and 97.5th percentiles',
                  pairwise_p='(1 + count(abs(delta_boot - delta_observed) >= abs(delta_observed))) / (B + 1)',
                  adjustment='Holm across six two-sided tests',
                  global_test=dict(statistic=wald, df=3, p=global_p, covariance=covariance.tolist()),
                  interpretation='conditional on this corpus and record-level exchangeability; not population inference')
    (out_dir/'audit_inference.json').write_text(json.dumps(report,indent=2)+'\n')
    if tex_dir:
        tex_dir.mkdir(parents=True,exist_ok=True)
        macros=[r'% Generated by reproduce_representation_audit.py; do not edit values manually.',
                r'\newcommand{\AuditWald}{'+f'{wald:.1f}'+'}']
        for name, row in zip(('Multi','Record','Occurrence','Capacity','CaseAverage'), metric_rows):
            macros += [r'\newcommand{\Audit'+name+r'CI}{'+f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"+'}']
        for name,row in zip(('Neglect','Emotional','Physical','Sexual'),domain_rows):
            macros += [r'\newcommand{\Audit'+name+r'CI}{'+f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"+'}']
        (tex_dir/'audit_statistics.tex').write_text('\n'.join(macros)+'\n')
    print(json.dumps(dict(wald=wald,global_p=global_p,pairwise=rows,record_metrics=metric_rows),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--tex-dir', type=Path)
    args=parser.parse_args()
    run(args.input, args.out_dir, tex_dir=args.tex_dir)
