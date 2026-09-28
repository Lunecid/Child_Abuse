#!/usr/bin/env python3
"""Local-only raw-score audit; exports aggregates, never record-level rows.

The source directory contains the already authorized, deidentified corpus.
Only four scores and the recorded category enter the analysis. Narratives,
source identifiers, and per-record predictions are never exported.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint

DOMAINS = ('방임', '정서학대', '신체학대', '성학대')
ENGLISH = ('Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse')
THRESHOLDS = np.array([4, 5, 5, 5])
PRIORITY = np.array([2, 1, 3, 4])  # S > P > N > E, a candidate rule specified for this exploratory audit


def read_minimal(source):
    scores, labels = [], []
    digest = hashlib.sha256()
    for path in sorted(source.glob('*.json')):
        raw = path.read_bytes()
        digest.update(hashlib.sha256(raw).digest())
        obj = json.loads(raw.decode('utf-8-sig'))
        found = {}
        for area in obj.get('list', []):
            if area.get('문항') == '학대여부':
                for item in area.get('list', []):
                    if item.get('항목') in DOMAINS:
                        found[item['항목']] = int(item['점수'])
        if set(found) != set(DOMAINS):
            raise ValueError('Missing domain score')
        score = [found[d] for d in DOMAINS]
        if not all(0 <= v <= 10 for v in score):
            raise ValueError('Score outside documented scale')
        label = str(obj['info'].get('학대의심', '')).strip().replace('(', '').replace(')', '')
        if label in DOMAINS:
            labels.append(DOMAINS.index(label))
        elif label in ('해당 없음', '해당없음', '없음', ''):
            labels.append(4)
        else:
            raise ValueError('Unknown category')
        scores.append(score)
    return np.asarray(scores), np.asarray(labels), digest.hexdigest()


def verify_corpus(scores, labels, frozen):
    positive = scores >= THRESHOLDS
    actual = Counter((tuple(p), int(y)) for p, y in zip(positive, labels))
    expected = Counter()
    table = pd.read_csv(frozen)
    for row in table.to_dict('records'):
        profile = tuple(d in row['profile'].split('+') for d in DOMAINS)
        for j, col in enumerate((*DOMAINS, '해당 없음')):
            expected[(profile, j)] += int(row[col])
    if actual != expected:
        raise ValueError('Raw corpus does not match the frozen 16-by-5 table')
    return positive


def design(scores, kind):
    # Centering does not change within-record utility differences.
    centered = scores - scores.max(axis=1, keepdims=True)
    score = centered[:, :, None].astype(float)
    effects = np.broadcast_to(np.eye(4)[:, [0, 2, 3]], (len(scores), 4, 3))
    if kind == 'score':
        return score
    if kind == 'domain':
        return effects
    if kind == 'combined':
        return np.concatenate([score, effects], axis=2)
    raise ValueError(kind)


def likelihood(beta, x, available, y):
    utilities = np.einsum('ndp,p->nd', x, beta)
    utilities = np.where(available, utilities, -np.inf)
    normalizer = logsumexp(utilities, axis=1)
    probabilities = np.exp(utilities - normalizer[:, None])
    nll = np.sum(normalizer - utilities[np.arange(len(y)), y])
    expected = np.einsum('nd,ndp->np', probabilities, x)
    gradient = np.sum(expected - x[np.arange(len(y)), y], axis=0)
    hessian = (np.einsum('nd,ndp,ndq->pq', probabilities, x, x)
               - np.einsum('np,nq->pq', expected, expected))
    return nll, gradient, hessian, probabilities


def fit(x, available, y, initial=None):
    if not np.all(available[np.arange(len(y)), y]):
        raise ValueError('Observed choice outside candidate set')
    start = np.zeros(x.shape[2]) if initial is None else initial
    result = minimize(lambda b: likelihood(b, x, available, y)[0], start,
                      jac=lambda b: likelihood(b, x, available, y)[1],
                      hess=lambda b: likelihood(b, x, available, y)[2],
                      method='trust-exact', options={'gtol': 1e-7, 'maxiter': 200})
    ll, grad, hess, prob = likelihood(result.x, x, available, y)
    good = (np.isfinite(result.x).all() and np.max(np.abs(grad)) < 1e-5
            and np.linalg.eigvalsh(hess).min() > 1e-7)
    if not good:
        raise RuntimeError('Choice model failed finite-estimate/convergence checks')
    return result.x, ll, np.linalg.inv(hess), prob


def rate(successes, n):
    low, high = proportion_confint(successes, n, method='wilson')
    return dict(matches=int(successes), n=int(n), percent=100*successes/n,
                ci_low=100*low, ci_high=100*high)


def audit(source, frozen, output, reps=999, seed=20260907):
    scores, labels, digest = read_minimal(source)
    positive = verify_corpus(scores, labels, frozen)
    k = positive.sum(axis=1)
    multi = k >= 2
    s, y, avail = scores[multi], labels[multi], positive[multi]
    n = len(y)
    named = y < 4
    eligible = named & avail[np.arange(n), np.minimum(y, 3)]
    maxima = s == s.max(axis=1, keepdims=True)
    tied = maxima.sum(axis=1) > 1
    compatible = named & maxima[np.arange(n), np.minimum(y, 3)]
    hierarchy = np.argmax(np.where(avail, PRIORITY, -1), axis=1)
    maximum_hierarchy = np.argmax(np.where(maxima, PRIORITY, -1), axis=1)
    uniform_tie_credit = np.where(compatible, 1/maxima.sum(axis=1), 0)
    rules = dict(
        argmax_compatible_all=rate(compatible.sum(), n),
        argmax_hierarchy_tiebreak_all=rate((maximum_hierarchy == y).sum(), n),
        hierarchy_all=rate((hierarchy == y).sum(), n),
        argmax_unique=rate((compatible & ~tied).sum(), (~tied).sum()),
        argmax_tied_compatible=rate((compatible & tied).sum(), tied.sum()),
        argmax_compatible_eligible=rate(compatible[eligible].sum(), eligible.sum()),
        hierarchy_eligible=rate((hierarchy[eligible] == y[eligible]).sum(), eligible.sum()),
        uniform_random_tie_expected_percent=100*uniform_tie_credit.mean(),
        random_among_positive_expected_percent=100*np.mean(np.where(eligible, 1/avail.sum(axis=1), 0)),
        none=int((y == 4).sum()), outside_positive=int((named & ~eligible).sum()),
        highest_score_ties=int(tied.sum()))
    output.mkdir(parents=True, exist_ok=True)
    distributions = []
    for scope, mask in [('all_records', np.ones(len(scores), bool)), ('multidomain', multi)]:
        for j, name in enumerate(ENGLISH):
            for value in range(11):
                at_score = mask & (scores[:, j] == value)
                count = int(at_score.sum())
                matching = int((at_score & (labels == j)).sum())
                distributions.append(dict(scope=scope, domain=name, score=value,
                                          total=count, same_type=matching, not_same=count-matching))
    pd.DataFrame(distributions).to_csv(output/'score_distribution.csv', index=False)
    profiles = []
    for p in sorted(set(map(tuple, avail))):
        m = np.all(avail == p, axis=1)
        profiles.append(dict(profile='+'.join(ENGLISH[j] for j in range(4) if p[j]), n=int(m.sum()),
                             argmax_compatible=int(compatible[m].sum()), hierarchy_matches=int((hierarchy[m]==y[m]).sum()),
                             none=int((y[m]==4).sum())))
    pd.DataFrame(profiles).to_csv(output/'rule_by_profile.csv', index=False)
    fit_scores, fit_y, fit_avail = s[eligible], y[eligible], avail[eligible]
    models = {}
    for kind in ['score', 'domain', 'combined']:
        x = design(fit_scores, kind)
        beta, nll, cov, prob = fit(x, fit_avail, fit_y)
        cv = []
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, fit_y):
            b, *_ = fit(x[train], fit_avail[train], fit_y[train])
            test_nll, _, _, pr = likelihood(b, x[test], fit_avail[test], fit_y[test])
            # Expected accuracy under uniform random resolution of exact ties.
            best = np.isclose(pr, pr.max(axis=1, keepdims=True), atol=1e-12, rtol=0)
            credit = best[np.arange(len(test)), fit_y[test]]/best.sum(axis=1)
            cv.append((len(test), float(test_nll), float(credit.sum())))
        models[kind] = dict(n=len(fit_y), parameters=beta.tolist(),
                            se=np.sqrt(np.diag(cov)).tolist(), log_likelihood=-float(nll),
                            aic=2*nll+2*len(beta),
                            cv_log_loss=sum(v[1] for v in cv)/len(fit_y),
                            cv_accuracy=sum(v[2] for v in cv)/len(fit_y))
    x = design(fit_scores, 'combined')
    beta = np.array(models['combined']['parameters'])
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(reps):
        sample = rng.integers(0, len(fit_y), len(fit_y))
        try:
            b, *_ = fit(x[sample], fit_avail[sample], fit_y[sample], initial=beta)
            bootstrap.append(b)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    if len(bootstrap) < .95*reps:
        raise RuntimeError('Too many failed bootstrap fits')
    ci = np.quantile(np.asarray(bootstrap), [.025, .975], axis=0)
    terms = ('Score (per point)', 'Neglect vs emotional', 'Physical vs emotional', 'Sexual vs emotional')
    coef = [dict(term=t, beta=float(beta[j]), odds_ratio=float(np.exp(beta[j])),
                 ci_low=float(np.exp(ci[0,j])), ci_high=float(np.exp(ci[1,j]))) for j,t in enumerate(terms)]
    pd.DataFrame(coef).to_csv(output/'choice_coefficients.csv', index=False)
    models['combined'].update(bootstrap_requested=reps, bootstrap_successful=len(bootstrap),
                              bootstrap_failed=reps-len(bootstrap), coefficients=coef)
    lr = 2*(models['combined']['log_likelihood'] - models['score']['log_likelihood'])
    models['domain_addition_lr'] = dict(statistic=lr, df=3, p=float(chi2.sf(lr, 3)))
    # Include all four domains in a sensitivity fit, including below-cutoff alternatives.
    all_named_x = design(s[named], 'combined')
    b, ll, cov, _ = fit(all_named_x, np.ones((named.sum(),4),bool), y[named])
    models['all_four_alternatives'] = dict(n=int(named.sum()), parameters=b.tolist(),
                                          se=np.sqrt(np.diag(cov)).tolist(), log_likelihood=-float(ll))
    summary = dict(source_content_sha256=digest, joint_table_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                   joint_table_match=True, all_records=len(scores), multidomain_records=n,
                   eligible_choice_records=int(eligible.sum()), rules=rules, models=models, seed=seed,
                   cv='5-fold stratified by recorded domain; internal random folds, not official partitions',
                   limitations=['Record exchangeability; repeated children cannot be linked.',
                                'Conditional on a positive domain being recorded; none/outside cases reported separately.',
                                'IIA and a common numerical score slope; no clinical cross-domain calibration.',
                                'Contemporaneous associations do not identify recording chronology or human decision rules.'])
    (output/'selection_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(n=n, eligible=int(eligible.sum()), rules=rules, coefficients=coef,
                          models={m:{k:v for k,v in models[m].items() if k in ['cv_log_loss','cv_accuracy']} for m in ['score','domain','combined']}),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--joint-table',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=999)
    args=p.parse_args()
    audit(args.source_dir,args.joint_table,args.out_dir,args.bootstrap)
