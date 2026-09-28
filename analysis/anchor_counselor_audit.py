#!/usr/bin/env python3
"""Local-only audit of fixed-statement benchmarks, counselor clustering, and
stage-specific agreement. Exports aggregates only.

Only the four maltreatment-domain scores, the single recorded type, crisis
stage, and the counselor-entry and type-classification fields are read, in
memory. Record identifiers, counselor codes, narratives, and per-record values
are never written. The corpus is verified against the frozen 16-by-5 table
before any statistic is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ('방임', '정서학대', '신체학대', '성학대')
ENGLISH = ('Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse')
THRESHOLDS = np.array([4, 5, 5, 5])
# Fixed-statement scores in the utilization guideline v3.5 (Methods 2.2).
ANCHORS = ((6, 8), (5, 8), (5, 8, 10), (5, 8, 10))
STAGES = ('관찰필요', '상담필요', '응급', '정상군', '학대의심')
STAGES_EN = ('Observation needed', 'Counseling needed', 'Emergency', 'Normal',
             'Suspected maltreatment')
B, SEED = 9999, 20260928


def _age(value) -> int:
    match = re.search(r'\d+', str(value or ''))
    if match is None:
        raise ValueError('Could not parse age')
    return int(match.group())


def _female(value) -> bool:
    return str(value or '').strip().startswith('여')


def _lower_grade(value) -> bool:
    text = str(value or '').strip()
    if '저' in text:
        return True
    if '고' in text:
        return False
    match = re.search(r'[1-6]', text)
    if match is None:
        raise ValueError('Could not parse grade group')
    return int(match.group()) <= 3


def read_minimal(source: Path):
    scores, labels, stages, counselors, kinds = [], [], [], [], []
    ages, female, lower, ids = [], [], [], []
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
        scores.append([found[d] for d in DOMAINS])
        info = obj['info']
        label = str(info.get('학대의심', '')).strip().replace('(', '').replace(')', '')
        labels.append(DOMAINS.index(label) if label in DOMAINS else 4)
        stage = str(info.get('위기단계', '')).strip()
        if stage not in STAGES:
            raise ValueError('Unknown crisis stage')
        stages.append(STAGES.index(stage))
        counselors.append(str(info.get('작성자(상담사)', '') or '').strip())
        kinds.append(str(info.get('유형구분', '') or '').strip())
        ages.append(_age(info.get('나이')))
        female.append(_female(info.get('성별')))
        lower.append(_lower_grade(info.get('학년')))
        ids.append(str(info.get('ID', '')).strip())
    demo = dict(age=np.asarray(ages), female=np.asarray(female), lower=np.asarray(lower),
                duplicate_ids=int(len(ids) - len(set(ids))), blank_ids=int(sum(i == '' for i in ids)))
    return (np.asarray(scores), np.asarray(labels), np.asarray(stages),
            counselors, kinds, demo, digest.hexdigest())


def verify_corpus(scores, labels, frozen: Path):
    positive = scores >= THRESHOLDS
    actual = Counter((tuple(p), int(y)) for p, y in zip(positive, labels))
    expected = Counter()
    for row in pd.read_csv(frozen, encoding='utf-8-sig').to_dict('records'):
        members = set() if row['profile'] == '기준 충족 영역 없음' else set(row['profile'].split('+'))
        profile = tuple(d in members for d in DOMAINS)
        for j, col in enumerate((*DOMAINS, '해당 없음')):
            expected[(profile, j)] += int(row[col])
    if actual != expected:
        raise ValueError('Raw corpus does not match the frozen 16-by-5 table')
    return positive


def anchor_level(score: int, d: int, rule: str) -> int:
    anchors = ANCHORS[d]
    if rule == 'floor':
        # Highest fixed statement at or below the score; criterion-meeting values
        # below the lowest statement (neglect 4-5) take the lowest level.
        return max(1, sum(a <= score for a in anchors))
    if rule == 'ceiling':
        # Lowest fixed statement at or above the score; values above the top
        # statement take the top level.
        above = [k + 1 for k, a in enumerate(anchors) if a >= score]
        return above[0] if above else len(anchors)
    raise ValueError(rule)


def benchmark(scores, positive, labels, rule: str):
    """Expected matches when each multidomain record names one highest-ranked
    criterion-meeting domain, ties split evenly."""
    multi = positive.sum(axis=1) >= 2
    exp = np.zeros(4)
    n = np.zeros(4, dtype=int)
    compatible = 0
    ties = 0
    for s, p, y in zip(scores[multi], positive[multi], labels[multi]):
        cand = np.flatnonzero(p)
        if rule == 'numeric':
            rank = np.array([s[d] for d in cand], dtype=float)
        else:
            rank = np.array([anchor_level(int(s[d]), d, rule) for d in cand], dtype=float)
        top = cand[rank == rank.max()]
        ties += len(top) > 1
        for d in cand:
            n[d] += 1
            if d in top:
                exp[d] += 1 / len(top)
        compatible += int(y in top)
    return exp, n, compatible, int(multi.sum()), ties


def joint_table(positive, labels, weights=None):
    profile = (positive * (2 ** np.arange(4))).sum(axis=1)
    cells = profile * 5 + labels
    return np.bincount(cells, weights=weights, minlength=80)


def stats_from_tables(tables):
    """Statistics from one or many 80-cell tables (profile*5 + label)."""
    tables = np.atleast_2d(tables).astype(float)
    profile = np.arange(80) // 5
    label = np.arange(80) % 5
    popcount = np.array([bin(p).count('1') for p in profile])
    positive = popcount >= 1
    match = np.array([(l < 4) and bool(p & (1 << l)) for p, l in zip(profile, label)])
    out = {}
    npos = tables[:, positive].sum(axis=1)
    out['record_correspondence'] = 100 * tables[:, match].sum(axis=1) / npos
    out['occurrence_representation'] = (100 * tables[:, match].sum(axis=1)
                                        / (tables * popcount).sum(axis=1))
    for d, name in enumerate(ENGLISH):
        score = (profile & (1 << d)) != 0
        typ = label == d
        other = (profile & ~(1 << d) & 15) != 0
        sn = tables[:, score].sum(axis=1)
        tn = tables[:, typ].sum(axis=1)
        out[f'nonrepresentation|{name}'] = 100 * (1 - tables[:, score & typ].sum(axis=1) / sn)
        out[f'score_cooccurrence|{name}'] = 100 * tables[:, score & other].sum(axis=1) / sn
        out[f'type_cooccurrence|{name}'] = 100 * tables[:, typ & other].sum(axis=1) / tn
        out[f'cooccurrence_difference|{name}'] = (out[f'score_cooccurrence|{name}']
                                                  - out[f'type_cooccurrence|{name}'])
    return out


def kappa(a, b, c, d):
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
    return float('nan') if pe == 1 else (po - pe) / (1 - pe)


def audit(source: Path, frozen: Path, out: Path):
    start = time.time()
    scores, labels, stages, counselors, kinds, demo, digest = read_minimal(source)
    positive = verify_corpus(scores, labels, frozen)
    out.mkdir(parents=True, exist_ok=True)
    multi = positive.sum(axis=1) >= 2

    # 1. Highest-rank benchmarks (numeric score and fixed-statement level).
    rows, agreement = [], []
    for rule in ('numeric', 'floor', 'ceiling'):
        exp, n, compatible, nmulti, ties = benchmark(scores, positive, labels, rule)
        for d, name in enumerate(ENGLISH):
            rows.append(dict(rule=rule, domain=name, multidomain_n=int(n[d]),
                             expected_matches=exp[d],
                             expected_nonrepresentation_percent=100 * (1 - exp[d] / n[d])))
        agreement.append(dict(rule=rule, multidomain_records=nmulti,
                              recorded_type_among_top=compatible,
                              percent=100 * compatible / nmulti,
                              records_with_tied_top=ties))
    bench = pd.DataFrame(rows)
    frozen_numeric = pd.read_csv(out.parent / 'contract_outputs/highest_score_choice.csv')
    frozen_numeric = frozen_numeric[frozen_numeric.scope == 'multi'].set_index('domain')
    check = bench[bench.rule == 'numeric'].set_index('domain')
    if not np.allclose(check.loc[list(ENGLISH), 'expected_matches'],
                       frozen_numeric.loc[list(ENGLISH), 'expected_matches']):
        raise ValueError('Numeric benchmark does not reproduce the archived output')
    bench.to_csv(out / 'highest_rank_benchmarks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(agreement).to_csv(out / 'highest_rank_agreement.csv', index=False,
                                   encoding='utf-8-sig')

    # Position of each domain among the highest fixed-statement levels in its multidomain records.
    position = []
    for rule in ('floor', 'ceiling'):
        for d, name in enumerate(ENGLISH):
            counts = Counter()
            rows_d = multi & positive[:, d]
            for srow, prow in zip(scores[rows_d], positive[rows_d]):
                cand = np.flatnonzero(prow)
                lv = {k: anchor_level(int(srow[k]), k, rule) for k in cand}
                tops = [k for k in cand if lv[k] == max(lv.values())]
                counts['unique_top' if tops == [d] else ('tied_top' if d in tops else 'below_top')] += 1
            position.append(dict(rule=rule, domain=name, multidomain_n=int(rows_d.sum()),
                                 unique_top=counts['unique_top'], tied_top=counts['tied_top'],
                                 below_top=counts['below_top']))
    pd.DataFrame(position).to_csv(out / 'top_level_position.csv', index=False, encoding='utf-8-sig')

    # Distribution of fixed-statement levels among criterion-meeting domains.
    levels = []
    for d, name in enumerate(ENGLISH):
        met = scores[positive[:, d], d]
        for rule in ('floor', 'ceiling'):
            counts = Counter(anchor_level(int(v), d, rule) for v in met)
            for level, count in sorted(counts.items()):
                levels.append(dict(rule=rule, domain=name, level=level, records=count))
    pd.DataFrame(levels).to_csv(out / 'fixed_statement_levels.csv', index=False,
                                encoding='utf-8-sig')

    # 2. Stage-specific any-concern agreement.
    stage_rows = []
    named = labels < 4
    anyc = positive.any(axis=1)
    for k, stage_en in enumerate(STAGES_EN):
        m = stages == k
        a = int((named & anyc & m).sum())
        b = int((named & ~anyc & m).sum())
        c = int((~named & anyc & m).sum())
        d_ = int((~named & ~anyc & m).sum())
        stage_rows.append(dict(stage=stage_en, records=int(m.sum()), both_concern=a,
                               type_only=b, score_only=c, both_none=d_,
                               agreement_percent=100 * (a + d_) / m.sum(),
                               kappa=kappa(a, b, c, d_)))
    pd.DataFrame(stage_rows).to_csv(out / 'stage_any_concern_agreement.csv', index=False,
                                    encoding='utf-8-sig')

    # 2b. Characteristics of the score- and type-defined samples.
    GROUPS = ('일반아동', '시설거주 아동', '저소득', '다문화가정', '학대경험 아동')
    GROUPS_EN = ('General', 'Residential care', 'Low income', 'Multicultural family',
                 'Maltreatment-experienced')
    kinds_arr = np.asarray(kinds)
    char_rows = []
    for d, name in enumerate(ENGLISH):
        score_mask = positive[:, d]
        type_mask = labels == d
        for definition, mask in (('score', score_mask), ('type', type_mask),
                                 ('shared', score_mask & type_mask),
                                 ('score_only', score_mask & ~type_mask)):
            row = dict(domain=name, definition=definition, n=int(mask.sum()),
                       age_mean=float(demo['age'][mask].mean()),
                       age_sd=float(demo['age'][mask].std(ddof=1)),
                       female_percent=100 * float(demo['female'][mask].mean()),
                       lower_grade_percent=100 * float(demo['lower'][mask].mean()))
            for k, stage_en in enumerate(STAGES_EN):
                row[f'stage|{stage_en}'] = 100 * float((stages[mask] == k).mean())
            for g, group_en in zip(GROUPS, GROUPS_EN):
                row[f'group|{group_en}'] = 100 * float((kinds_arr[mask] == g).mean())
            char_rows.append(row)
    chars = pd.DataFrame(char_rows)
    archived = pd.read_csv(out.parent / 'sensitivity_outputs/sample_definition_comparison.csv',
                           encoding='utf-8-sig').set_index('domain')
    for d, name in enumerate(ENGLISH):
        for definition, prefix in (('score', 'domain_defined_'), ('type', 'one_label_defined_')):
            mine = chars[(chars.domain == name) & (chars.definition == definition)].iloc[0]
            ref = archived.loc[DOMAINS[d]]
            pairs = [('n', 'n'), ('age_mean', 'age_mean'), ('age_sd', 'age_sd'),
                     ('female_percent', 'female_percent'),
                     ('lower_grade_percent', 'lower_grade_percent'),
                     ('stage|Observation needed', 'observation_percent'),
                     ('stage|Counseling needed', 'counseling_percent'),
                     ('stage|Emergency', 'emergency_percent'), ('stage|Normal', 'normal_percent'),
                     ('stage|Suspected maltreatment', 'suspected_percent')]
            for a, b in pairs:
                if not np.isclose(float(mine[a]), float(ref[prefix + b]), atol=1e-6):
                    raise ValueError(f'Characteristic mismatch: {name} {definition} {a}')
    chars.to_csv(out / 'definition_characteristics.csv', index=False, encoding='utf-8-sig')

    # 3. Type-classification field: closed categories only.
    kind_counts = Counter(kinds)
    kind_summary = dict(distinct_values=len(kind_counts), blank=kind_counts.get('', 0),
                        max_length=max(len(k) for k in kind_counts))
    closed = kind_summary['distinct_values'] <= 12 and kind_summary['max_length'] <= 20
    kind_summary['exported_as_categories'] = closed
    if closed:
        kt = pd.DataFrame([dict(value=k, records=v) for k, v in sorted(kind_counts.items())])
        by_stage = pd.crosstab(pd.Series(kinds, name='value'),
                               pd.Series([STAGES_EN[s] for s in stages], name='stage'))
        kt.to_csv(out / 'type_classification_field.csv', index=False, encoding='utf-8-sig')
        by_stage.to_csv(out / 'type_classification_by_stage.csv', encoding='utf-8-sig')
        groups = pd.DataFrame(dict(value=kinds, criterion_meeting=anyc, multidomain=multi,
                                   type_named=named))
        by_concern = groups.groupby('value').agg(
            records=('criterion_meeting', 'size'),
            criterion_meeting=('criterion_meeting', 'sum'),
            multidomain=('multidomain', 'sum'),
            type_named=('type_named', 'sum')).reset_index()
        by_concern.to_csv(out / 'type_classification_by_concern.csv', index=False,
                          encoding='utf-8-sig')

    # 4. Counselor-entry field: counts only, then cluster bootstrap.
    codes = pd.Series(counselors)
    blank = int((codes == '').sum())
    nonblank = pd.Series(codes[codes != ''])
    per = nonblank.value_counts()
    positive_per = pd.Series(codes[(codes != '') & anyc]).value_counts()
    counselor_summary = dict(
        records=len(codes), blank=blank, distinct=int(per.size),
        records_per_counselor=dict(min=int(per.min()), q1=float(per.quantile(.25)),
                                   median=float(per.median()), q3=float(per.quantile(.75)),
                                   max=int(per.max())),
        counselors_with_criterion_meeting_records=int(positive_per.size),
        largest_counselor_share_percent=100 * float(per.max()) / len(codes),
        # The public dataset page shows this credential label as the record author.
        matches_public_credential_label=bool(set(codes) == {'임상심리사 2급'}),
        code_length=dict(min=int(nonblank.str.len().min()),
                         max=int(nonblank.str.len().max())))

    point = stats_from_tables(joint_table(positive, labels))
    record_draws = np.random.default_rng(SEED).multinomial(
        len(labels), joint_table(positive, labels) / len(labels), size=B)
    record_stats = stats_from_tables(record_draws)
    boot_rows = []
    if blank == 0 and per.size >= 20:
        index = pd.Categorical(codes).codes
        C = index.max() + 1
        per_tables = np.zeros((C, 80))
        profile = (positive * (2 ** np.arange(4))).sum(axis=1)
        np.add.at(per_tables, (index, profile * 5 + labels), 1)
        rng = np.random.default_rng(SEED + 1)
        weights = np.stack([np.bincount(rng.integers(0, C, C), minlength=C) for _ in range(B)])
        cluster_stats = stats_from_tables(weights @ per_tables)
        for key, value in point.items():
            rlo, rhi = np.nanquantile(record_stats[key], [.025, .975])
            clo, chi = np.nanquantile(cluster_stats[key], [.025, .975])
            boot_rows.append(dict(statistic=key, estimate=float(value[0]),
                                  record_ci_low=rlo, record_ci_high=rhi,
                                  counselor_ci_low=clo, counselor_ci_high=chi,
                                  width_ratio=(chi - clo) / (rhi - rlo)))
        pd.DataFrame(boot_rows).to_csv(out / 'counselor_cluster_bootstrap.csv', index=False,
                                       encoding='utf-8-sig')

        # Between-counselor variation among counselors with enough records.
        het = []
        frame = pd.DataFrame(dict(code=codes, named=named, anyc=anyc, multi=multi,
                                  emo=positive[:, 1], emo_same=labels == 1))
        exp_top = benchmark_membership(scores, positive, labels)
        frame['top'] = exp_top
        for metric, mask, value, minimum in [
            ('explicit_none_among_criterion_meeting', frame.anyc, ~frame.named, 20),
            ('recorded_type_among_highest_score_multidomain', frame.multi, frame.top, 10),
            ('emotional_nonrepresentation', frame.emo, ~frame.emo_same, 10)]:
            g = pd.DataFrame(dict(code=frame.code[mask], value=value[mask].astype(float)))
            agg = g.groupby('code')['value'].agg(['size', 'mean'])
            agg = agg[agg['size'] >= minimum]
            if len(agg) >= 3:
                q = pd.Series(100 * agg['mean']).quantile([0, .25, .5, .75, 1]).to_list()
                het.append(dict(metric=metric, minimum_records=minimum,
                                counselors=int(len(agg)), records=int(agg['size'].sum()),
                                pooled_percent=100 * float(g['value'].mean()),
                                min=q[0], q1=q[1], median=q[2], q3=q[3], max=q[4]))
        pd.DataFrame(het).to_csv(out / 'counselor_heterogeneity.csv', index=False,
                                 encoding='utf-8-sig')

    meta = dict(corpus_sha256_of_file_hashes=digest, frozen_table_match=True,
                records=int(len(labels)), duplicate_record_ids=demo['duplicate_ids'],
                blank_record_ids=demo['blank_ids'], bootstrap_replicates=B, seed=SEED,
                cluster_seed=SEED + 1, anchors={ENGLISH[d]: list(a) for d, a in enumerate(ANCHORS)},
                anchor_rules=dict(floor='highest fixed statement at or below the score; '
                                        'criterion-meeting values below the lowest statement take level 1',
                                  ceiling='lowest fixed statement at or above the score; '
                                          'values above the top statement take the top level'),
                counselor_field=counselor_summary, type_classification_field=kind_summary,
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                runtime_seconds=round(time.time() - start, 1),
                exports='aggregates only; no identifiers, counselor codes, or per-record values')
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def benchmark_membership(scores, positive, labels):
    """Per-record flag (in memory only): recorded type among numeric top scores."""
    flags = np.zeros(len(labels), dtype=bool)
    for i, (s, p, y) in enumerate(zip(scores, positive, labels)):
        cand = np.flatnonzero(p)
        if len(cand) >= 2:
            top = cand[np.array([s[d] for d in cand]) == max(s[d] for d in cand)]
            flags[i] = y in top
    return flags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--joint-table', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    meta = audit(args.source_dir, args.joint_table, args.out_dir)
    print(json.dumps({k: meta[k] for k in ('records', 'frozen_table_match', 'counselor_field',
                                            'type_classification_field', 'runtime_seconds')},
                     ensure_ascii=False, indent=2))
