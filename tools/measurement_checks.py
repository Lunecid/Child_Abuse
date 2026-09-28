#!/usr/bin/env python3
"""Measurement and allocation sensitivity checks; exports aggregates only.

Uses the authorized local corpus in memory, verifies all three archived joint
tables and the corpus digest, and never exports identifiers or narratives.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
AN = R/'analysis'
sys.path.insert(0, str(AN))
from selection_audit import read_minimal
from anchor_counselor_audit import anchor_level

KO = ('방임', '정서학대', '신체학대', '성학대')
EN = ('Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse')
BASE = np.array([4, 5, 5, 5])
B, SEED = 9999, 20260914


def composition(s, y, d, cut):
    positive = s >= cut
    score, typ = positive[:, d], y == d
    other = np.delete(positive, d, axis=1).any(axis=1)
    # Preserve the archived 80-cell draw sequence for baseline/common cutoffs.
    profile = (positive * (2**np.arange(4))).sum(axis=1)
    counts = np.bincount(profile*5+y, minlength=80)
    draws = np.random.default_rng(SEED).multinomial(len(s), counts / len(s), size=B)
    p, label = np.arange(80)//5, np.arange(80)%5
    a, b, o = p & (1<<d) != 0, label == d, p & (15 ^ (1<<d)) != 0
    sn, tn, shared = int(score.sum()), int(typ.sum()), int((score & typ).sum())
    sm, tm = int((score & other).sum()), int((typ & other).sum())
    delta = 100 * (sm / sn - tm / tn)
    boot = 100 * (draws[:, a & o].sum(axis=1) / draws[:, a].sum(axis=1)
                  - draws[:, b & o].sum(axis=1) / draws[:, b].sum(axis=1))
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(domain=EN[d], score_n=sn, type_n=tn, shared=shared,
                score_only=sn-shared, type_only=tn-shared,
                jaccard=shared/(sn+tn-shared), score_cooccurring=sm,
                type_cooccurring=tm, score_percent=100*sm/sn,
                type_percent=100*tm/tn, difference_pp=delta,
                ci_low_pp=float(lo), ci_high_pp=float(hi))


def shared_restriction(archived):
    a = archived[archived.scheme == 'baseline'].sort_values(['profile','recorded_type'])
    counts = a['count'].to_numpy()
    draws = np.random.default_rng(SEED).multinomial(3236, counts/3236, size=B)
    p, y = np.arange(80)//5, np.arange(80)%5
    rows = []
    for d in range(4):
        score = p & (1 << d) != 0
        shared = score & (y == d)
        other = p & (15 ^ (1 << d)) != 0
        sn, tn = counts[score].sum(), counts[shared].sum()
        sm, tm = counts[score & other].sum(), counts[shared & other].sum()
        boot = 100*(draws[:,score & other].sum(axis=1)/draws[:,score].sum(axis=1)
                    - draws[:,shared & other].sum(axis=1)/draws[:,shared].sum(axis=1))
        lo, hi = np.quantile(boot,[.025,.975])
        rows.append(dict(domain=EN[d], score_n=int(sn), shared_n=int(tn),
                         score_percent=100*sm/sn, shared_percent=100*tm/tn,
                         difference_pp=100*(sm/sn-tm/tn),
                         ci_low_pp=float(lo),ci_high_pp=float(hi)))
    return pd.DataFrame(rows)


def allocations(s, y):
    pos = s >= BASE
    multi = pos.sum(axis=1) >= 2
    consistent = (y < 4) & pos[np.arange(len(y)), np.minimum(y,3)]
    rows, tie_rows = [], []
    for rule in ['uniform','floor','ceiling','numeric']:
        credit = np.zeros((len(s),4))
        top_count = np.zeros(len(s),int)
        compatible = np.zeros(len(s),bool)
        for i in np.flatnonzero(multi):
            cand = np.flatnonzero(pos[i])
            rank = (np.ones(len(cand)) if rule == 'uniform' else s[i,cand]
                    if rule == 'numeric' else
                    np.array([anchor_level(int(s[i,d]),int(d),rule) for d in cand]))
            top = cand[rank == rank.max()]
            credit[i,top] = 1/len(top)
            top_count[i] = len(top)
            compatible[i] = y[i] in top
        if rule != 'uniform':
            for status, mask in [('unique',multi & (top_count==1)),
                                 ('tied',multi & (top_count>1))]:
                tie_rows.append(dict(rule=rule,status=status,n=int(mask.sum()),
                                     matches=int(compatible[mask].sum()),
                                     percent=100*compatible[mask].mean()))
        for scope, change in [('all_multi',multi),('consistent_multi',multi & consistent)]:
            weights = np.column_stack([y == d for d in range(4)]).astype(float)
            weights[change] = credit[change]
            for d in range(4):
                other = np.delete(pos,d,axis=1).any(axis=1)
                n = weights[:,d].sum()
                m = weights[other,d].sum()
                score_rate = 100*other[pos[:,d]].mean()
                type_rate = 100*m/n
                rows.append(dict(rule=rule,scope=scope,domain=EN[d],
                                 reassigned_records=int(change.sum()),expected_type_n=n,
                                 expected_type_cooccurring=m,type_percent=type_rate,
                                 score_percent=score_rate,difference_pp=score_rate-type_rate))
    assert int((multi & ~consistent).sum()) == 37
    return pd.DataFrame(rows),pd.DataFrame(tie_rows)


def tex_tables(cutoffs, restricted, alloc, ties):
    directory = R/'generated/tables/supplement'
    directory.mkdir(parents=True,exist_ok=True)
    def write(name,caption,label,header,cols,lines,long=False):
        if long:
            content = [r'\par\begingroup\singlespacing\scriptsize',
                r'\begin{longtable}{'+cols+'}',
                r'\caption{'+caption+r'}\label{'+label+r'}\\',
                r'\toprule',header+r'\\\midrule\endfirsthead',
                r'\toprule',header+r'\\\midrule\endhead',
                r'\bottomrule\endfoot',*lines,r'\end{longtable}\endgroup']
        else:
            content = [r'\begin{table}[H]\singlespacing\centering\scriptsize',
                r'\caption{'+caption+r'}\label{'+label+'}',
                r'\begin{tabular}{'+cols+r'}\toprule',header+r'\\\midrule',
                *lines,r'\bottomrule\end{tabular}\end{table}']
        (directory/name).write_text('\n'.join(content)+'\n')
    lines=[]
    for d,name in enumerate(EN):
        for scheme,lab in [('baseline','基本'),('common6','6'),('common7','7')]:
            a=cutoffs[(cutoffs.domain==name)&(cutoffs.scheme==scheme)].iloc[0]
            lab='기본' if lab=='基本' else lab
            lines.append(f'{KO[d]} & {lab} & {a.score_n} & {a.type_n} & {a.shared} & {a.score_only} & {a.type_only} & {a.jaccard:.3f}'+r'\\')
    write('threshold_membership.tex','기준점에 따른 두 표본의 포함·제외와 중복도',
          'tab:threshold_membership',r'유형 & 기준점 & 점수 $n$ & 기록유형 $n$ & 공통 $n$ & 점수만 $n$ & 기록유형만 $n$ & Jaccard',
          'llrrrrrr',lines)
    labels={'baseline':'기본','focal6':'해당 영역만 6','other6':'다른 영역만 6','common6':'전 영역 6',
            'focal7':'해당 영역만 7','other7':'다른 영역만 7','common7':'전 영역 7'}
    lines=[]
    for d,name in enumerate(EN):
        for scheme,lab in labels.items():
            a=cutoffs[(cutoffs.domain==name)&(cutoffs.scheme==scheme)].iloc[0]
            lines.append(f'{KO[d]} & {lab} & {a.score_n} & {a.score_percent:.1f} & {a.type_percent:.1f} & {a.difference_pp:.1f} & [{a.ci_low_pp:.1f}, {a.ci_high_pp:.1f}]'+r'\\')
    write('separate_cutoff_changes.tex','표본 선정 기준과 동반 의심 기준을 구분하여 변경한 비교',
          'tab:separate_cutoffs',r'유형 & 변경 조건 & 점수 $n$ & 점수, \% & 기록유형, \% & 차이 & 95\% CI',
          'llrrrrr',lines,long=True)
    labels={'floor':'하위 문구 대응','ceiling':'상위 문구 대응','numeric':'원점수'}
    lines=[]
    for rule,lab in labels.items():
        for status,name in [('unique','단독 최고'),('tied','공동 최고')]:
            a=ties[(ties.rule==rule)&(ties.status==status)].iloc[0]
            lines.append(f'{lab} & {name} & {a.n} & {a.matches} & {a.percent:.1f}'+r'\\')
    write('severity_tie_agreement.tex','최고 심각도 유형의 동률 여부에 따른 기록유형 일치',
          'tab:severity_ties',r'비교 기준 & 최고 유형 & 기록 수 & 일치 수 & 일치율, \%',
          'llrrr',lines)
    old=pd.read_csv(AN/'benchmark_outputs/benchmark_contrasts.csv')
    lines=[]
    for d,name in enumerate(EN):
        vals=[old[(old.domain==name)&(old.allocation=='observed')].iloc[0].difference_pp]
        for rule,scope in [('uniform','all_multi'),('uniform','consistent_multi'),
                           ('floor','all_multi'),('floor','consistent_multi')]:
            vals.append(alloc[(alloc.domain==name)&(alloc.rule==rule)&(alloc.scope==scope)].iloc[0].difference_pp)
        lines.append(KO[d]+' & '+' & '.join(f'{x:.1f}' for x in vals)+r'\\')
    write('consistent_reallocation.tex','가상 유형 재배정 대상에 따른 동반 의심률 차이',
          'tab:consistent_reallocation',r'유형 & 관찰 & \makecell{균등 선택\\전체 708건} & \makecell{균등 선택\\일치 671건} & \makecell{심각도 우선\\전체 708건} & \makecell{심각도 우선\\일치 671건}',
          'lrrrrr',lines)


def main(source):
    s,y,digest=read_minimal(source)
    metadata=json.loads((AN/'anchor_counselor_outputs/metadata.json').read_text())
    assert s.shape==(3236,4) and digest==metadata['corpus_sha256_of_file_hashes']
    archived=pd.read_csv(AN/'composition_ci_outputs/joint_counts.csv')
    for scheme,cut in [('baseline',BASE),('common6',np.full(4,6)),('common7',np.full(4,7))]:
        profile=((s>=cut)*(2**np.arange(4))).sum(axis=1)
        counts=np.bincount(profile*5+y,minlength=80)
        a=archived[archived.scheme==scheme].sort_values(['profile','recorded_type'])
        assert np.array_equal(counts,a['count'].to_numpy())
    rows=[]
    for d in range(4):
        schemes={'baseline':BASE.copy()}
        for value in [6,7]:
            focal=BASE.copy();focal[d]=value
            other=np.full(4,value);other[d]=BASE[d]
            schemes.update({f'focal{value}':focal,f'other{value}':other,f'common{value}':np.full(4,value)})
        for scheme,cut in schemes.items():
            rows.append(dict(scheme=scheme,**composition(s,y,d,cut)))
    cutoffs=pd.DataFrame(rows)
    # Reproduce published point estimates from independently archived aggregates.
    ref=pd.read_csv(AN/'benchmark_outputs/cooccurrence_share_ci.csv')
    for a in ref.itertuples():
        b=cutoffs[(cutoffs.scheme==a.scheme)&(cutoffs.domain==a.domain)].iloc[0]
        for col in ['score_n','type_n','shared','score_percent','type_percent','difference_pp']:
            assert np.isclose(b[col],getattr(a,col)),(a.scheme,a.domain,col)
        assert np.isclose(b.ci_low_pp,a.ci_lower_pp) and np.isclose(b.ci_high_pp,a.ci_upper_pp)
    alloc,ties=allocations(s,y)
    ref=pd.read_csv(AN/'benchmark_outputs/benchmark_contrasts.csv')
    for a in alloc[alloc.scope=='all_multi'].itertuples():
        b=ref[(ref.domain==a.domain)&(ref.allocation==a.rule)].iloc[0]
        assert np.isclose(a.difference_pp,b.difference_pp)
    restricted=shared_restriction(archived)
    out=R/'generated/measurement_checks';out.mkdir(parents=True,exist_ok=True)
    for name,data in [('separate_cutoffs',cutoffs),('shared_restriction',restricted),
                      ('allocation_scopes',alloc),('severity_ties',ties)]:
        data.to_csv(out/f'{name}.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(
        source_content_sha256=digest,archived_three_joint_tables_match=True,
        existing_point_estimates_reproduced=True,replicates=B,seed=SEED,
        bootstrap='paired whole-record multinomial; percentile 95% CI',
        cutoff_intervals='80 joint cells for each cutoff vector; archived draws reproduced for baseline/common cutoffs',
        restriction_intervals='original 80 baseline cells and archived seed',
        expected_rates='ratios of probability-weighted expected frequencies; not mean simulated ratios',
        exports='anonymous aggregates only'),indent=2)+'\n')
    tex_tables(cutoffs,restricted,alloc,ties)
    print(cutoffs[cutoffs.domain=='Emotional abuse'].round(3).to_string(index=False))
    print(restricted.round(3).to_string(index=False))
    print(ties.round(3).to_string(index=False))
    print(alloc[(alloc.scope=='consistent_multi')&alloc.rule.isin(['floor','uniform'])].round(3).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    main(p.parse_args().source_dir)
