#!/usr/bin/env python3
"""Compare set membership and single-choice tie handling; aggregate outputs only."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
AN = R/'analysis'
sys.path.insert(0,str(AN))
from selection_audit import read_minimal
from anchor_counselor_audit import anchor_level, ANCHORS

NAMES={'floor':'하위 문구 대응','ceiling':'상위 문구 대응','numeric':'원점수'}
DOMAINS=('방임','정서학대','신체학대','성학대')
PRIORITY=np.array([2,1,3,4])  # sexual > physical > neglect > emotional
THRESHOLDS=np.array([4,5,5,5])


def mapping_table():
    rows=[]
    mapping=[]
    for rule in ['floor','ceiling']:
        rows.append(r'\multicolumn{5}{l}{\textit{'+NAMES[rule]+r'}}\\')
        for d,name in enumerate(DOMAINS):
            levels=[]
            for level in [1,2,3]:
                scores=[v for v in range(int(THRESHOLDS[d]),11) if anchor_level(v,d,rule)==level]
                label=(str(scores[0]) if len(scores)==1 else f'{scores[0]}--{scores[-1]}') if scores else '--'
                levels.append(label)
                mapping.append(dict(rule=rule,domain=name,level=level,scores=','.join(map(str,scores))))
            rows.append(name+' & '+'·'.join(map(str,ANCHORS[d]))+' & '+' & '.join(levels)+r'\\')
        rows.append(r'\addlinespace')
    table=[r'\begin{table}[H]\singlespacing\centering\small',
           r'\caption{가상 심각도 비교에 사용한 영역별 점수와 문구 단계의 대응}\label{tab:severity_mapping}',
           r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule',
           r'유형 & 문구가 있는 점수 & 1단계 점수 & 2단계 점수 & 3단계 점수\\\midrule',
           *rows,r'\bottomrule\end{tabular*}\end{table}']
    return '\n'.join(table)+'\n',pd.DataFrame(mapping)


def run(source):
    s,y,digest=read_minimal(source)
    meta=json.loads((AN/'anchor_counselor_outputs/metadata.json').read_text())
    assert s.shape==(3236,4) and digest==meta['corpus_sha256_of_file_hashes']
    pos=s>=THRESHOLDS
    profile=(pos*(2**np.arange(4))).sum(axis=1)
    archived=pd.read_csv(AN/'composition_ci_outputs/joint_counts.csv')
    a=archived[archived.scheme=='baseline'].sort_values(['profile','recorded_type'])
    assert np.array_equal(np.bincount(profile*5+y,minlength=80),a['count'].to_numpy())
    multi=pos.sum(axis=1)>=2
    records=[]
    for rule in NAMES:
        compatible,credit,hierarchy,ties=0,0.0,0,0
        unique_n,unique_match,tied_match=0,0,0
        for i in np.flatnonzero(multi):
            cand=np.flatnonzero(pos[i])
            rank=s[i,cand] if rule=='numeric' else np.array([anchor_level(int(s[i,d]),int(d),rule) for d in cand])
            top=cand[rank==rank.max()]
            match=int(y[i] in top)
            compatible+=match;credit+=match/len(top)
            hierarchy+=int(top[np.argmax(PRIORITY[top])]==y[i])
            if len(top)==1: unique_n+=1;unique_match+=match
            else: ties+=1;tied_match+=match
        n=int(multi.sum())
        records.append(dict(rule=rule,n=n,set_matches=compatible,set_percent=100*compatible/n,
                            unique_n=unique_n,unique_matches=unique_match,tied_n=ties,tied_matches=tied_match,
                            uniform_tie_expected_matches=credit,uniform_tie_expected_percent=100*credit/n,
                            hierarchy_tie_matches=hierarchy,hierarchy_tie_percent=100*hierarchy/n))
    result=pd.DataFrame(records)
    assert result.set_matches.tolist()==[623,622,596]
    assert result.tied_n.tolist()==[249,232,150]
    raw=result[result.rule=='numeric'].iloc[0]
    assert raw.hierarchy_tie_matches==561
    assert np.isclose(raw.uniform_tie_expected_percent,74.59981167608287)
    old=pd.read_csv(R/'analysis/measurement_checks/severity_ties.csv')
    for row in result.itertuples():
        for status,n,m in [('unique',row.unique_n,row.unique_matches),('tied',row.tied_n,row.tied_matches)]:
            ref=old[(old.rule==row.rule)&(old.status==status)].iloc[0]
            assert ref.n==n and ref.matches==m
    table_dir=R/'generated/tables/supplement'
    table_dir.mkdir(parents=True,exist_ok=True)
    # Preserve the previous six rows and add comparable single-choice summaries.
    table=[r'\begin{table}[H]\singlespacing\centering\small',
        r'\caption{최고 심각도 유형의 동률 여부와 해소 방식에 따른 기록유형 일치}\label{tab:severity_ties}',
        r'\textit{A. 단독·공동 최고 유형과 기록유형의 집합 일치}\par\smallskip',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule',
        r'비교 기준 & 최고 유형 & 기록 수 & 일치 수 & 일치율, \%\\\midrule']
    for a in result.itertuples():
        for status,n,m in [('단독 최고',a.unique_n,a.unique_matches),('공동 최고',a.tied_n,a.tied_matches)]:
            table.append(f'{NAMES[a.rule]} & {status} & {n} & {m} & {100*m/n:.1f}'+r'\\')
    table += [r'\bottomrule\end{tabular*}\par\medskip',
              r'\textit{B. 전체 708건에서 일치 조건과 동률 해소 방식의 비교}\par\smallskip',
              r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule',
              r'비교 기준 & \makecell{최고 집합과 일치\\\%} & \makecell{동률 균등 해소\\예상 일치율, \%} & \makecell{동률 위계 해소\\일치율, \%}\\\midrule']
    for a in result.itertuples():
        table.append(f'{NAMES[a.rule]} & {a.set_percent:.1f} & {a.uniform_tie_expected_percent:.1f} & {a.hierarchy_tie_percent:.1f}'+r'\\')
    table += [r'\bottomrule\end{tabular*}\end{table}']
    (table_dir/'severity_tie_agreement.tex').write_text('\n'.join(table)+'\n')
    mapping,frame=mapping_table()
    (table_dir/'severity_mapping.tex').write_text(mapping)
    out=R/'generated/tie_resolution';out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'tie_resolution.csv',index=False)
    frame.to_csv(out/'severity_mapping.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(dict(source_content_sha256=digest,records=3236,
        multidomain_records=708,prior_counts_reproduced=True,
        hierarchy='sexual > physical > neglect > emotional',
        uniform_tie_rule='one of tied top types selected with equal probability',
        expected_matches='sum of per-record matching probabilities',
        none_and_outside_top='counted as nonmatches, not excluded',
        intervals='no CI for conditional expected agreement',exports='aggregates only'),indent=2)+'\n')
    print(result.round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    run(p.parse_args().source_dir)
