#!/usr/bin/env python3
"""Render domain score distributions using aggregate counts only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_fonts import KOREAN_FONT
import pandas as pd
import numpy as np


def render(source, output, language):
    data = pd.read_csv(source)
    data = data[data.scope.eq('all_records')]
    ko = language == 'ko'
    plt.rcParams.update({'font.family': KOREAN_FONT if ko else 'DejaVu Sans',
                         'font.size': 9, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    domains = ['Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse']
    labels = ['방임', '정서학대', '신체학대', '성학대'] if ko else domains
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True, sharey=True)
    for ax, domain, label, cutoff in zip(axes.flat, domains, labels, [4, 5, 5, 5]):
        d = data[data.domain.eq(domain)].sort_values('score')
        assert len(d) == 11 and d.total.sum() == 3236
        assert (d.total == d.same_type+d.not_same).all()
        # Zero counts remain in the complete score-support table; omit them
        # here so positive-score structure is legible on a linear frequency axis.
        d = d[d.score.gt(0)]
        ax.bar(d.score, d.same_type, color='#0072B2', width=.78,
               label='같은 유형으로 기록' if ko else 'Recorded as this type')
        ax.bar(d.score, d.not_same, bottom=d.same_type, color='#C6CDD3', width=.78,
               label='다른 유형·해당 없음으로 기록' if ko else 'Recorded as another type or none')
        ax.axvline(cutoff, linestyle='--', color='#A54E00', linewidth=1.1,
                   label='기준점' if ko else 'Cutoff')
        ax.set_title(label, loc='left', fontsize=10, fontweight='bold')
        ax.set_axisbelow(True); ax.grid(axis='y', color='#E5E8EB', linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xticks(np.arange(1,11)); ax.set_xlim(.4, 10.6); ax.set_ylim(0, 820)
        for row in d.itertuples():
            if row.total >= 20:
                ax.text(row.score, row.total+12, str(row.total), ha='center', fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel('영역 점수 (0점 제외)' if ko else 'Domain score (zero omitted)')
    for ax in axes[:,0]:
        ax.set_ylabel('기록 수' if ko else 'Records')
    handles, legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.52,.005), fontsize=8)
    fig.subplots_adjust(left=.09,right=.99,top=.95,bottom=.15,hspace=.30,wspace=.13)
    output.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(output/f'score_distribution.{ext}',dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--language',choices=['en','ko'],default='en')
    args=parser.parse_args()
    render(args.input,args.out_dir,args.language)
