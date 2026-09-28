#!/usr/bin/env python3
"""Figure 2: non-representation in records with a single concern and with
multiple concerns, with allocation benchmarks for records with multiple
concerns.

Reads aggregate outputs only and regenerates only
figures/domain_not_same_type_rates.{pdf,png}; other figures are untouched.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from plot_fonts import KOREAN_FONT
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / 'analysis'
ORDER = ('정서학대', '방임', '신체학대', '성학대')
EN = {'방임': 'Neglect', '정서학대': 'Emotional abuse', '신체학대': 'Physical abuse', '성학대': 'Sexual abuse'}
GREY, ORANGE, DARK, RAIL = '#8A949E', '#D55E00', '#26323A', '#C7CDD1'
TEXT = {
    'en': dict(single='Single concern', multi='Multiple concerns',
               uniform='Expected, uniform allocation', statement='Expected, severity-based allocation',
               xlabel='Non-representation (%)', font='DejaVu Sans'),
    'ko': dict(single='의심 하나', multi='의심 둘 이상',
               uniform='예상값: 균등 선택', statement='예상값: 심각도 우선',
               xlabel='해당 유형으로 기록되지 않은 비율 (%)', font=KOREAN_FONT),
}


def figure(lang: str) -> None:
    t = TEXT[lang]
    plt.rcParams.update({'font.family': t['font'], 'axes.unicode_minus': False, 'font.size': 9})
    data = pd.read_csv(AN / 'sensitivity_outputs/single_domain_decomposition.csv',
                       encoding='utf-8-sig').set_index('domain').loc[list(ORDER)]
    bench = pd.read_csv(AN / 'benchmark_outputs/benchmark_contrasts.csv').set_index(['domain', 'allocation'])
    single = 100 - data['single_covered_percent'].to_numpy()
    multi = 100 - data['multidomain_covered_percent'].to_numpy()
    missed = data['multidomain_n'] - data['multidomain_covered_n']
    uniform = np.array([bench.loc[(EN[d], 'uniform'), 'multidomain_nonrepresentation_percent'] for d in ORDER])
    statement = np.array([bench.loc[(EN[d], 'floor'), 'multidomain_nonrepresentation_percent'] for d in ORDER])
    observed = np.array([bench.loc[(EN[d], 'observed'), 'multidomain_nonrepresentation_percent'] for d in ORDER])
    assert np.allclose(observed, multi)
    y = np.arange(len(ORDER))
    rail = y + 0.30
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for yy, left, right in zip(y, single, multi, strict=True):
        ax.plot([left, right], [yy, yy], color=RAIL, linewidth=2, zorder=1)
    ax.scatter(single, y, s=52, marker='s', color=GREY, label=t['single'], zorder=3)
    ax.scatter(multi, y, s=58, marker='o', color=ORANGE, label=t['multi'], zorder=4)
    for yy, u, s in zip(rail, uniform, statement, strict=True):
        ax.plot([min(u, s), max(u, s)], [yy, yy], color='#E1E5E8', linewidth=1, zorder=1)
    ax.scatter(uniform, rail, s=46, marker='|', color=DARK, linewidths=1.4, label=t['uniform'], zorder=3)
    ax.scatter(statement, rail, s=30, marker='D', facecolor='white', edgecolor=DARK, linewidth=1,
               label=t['statement'], zorder=3)
    for left, right, m, total, yy in zip(single, multi, missed, data['multidomain_n'], y, strict=True):
        ax.text(left - 1.5, yy, f'{left:.1f}%', ha='right', va='center', fontsize=8)
        ax.text(right + 1.5, yy, f'{right:.1f}% ({int(m)}/{int(total)})', ha='left', va='center', fontsize=8)
    ax.set_yticks(y, [EN[d] for d in ORDER] if lang == 'en' else list(ORDER))
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_ylim(len(ORDER) - 0.35, -0.5)
    ax.set_xlabel(t['xlabel'])
    ax.grid(axis='x', color='#E1E5E8', linewidth=0.7)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=26)
    fig.subplots_adjust(bottom=0.25)
    out = ROOT / f'generated/figures/{lang}'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(out / f'domain_not_same_type_rates.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for language in ('en', 'ko'):
        figure(language)
