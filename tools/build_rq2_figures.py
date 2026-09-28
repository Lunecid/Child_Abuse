from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated/figures/rq2"
OUT.mkdir(parents=True,exist_ok=True)
SOURCE = ROOT / 'analysis/benchmark_outputs/cooccurrence_share_ci.csv'
TABLE = ROOT / 'checks/manuscript_tables/sample_definition_comparison.tex'
PANEL = TABLE.with_name('cooccurrence_panel.tex')
with SOURCE.open() as f:
    rows = [r for r in csv.DictReader(f) if r['scheme'] == 'baseline']
assert [r['domain'] for r in rows] == ['Neglect', 'Emotional abuse', 'Physical abuse', 'Sexual abuse']
table, panel = TABLE.read_text(), PANEL.read_text()
for r in rows:
    for k in list(r):
        if k not in ('scheme', 'domain'):
            r[k] = float(r[k])
    s, t, shared = (int(r[k]) for k in ('score_n', 'type_n', 'shared'))
    r['union_n'] = s + t - shared
    r['jaccard'] = shared / r['union_n']
    r['score_only'] = s - shared
    r['type_only'] = t - shared
    assert f'{s} & {t} & {shared} & {s-shared} & {t-shared}' in table
    assert f"{r['jaccard']:.3f}"[1:] in table
    for prefix in ('score', 'type'):
        assert np.isclose(r[f'{prefix}_percent'], 100*r[f'{prefix}_cooccurring']/r[f'{prefix}_n'])
        assert f"{int(r[f'{prefix}_cooccurring'])}/{int(r[f'{prefix}_n'])}" in panel
        assert f"{r[f'{prefix}_percent']:.1f}" in panel
    assert np.isclose(r['difference_pp'], r['score_percent']-r['type_percent'])
    for k in ('difference_pp', 'ci_lower_pp', 'ci_upper_pp'):
        assert f'{abs(r[k]):.1f}' in panel

with (OUT/'rq2_plot_data.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

BLUE, GOLD, DARK = '#25638B', '#B07727', '#25313B'
import sys
sys.path.insert(0,str(ROOT/'analysis'))
from plot_fonts import KOREAN_FONT
L = {
    'ko': dict(domains=['방임','정서학대','신체학대','성학대'],
        series=['점수 기준 표본','기록유형 기준 표본'],
        titles=['A  표본 크기','B  두 표본의 중복 정도','C  다른 학대유형의 동반 의심률','D  동반 의심률 차이와 95% 신뢰구간'],
        subtitles=['각 기준으로 선정된 회기 기록 수','Jaccard 유사도 = 공통 기록 수 / 두 표본의 합집합 기록 수',
            '각 표본에서 다른 영역도 기준점을 충족한 기록의 비율',
            '점수 기준 표본의 비율 - 기록유형 기준 표본의 비율'],
        axes=['회기 기록 수 (건)','Jaccard 유사도','동반 의심률 (%)','동반 의심률 차이 (%포인트)']),
    'en': dict(domains=['Neglect','Emotional abuse','Physical abuse','Sexual abuse'],
        series=['Score-defined sample','Recorded-type-defined sample'],
        titles=['A  Sample size','B  Overlap between samples','C  Co-occurring suspected maltreatment','D  Difference in co-occurrence with 95% CI'],
        subtitles=['Session records selected under each definition','Jaccard similarity = shared records / records in either sample',
            'Records meeting the threshold in at least one other domain',
            'Score-defined percentage - recorded-type-defined percentage'],
        axes=['Number of session records','Jaccard similarity','Records with co-occurring suspicion (%)','Difference (percentage points)']),
}

def create(lang):
    plt.rcParams.update({'font.family': KOREAN_FONT if lang=='ko' else 'DejaVu Sans',
        'font.size':9, 'text.color':DARK, 'axes.labelcolor':DARK,
        'xtick.color':DARK, 'ytick.color':DARK, 'pdf.fonttype':42,
        'ps.fonttype':42, 'svg.fonttype':'path', 'axes.unicode_minus':False})
    d=L[lang]
    fig, axes=plt.subplots(4,1,figsize=(6.5,8.0))
    fig.subplots_adjust(left=.22,right=.96,bottom=.06,top=.90,hspace=.70)
    legend=[Patch(facecolor=BLUE,edgecolor=BLUE,label=d['series'][0]),
        Patch(facecolor='white',edgecolor=GOLD,hatch='////',label=d['series'][1])]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,fontsize=8.5,handlelength=1.5)
    y=np.arange(4)
    for i,ax in enumerate(axes.flat):
        ax.set_ylim(3.65,-.65)
        ax.set_yticks(y,d['domains'])
        ax.tick_params(axis='y',length=0,pad=6)
        ax.tick_params(axis='x',length=3,color='#9AA2A9')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#9AA2A9')
        ax.grid(axis='x',color='#E6E9EC',linewidth=.65,zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(d['axes'][i],labelpad=3)
        ax.text(0,1.08,d['titles'][i],transform=ax.transAxes,fontsize=10,weight='bold',ha='left')

    for ax,key in [(axes[0],'n'),(axes[2],'percent')]:
        for j,prefix in enumerate(('score','type')):
            values=np.array([r[f'{prefix}_{key}'] for r in rows])
            yy=y+(-.24 if j==0 else .24)
            ax.barh(yy,values,height=.29,color=BLUE if j==0 else 'white',
                edgecolor=BLUE if j==0 else GOLD,hatch=None if j==0 else '////',linewidth=1,zorder=3)
            for pos,v in zip(yy,values):
                label=f'{int(v):,}' if key=='n' else f'{v:.1f}%'
                ax.text(v+(15 if key=='n' else 1.5),pos,label,va='center',fontsize=8)
        ax.set_xlim(0,1100 if key=='n' else 100)
        ax.set_xticks(np.arange(0,1001,250) if key=='n' else np.arange(0,101,25))
    ax=axes[1]
    vals=[r['jaccard'] for r in rows]
    ax.barh(y,vals,height=.47,color='#647886',zorder=3)
    for yy,v in zip(y,vals): ax.text(v+.025,yy,f'{v:.3f}',va='center',fontsize=8.5)
    ax.set_xlim(0,1.12); ax.set_xticks([0,.25,.5,.75,1],['0','.25','.50','.75','1.00'])
    ax=axes[3]
    ax.axvline(0,color='#65727C',linewidth=1,linestyle='--',zorder=1)
    for yy,r in zip(y,rows):
        v,lo,hi=(r[k] for k in ('difference_pp','ci_lower_pp','ci_upper_pp'))
        ax.errorbar(v,yy,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE,markersize=6,capsize=4,elinewidth=1.6,zorder=3)
        ax.text(v,yy-.22,f'{v:.1f} [{lo:.1f}, {hi:.1f}]',ha='center',va='bottom',fontsize=8,
            bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    ax.set_xlim(-7,47);ax.set_xticks([0,10,20,30,40])
    for ext in ('pdf','png','svg'):
        fig.savefig(OUT/f'rq2_sample_comparison_{lang}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

for lang in ('ko','en'): create(lang)
manifest={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'manuscript_table':str(TABLE.relative_to(ROOT)),'cooccurrence_table':str(PANEL.relative_to(ROOT)),
    'scheme':'baseline','rows':4,'checks':'Counts, Jaccard values, rates, differences and CI endpoints match manuscript tables; rates and differences recalculated from numerators/denominators.',
    'unit':'session record, not unique child','ci':'Previously computed record-level percentile bootstrap, 9,999 resamples; samples reconstructed jointly.',
    'scope':'Bilingual RQ2 manuscript figures.',
    'legacy_source_excluded':'composition_outputs/sample_composition.csv uses a different multidomain definition; use benchmark_outputs/cooccurrence_share_ci.csv to match current RQ2.'}
(OUT/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Created Korean and English PDF, PNG and SVG figures; four source rows validated.')
