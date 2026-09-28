"""Recreate the manuscript UpSet plot, including recorded-type composition."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_fonts import KOREAN_FONT
ROOT=Path(__file__).resolve().parents[1]
KO=['방임','정서학대','신체학대','성학대']
EN=['Neglect','Emotional abuse','Physical abuse','Sexual abuse']

def profile_figure(lang):
    joint=pd.read_csv(ROOT/'analysis/sensitivity_outputs/combination_record_matrix.csv',encoding='utf-8-sig')
    assert joint[joint.n_domains.gt(0)].profile_n.tolist()==[273,197,155,146,290,155,44,34,16,10,137,14,6,0,2]
    assert joint[joint.n_domains.gt(0)].profile_n.sum()==1479
    ko=lang=='ko';labels=KO if ko else EN
    out=ROOT/'generated/figures'/lang;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':KOREAN_FONT if ko else 'DejaVu Sans','font.size':10,
                         'axes.unicode_minus':False,'pdf.fonttype':42})
    plot=joint[joint.n_domains.gt(0)].copy()
    x=np.arange(len(plot)); colors=['#009E73','#CC79A7','#0072B2','#E69F00','#C7CCD1']
    fig,(ax,mat)=plt.subplots(2,1,figsize=(9.5,5.6),sharex=True,
                            gridspec_kw={'height_ratios':[3.1,1.15],'hspace':.07})
    bottom=np.zeros(len(plot))
    for col,label,color in zip(KO+['해당 없음'],labels+(['해당 없음'] if ko else ['None']),colors):
        vals=plot[col].to_numpy();ax.bar(x,vals,bottom=bottom,width=.73,color=color,label=label,edgecolor='white',linewidth=.35)
        bottom+=vals
    np.testing.assert_array_equal(bottom,plot.profile_n)
    for i,n in enumerate(plot.profile_n):ax.text(i,n+5,str(n),ha='center',fontsize=8.5)
    ax.set_ylim(0,330);ax.set_ylabel('기록 수' if ko else 'Records');ax.set_axisbelow(True)
    ax.grid(axis='y',color='#E5E8EB',linewidth=.6);ax.spines[['top','right','bottom']].set_visible(False)
    ax.tick_params(axis='x',bottom=False,labelbottom=False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=5,frameon=False,fontsize=9,
              title='기록된 유형' if ko else 'Recorded type',title_fontsize=9)
    for i,p in enumerate(plot.profile):
        active=[j for j,d in enumerate(KO) if d in p.split('+')]
        mat.scatter([i]*4,range(4),s=22,color='#DFE3E6')
        mat.plot([i]*len(active),active,color='#303941',lw=1.5)
        mat.scatter([i]*len(active),active,s=30,color='#303941')
    mat.set_yticks(range(4),labels);mat.set_ylim(3.5,-.5);mat.set_xticks([])
    mat.set_xlim(-.6,len(plot)-.4);mat.spines[:].set_visible(False);mat.tick_params(axis='y',length=0)
    mat.set_xlabel('의심 조합' if ko else 'Combination of concerns',labelpad=10)
    fig.subplots_adjust(left=.15,right=.99,bottom=.09,top=.80)
    for ext in ['png','pdf']:fig.savefig(out/f'profile_upset.{ext}',dpi=240,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

if __name__=='__main__':
    for lang in ('ko','en'):profile_figure(lang)
