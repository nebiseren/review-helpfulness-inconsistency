"""Create manuscript tables and empirical figures directly from saved outputs."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;r=root/'results';tables=root/'tables';figs=root/'figures'
    tables.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    coef=pd.read_csv(r/'all_coefficients.csv').set_index(['model','term']);contr=pd.read_csv(r/'contrasts.csv').set_index(['model','contrast'])
    met=json.loads((r/'text_model_metrics.json').read_text());desc=pd.read_csv(r/'descriptives.csv',index_col=0)
    def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
    def bcell(x):return f"{x.b:.3f}{stars(x.p)}\n[{x.ci_low:.3f}, {x.ci_high:.3f}]"
    def ratio(x):return f"{np.exp(x.b):.3f}\n[{np.exp(x.ci_low):.3f}, {np.exp(x.ci_high):.3f}]"
    def pf(p):return '< .001' if p<.001 else f'{p:.3f}'.lstrip('0')
    def write(n,rows):pd.DataFrame(rows[1:],columns=rows[0]).to_csv(tables/f'Table_{n}.csv',index=False)
    # Copy the selected-literature table.
    original=root/'reference_materials/Table_1.csv'
    shutil.copy2(original,tables/'Table_1.csv')
    cat=pd.read_csv(r/'category_descriptives.csv')
    labels={'Beauty_and_Personal_Care':'Beauty & Personal Care','Books':'Books','CDs_and_Vinyl':'CDs & Vinyl','Electronics':'Electronics','Industrial_and_Scientific':'Industrial & Scientific','Office_Products':'Office Products'}
    rows=[['Category','Product type','N','Products','Verified (%)','Mean helpful votes','Mean inconsistency']]
    for _,x in cat.iterrows():rows.append([labels[x.category],'Experience' if x.category in list(labels)[:3] else 'Search',f'{int(x.n):,}',f'{int(x.products):,}',f'{100*x.verified_share:.1f}',f'{x.mean_helpful:.3f}',f'{x.mean_inconsistency:.3f}'])
    write(2,rows)
    test=met['splits']['test']
    rows=[['Metric','Value'],['Development reviews',f"{met['development_n']:,}"],['Development products',f"{met['development_products']:,}"],
      ['Test reviews',f"{test['reviews']:,}"],['Test products',f"{test['products']:,}"],
      ['Exact-rating accuracy',f"{100*met['test_exact_accuracy']:.1f}%"],['Accuracy within one rating point',f"{100*met['test_within_one']:.1f}%"],
      ['Macro-F1',f"{met['test_macro_f1']:.3f}"],['Calibrated expected-rating MAE',f"{met['test_calibrated_mae']:.3f}"],
      ['Raw expected-rating MAE',f"{met['test_raw_mae']:.3f}"],['Majority-class accuracy benchmark',f"{100*max(met['test_rating_counts'].values())/test['reviews']:.1f}%"]]
    write(3,rows)
    names=[('Helpful votes','helpful_vote'),('At least one helpful vote','any_helpful'),('Posted rating','rating_num'),('Rating–text inconsistency','inconsistency'),('Verified Purchase','verified_binary'),('Review depth','review_depth'),('Characters per word','avg_word_length'),('Words per detected sentence','avg_sentence_length')]
    rows=[['Variable','N','Mean','SD','Median','Maximum']]
    for label,key in names:
        x=desc.loc[key];rows.append([label,f'{int(x["count"]):,}',f'{x["mean"]:.3f}',f'{x["std"]:.3f}',f'{x["50%"]:.3f}',f'{x["max"]:.3f}'])
    write(4,rows)
    rows=[['Variable','M0','M1','M2']]
    terms=[('Rating–text inconsistency','inconsistency'),('Verified Purchase','verified_binary'),('Inconsistency × Verified','interaction'),('Review depth','review_depth'),('Rating extremity','rating_extremity'),('Characters per word','avg_word_length'),('Words per detected sentence','avg_sentence_length')]
    for label,term in terms:rows.append([label]+[bcell(coef.loc[(m,term)]) if (m,term) in coef.index else '' for m in ['M0','M1','M2']])
    rows.extend([['Category fixed effects','Yes','Yes','Yes'],['Year–month fixed effects','Yes','Yes','Yes'],['N']+[f"{int(coef.loc[(m,'Intercept')]['n']):,}" for m in ['M0','M1','M2']],['Product clusters']+[f"{int(coef.loc[(m,'Intercept')].clusters):,}" for m in ['M0','M1','M2']]])
    write(5,rows)
    specs=[('Rating fixed effects','rating_FE','inconsistency','interaction'),('Text more positive','directional','text_more_positive','text_more_positive:verified_binary'),('Text more negative','directional','text_more_negative','text_more_negative:verified_binary'),('Logit any helpful vote','any_helpful','inconsistency','interaction'),('Highest helpful-vote tail removed','tail_removed','inconsistency','interaction'),('Products with both verification statuses','mixed_status','inconsistency','interaction'),('Experience categories','experience','inconsistency','interaction'),('Search categories','search','inconsistency','interaction'),('Discrete rating discrepancy','discrete_score','discrete_inconsistency','discrete_inconsistency:verified_binary'),('Text confidence at least 0.50','high_confidence','inconsistency','interaction')]
    rows=[['Model or sample','N','Reference slope ratio\n[95% CI]','Interaction b\n[95% CI]','Interaction p']]
    for label,m,term,interaction in specs:
        x=coef.loc[(m,term)];z=coef.loc[(m,interaction)];rows.append([label,f'{int(x.n):,}',ratio(x),bcell(z),pf(z.p)])
    write(6,rows)
    rows=[['Specification','N','Pooled or unverified ratio\n[95% CI]','Interaction b\n[95% CI]','Verified ratio\n[95% CI]']]
    for label,m in [('Matched-sample M1','matched_sample_M1'),('Product fixed effects M1','product_FE_M1'),('Matched-sample M2','matched_sample_M2'),('Product fixed effects M2','product_FE_M2'),('Product and rating fixed effects M2','product_FE_rating')]:
        x=coef.loc[(m,'inconsistency')];has=(m,'interaction') in coef.index
        rows.append([label,f'{int(x.n):,}',ratio(x),bcell(coef.loc[(m,'interaction')]) if has else 'Not included',ratio(contr.loc[(m,'verified_inconsistency_slope')]) if has else 'Pooled slope'])
    write(7,rows)
    rows=[['Direction','Verification status','Ratio [95% CI]','p']]
    for label,term in [('Text more positive','text_more_positive'),('Text more negative','text_more_negative')]:
        for group in ['unverified','verified']:
            x=contr.loc[('directional',group+'_'+term)];rows.append([label,group.capitalize(),ratio(x),pf(x.p)])
    write(8,rows)
    # Existing framework and cleaning-flow figures remain numerically unchanged.
    source=root/'reference_materials/figures'
    for name in ['Figure_1_framework','Figure_2_sample_flow']:
        for ext in ['png','svg','eps','tiff']:shutil.copy2(source/(name+'.'+ext),figs/(name+'.'+ext))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        fig.savefig(figs/(name+'.eps'),format='eps',facecolor='white')
        fig.savefig(figs/(name+'.svg'),format='svg',facecolor='white')
        fig.savefig(figs/(name+'.png'),dpi=200,facecolor='white')
        temp=figs/(name+'_temp.png');fig.savefig(temp,dpi=600,facecolor='white')
        with Image.open(temp) as im:im.convert('RGB').save(figs/(name+'.tiff'),compression='raw',dpi=(600,600))
        temp.unlink();plt.close(fig)
    cm=pd.read_csv(r/'test_confusion_matrix.csv',index_col=0).to_numpy()
    frac=cm/cm.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(150/25.4,125/25.4));im=ax.imshow(frac,cmap='Blues',vmin=0,vmax=1)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{cm[i,j]:,}\n{100*frac[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if frac[i,j]>.6 else 'black')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=range(1,6),yticklabels=range(1,6),xlabel='Predicted star rating (probability argmax)',ylabel='Observed star rating')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Within observed-rating proportion');fig.tight_layout()
    export(fig,'Figure_3_test_confusion')
    selected=[('Category and time effects','M2'),('Additional rating effects','rating_FE'),('Mixed-status products','mixed_status'),('Product and time effects','product_FE_M2'),('Product rating and time effects','product_FE_rating')]
    fig,ax=plt.subplots(figsize=(150/25.4,115/25.4))
    chart=[]
    for i,(label,m) in enumerate(selected):
        for offset,group,color in [(-.13,'Unverified','#24516D'),(.13,'Verified','#985329')]:
            x=coef.loc[(m,'inconsistency')] if group=='Unverified' else contr.loc[(m,'verified_inconsistency_slope')]
            ratio1=np.exp(x.b);low=np.exp(x.ci_low);high=np.exp(x.ci_high)
            ax.errorbar(ratio1,i+offset,xerr=[[ratio1-low],[high-ratio1]],fmt='o',markersize=4,capsize=2,color=color,label=group if i==0 else None)
            chart.append({'model':m,'group':group,'ratio':ratio1,'ci_low':low,'ci_high':high})
    ax.axvline(1,color='#777777',linewidth=.8,linestyle='--');ax.set_yticks(range(len(selected)),[x[0] for x in selected]);ax.invert_yaxis()
    ax.set_xlabel('Incidence rate ratio (95% CI)');ax.legend(frameon=False,loc='lower right',fontsize=8)
    ax.grid(axis='x',color='#dddddd',linewidth=.5);fig.tight_layout()
    export(fig,'Figure_4_group_slopes');pd.DataFrame(chart).to_csv(r/'figure_4_data.csv',index=False)
    print('Built eight tables and four figures from saved reanalysis outputs')

if __name__=='__main__':main()
