"""Exploratory checks of rating support, review depth and group interactions.

These analyses were specified after inspecting earlier results. They are not
an independent validation of the text-based measurement against human ratings.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,json,time,importlib.util,gc
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import norm,chi2

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root;out=root/'results/additional_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'covariance').mkdir(exist_ok=True)
    d=pd.read_csv(root/'results/analysis_derived.csv.gz')
    d['interaction']=d.inconsistency*d.verified_binary
    positive=d.text_more_positive>0
    support=d.groupby('rating_num').agg(n=('analysis_id','size'),positive_n=('text_more_positive',lambda x:int((x>0).sum())),negative_n=('text_more_negative',lambda x:int((x>0).sum())),positive_sum=('text_more_positive','sum'),negative_sum=('text_more_negative','sum'))
    support['positive_share_within_rating']=support.positive_n/support.n
    support['share_of_positive_records']=support.positive_n/positive.sum()
    support.to_csv(out/'directional_support_by_rating.csv')
    summary={'positive_records':int(positive.sum()),'positive_records_rating_1':int(((d.rating_num==1)&positive).sum()),
             'share_rating_1_among_positive_records':float((d.loc[positive,'rating_num']==1).mean()),
             'share_positive_among_rating_1_records':float(positive.loc[d.rating_num==1].mean()),
             'share_positive_magnitude_from_rating_1':float(d.loc[d.rating_num==1,'text_more_positive'].sum()/d.text_more_positive.sum()),
             'spearman_inconsistency_review_depth':float(d[['inconsistency','review_depth']].corr(method='spearman').iloc[0,1]),
             'missing_cells_in_derived_data':int(d.isna().sum().sum()),
             'scope':'Exploratory sensitivity checks added after earlier results were examined; no human-coded validation.'}
    (out/'measurement_summary.json').write_text(json.dumps(summary,indent=2))
    test_path=root/'results/test_predictions.csv.gz'
    if test_path.exists():
        test=pd.read_csv(test_path);test['absolute_error']=abs(test.observed_rating-test.calibrated_expected)
        perf=test.groupby('observed_rating').agg(n=('argmax','size'),mean_calibrated_rating=('calibrated_expected','mean'),calibrated_mae=('absolute_error','mean'))
        perf['correct_argmax']=test.assign(correct=test.observed_rating==test.argmax).groupby('observed_rating').correct.sum()
        perf['argmax_recall']=perf.correct_argmax/perf.n
        perf.to_csv(out/'test_performance_by_rating.csv')
    spec=importlib.util.spec_from_file_location('core_estimator',root/'code/02_estimate_models.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    rows=[];diagnostics=[];contrasts=[]
    def save(name,terms,b,cov,diag):
        b=np.asarray(b);cov=np.asarray(cov);se=np.sqrt(np.diag(cov));p=2*norm.sf(abs(b/se));lo=b-1.959963984540054*se;hi=b+1.959963984540054*se
        pd.DataFrame(cov,index=terms,columns=terms).to_csv(out/'covariance'/f'{name}.csv')
        for j,t in enumerate(terms):rows.append({'model':name,'term':t,'b':b[j],'se':se[j],'p':p[j],'ci_low':lo[j],'ci_high':hi[j],'ratio':np.exp(b[j]),'ratio_low':np.exp(lo[j]),'ratio_high':np.exp(hi[j]),'n':diag['n'],'clusters':diag['clusters']})
        diagnostics.append({'model':name,**diag})
        tests={}
        if 'interaction' in terms:
            if 'experience_interaction' in terms:
                tests['verified_search_inconsistency_slope']={'inconsistency':1,'interaction':1}
                tests['unverified_experience_inconsistency_slope']={'inconsistency':1,'experience_inconsistency':1}
                tests['verified_experience_inconsistency_slope']={'inconsistency':1,'interaction':1,'experience_inconsistency':1,'experience_interaction':1}
            else:tests['verified_inconsistency_slope']={'inconsistency':1,'interaction':1}
        for term in ['text_more_positive','text_more_negative']:
            if term in terms:
                tests['unverified_'+term]={term:1}
                tests['verified_'+term]={term:1,term+':verified_binary':1}
        if 'text_more_positive' in terms:
            tests['unverified_directional_equality']={'text_more_positive':1,'text_more_negative':-1}
            tests['verified_directional_equality']={'text_more_positive':1,'text_more_negative':-1,'text_more_positive:verified_binary':1,'text_more_negative:verified_binary':-1}
        for label,mapping in tests.items():
            v=np.array([mapping.get(t,0) for t in terms]);estimate=float(v@b);err=float(np.sqrt(v@cov@v))
            contrasts.append({'model':name,'contrast':label,'b':estimate,'se':err,'p':float(2*norm.sf(abs(estimate/err))),'ci_low':estimate-1.959963984540054*err,'ci_high':estimate+1.959963984540054*err,'ratio':float(np.exp(estimate))})
        if 'experience_interaction' in terms:
            ix=[terms.index(t) for t in ['experience_inconsistency','experience_interaction']]
            v=b[ix];c=cov[np.ix_(ix,ix)];stat=float(v@np.linalg.solve(c,v))
            contrasts.append({'model':name,'contrast':'joint_category_group_slope_difference','wald_chi2':stat,'df':2,'p':float(chi2.sf(stat,2))})
        pd.DataFrame(rows).to_csv(out/'all_coefficients.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'model_diagnostics.csv',index=False)
        pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    def glm(name,q,rhs):
        start=time.time();print('Fitting',name,len(q),flush=True)
        y,x=patsy.dmatrices('helpful_vote ~ '+rhs,q,return_type='dataframe')
        if np.linalg.matrix_rank(x.to_numpy())!=x.shape[1]:raise RuntimeError('Rank-deficient design: '+name)
        fit=sm.GLM(y,x,family=sm.families.Poisson()).fit(maxiter=150,tol=1e-9,cov_type='cluster',cov_kwds={'groups':q.product_group.to_numpy(),'use_correction':True},use_t=False)
        if not fit.converged:raise RuntimeError('Nonconvergence: '+name)
        save(name,list(x.columns),fit.params,fit.cov_params(),{'n':int(fit.nobs),'clusters':int(q.product_group.nunique()),'converged':True,'seconds':time.time()-start})
        print('Done',name,round(time.time()-start,1),flush=True)
        del fit,x,y;gc.collect()
    base='verified_binary + avg_word_length + avg_sentence_length + C(category) + C(year_month) + C(rating_num)'
    glm('rating_FE_spline',d,'inconsistency + interaction + bs(review_depth, df=4, degree=3, include_intercept=False) + '+base)
    glm('directional_interior_ratings',d.loc[d.rating_num.between(2,4)],'text_more_positive + text_more_negative + text_more_positive:verified_binary + text_more_negative:verified_binary + review_depth + '+base)
    d['experience_inconsistency']=d.inconsistency*(d.product_type=='experience')
    d['experience_verified']=d.verified_binary*(d.product_type=='experience')
    d['experience_interaction']=d.interaction*(d.product_type=='experience')
    glm('category_group_interactions',d,'inconsistency + interaction + experience_inconsistency + experience_verified + experience_interaction + review_depth + '+base)
    totals=d.groupby('product_group').helpful_vote.agg(['size','sum']);keep=totals.index[(totals['size']>1)&(totals['sum']>0)]
    q=d.loc[d.product_group.isin(keep)].reset_index(drop=True)
    spline=patsy.dmatrix('bs(review_depth, df=4, degree=3, include_intercept=False) - 1',q,return_type='dataframe')
    x=pd.concat([q[['inconsistency','verified_binary','interaction','avg_word_length','avg_sentence_length']],spline,
                 pd.get_dummies(q.year_month,prefix='month',drop_first=True,dtype=float),pd.get_dummies(q.rating_num,prefix='rating',drop_first=True,dtype=float)],axis=1)
    start=time.time();print('Fitting product_FE_rating_spline',len(q),flush=True)
    b,cov,diag=core.concentrated(x.to_numpy(float),q.helpful_vote.to_numpy(float),q.product_group.to_numpy())
    diag.update(converged=True,seconds=time.time()-start);save('product_FE_rating_spline',list(x.columns),b,cov,diag)
    (out/'completion.json').write_text(json.dumps({'models':len(diagnostics),'all_converged':all(x['converged'] for x in diagnostics),'exploratory':True},indent=2))
    tables=root/'tables';tables.mkdir(exist_ok=True)
    if test_path.exists():
        support.reset_index().rename(columns={'n':'analysis_n'}).merge(perf.reset_index().rename(columns={'observed_rating':'rating_num','n':'test_n'}),on='rating_num',validate='one_to_one').to_csv(tables/'Supplementary_Table_S1.csv',index=False)
    contrasts_frame=pd.DataFrame(contrasts)
    contrasts_frame.loc[contrasts_frame.contrast.isin(['unverified_text_more_positive','unverified_text_more_negative','verified_text_more_positive','verified_text_more_negative'])].to_csv(tables/'Supplementary_Table_S2.csv',index=False)
    coefficients=pd.DataFrame(rows)
    coefficients.loc[coefficients.term.isin(['inconsistency','interaction','experience_inconsistency','experience_interaction'])].to_csv(tables/'Supplementary_Table_S3.csv',index=False)
    print('All four exploratory models converged.',flush=True)

if __name__=='__main__':main()
