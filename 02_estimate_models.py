"""Estimate documented reanalysis models and save full coefficients/covariances.

Poisson and logit GLMs use product-clustered sandwich covariance. Product
fixed effects use the concentrated Poisson likelihood, retaining multi-review
products with a positive total count. A small real-data validation compares
the concentrated estimator with Poisson GLM containing explicit product dummies.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,json,time,gc
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
import patsy

def concentrated(X,y,groups,tol=1e-9):
    order=np.argsort(groups,kind='stable');X=X[order];y=y[order];groups=groups[order]
    _,g=np.unique(groups,return_inverse=True)
    starts=np.r_[0,np.flatnonzero(np.diff(g))+1];ng=len(starts)
    sums=np.add.reduceat(y,starts);n=len(y)
    scale=X.std(axis=0);scale[scale<1e-12]=1
    Z=X/scale
    def stats(b,need_h=False):
        eta=Z@b;maxima=np.maximum.reduceat(eta,starts)
        e=np.exp(eta-maxima[g]);den=np.add.reduceat(e,starts)
        mu=sums[g]*e/den[g]
        loss=((sums*(maxima+np.log(den))).sum()-y@eta)/n
        grad=Z.T@(mu-y)/n
        if not need_h:return loss,grad
        wz=mu[:,None]*Z
        sx=np.add.reduceat(wz,starts,axis=0)
        H=Z.T@wz-sx.T@(sx/sums[:,None])
        U=np.add.reduceat((y-mu)[:,None]*Z,starts,axis=0)
        return H,U,mu,grad
    opt=minimize(stats,np.zeros(X.shape[1]),jac=True,method='BFGS',
                 options={'gtol':tol,'maxiter':1000})
    H,U,mu,grad=stats(opt.x,True)
    if np.max(abs(grad))>1e-6:raise RuntimeError('Conditional likelihood gradient failed convergence: '+str(opt.message))
    eig=np.linalg.eigvalsh(H)
    if eig.min()<=0:raise RuntimeError('Conditional likelihood information is not positive definite')
    bread=np.linalg.inv(H);factor=ng/(ng-1)*(n-1)/(n-X.shape[1])
    cov= bread@(U.T@U)@bread*factor
    b=opt.x/scale;cov=cov/scale[:,None]/scale[None,:]
    return b,cov,{'n':n,'clusters':ng,'iterations':opt.nit,'optimizer_success':bool(opt.success),
       'optimizer_message':str(opt.message),'max_scaled_gradient':float(np.max(abs(grad))),
       'min_information_eigenvalue':float(eig.min()),'covariance_correction':factor}

def validation(d,out):
    counts=d.groupby('product_group').helpful_vote.agg(['size','sum','std'])
    ids=counts.loc[(counts['size']>=3)&(counts['size']<=12)&(counts['sum']>0)&(counts['std']>0)].head(80).index
    q=d.loc[d.product_group.isin(ids)].copy()
    names=['inconsistency','verified_binary','review_depth','rating_extremity']
    X=q[names].to_numpy(float);y=q.helpful_vote.to_numpy(float);g=q.product_group.to_numpy()
    b,cov,diag=concentrated(X,y,g)
    dummy=pd.get_dummies(q.product_group,dtype=float).to_numpy()
    fit=sm.GLM(y,np.column_stack([X,dummy]),family=sm.families.Poisson()).fit(
      maxiter=200,tol=1e-11,cov_type='cluster',cov_kwds={'groups':g,'use_correction':False})
    delta=float(np.max(abs(b-np.asarray(fit.params)[:len(names)])))
    if delta>1e-5:raise RuntimeError('Concentrated versus dummy GLM mismatch: '+str(delta))
    rawcov=cov/diag['covariance_correction']
    reference=np.asarray(fit.cov_params())[:len(names),:len(names)]
    covdelta=float(np.max(abs(rawcov-reference)))
    if covdelta>1e-6:raise RuntimeError('Concentrated versus dummy GLM robust covariance mismatch: '+str(covdelta))
    result={'actual_data_rows':len(q),'actual_products':len(ids),'max_absolute_coefficient_difference':delta,
      'max_absolute_uncorrected_robust_covariance_difference':covdelta,
      'dummy_glm_converged':bool(fit.converged),'passed':True,**diag}
    (out/'conditional_estimator_validation.json').write_text(json.dumps(result,indent=2))
    print('Validated concentrated estimator',result,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=root/'results';(out/'covariance').mkdir(exist_ok=True)
    d=pd.read_csv(out/'analysis_derived.csv.gz')
    d['interaction']=d.inconsistency*d.verified_binary
    controls='verified_binary + review_depth + rating_extremity + avg_word_length + avg_sentence_length + C(category) + C(year_month)'
    starcontrols='verified_binary + review_depth + avg_word_length + avg_sentence_length + C(category) + C(year_month) + C(rating_num)'
    rhs='inconsistency + interaction + '+controls
    stats=[];rows=[];contrasts=[]
    def save(name,terms,b,cov,diag,kind):
        b=np.asarray(b);cov=np.asarray(cov);se=np.sqrt(np.diag(cov));z=b/se;p=2*norm.sf(abs(z));lo=b-1.959963984540054*se;hi=b+1.959963984540054*se
        pd.DataFrame(cov,index=terms,columns=terms).to_csv(out/'covariance'/f'{name}.csv')
        for j,term in enumerate(terms):rows.append({'model':name,'term':term,'b':b[j],'se':se[j],'p':p[j],'ci_low':lo[j],'ci_high':hi[j],
          'ratio':np.exp(b[j]),'ratio_low':np.exp(lo[j]),'ratio_high':np.exp(hi[j]),'n':diag['n'],'clusters':diag['clusters'],'family':kind})
        stats.append({'model':name,'family':kind,**diag})
        for label,linear in [('verified_inconsistency_slope',{'inconsistency':1,'interaction':1}),
                             ('directional_equality',{'text_more_positive':1,'text_more_negative':-1})]:
            if all(k in terms for k in linear):
                v=np.array([linear.get(t,0) for t in terms]);est=float(v@b);se1=float(np.sqrt(v@cov@v));pv=float(2*norm.sf(abs(est/se1)))
                contrasts.append({'model':name,'contrast':label,'b':est,'se':se1,'p':pv,'ci_low':est-1.959963984540054*se1,'ci_high':est+1.959963984540054*se1,'ratio':float(np.exp(est))})
        pd.DataFrame(rows).to_csv(out/'all_coefficients.csv',index=False)
        pd.DataFrame(stats).to_csv(out/'model_diagnostics.csv',index=False)
        pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    def glm(name,q,rhs,family='poisson'):
        start=time.time();print('Fitting',name,len(q),flush=True)
        response='any_helpful' if family=='logit' else 'helpful_vote'
        yy,xx=patsy.dmatrices(response+' ~ '+rhs,q,return_type='dataframe')
        fit=sm.GLM(yy,xx,family=sm.families.Binomial() if family=='logit' else sm.families.Poisson()).fit(
          maxiter=150,tol=1e-9,cov_type='cluster',cov_kwds={'groups':q.product_group.to_numpy(),'use_correction':True},use_t=False)
        if not fit.converged:raise RuntimeError(name+' did not converge')
        save(name,list(xx.columns),fit.params,fit.cov_params(),{'n':int(fit.nobs),'clusters':q.product_group.nunique(),
          'converged':True,'seconds':time.time()-start,'deviance':float(fit.deviance),'loglike':float(fit.llf)},family)
        print('Done',name,'seconds',time.time()-start,flush=True)
        del fit,xx,yy;gc.collect()
    jobs=[('M0',d,controls,'poisson'),('M1',d,'inconsistency + '+controls,'poisson'),('M2',d,rhs,'poisson'),
      ('rating_FE',d,'inconsistency + interaction + '+starcontrols,'poisson'),
      ('directional',d,'text_more_positive + text_more_negative + text_more_positive:verified_binary + text_more_negative:verified_binary + '+starcontrols,'poisson'),
      ('any_helpful',d,rhs,'logit'),('tail_removed',d.loc[d.helpful_vote<=65],rhs,'poisson'),
      ('mixed_status',d.loc[d.groupby('product_group').verified_binary.transform('nunique')==2],rhs,'poisson'),
      ('experience',d.loc[d.product_type=='experience'],rhs,'poisson'),('search',d.loc[d.product_type=='search'],rhs,'poisson'),
      ('discrete_score',d,'discrete_inconsistency + discrete_inconsistency:verified_binary + '+controls,'poisson'),
      ('high_confidence',d.loc[d.text_max_probability>=.5],rhs,'poisson')]
    for job in jobs:glm(*job)
    validation(d,out)
    groupinfo=d.groupby('product_group').helpful_vote.agg(['size','sum'])
    keep=groupinfo.index[(groupinfo['size']>1)&(groupinfo['sum']>0)]
    q=d.loc[d.product_group.isin(keep)].copy()
    (out/'conditional_sample.json').write_text(json.dumps({'full_reviews':len(d),'full_products':d.product_group.nunique(),
      'conditional_reviews':len(q),'conditional_products':len(keep),
      'rule':'At least two sampled reviews and positive helpful-vote total; constant positive outcomes retained.',
      'zero_total_products':int((groupinfo['sum']==0).sum()),'single_review_products':int((groupinfo['size']==1).sum())},indent=2))
    for name,terms in [('product_FE_M1',['inconsistency','verified_binary','review_depth','rating_extremity','avg_word_length','avg_sentence_length']),
      ('product_FE_M2',['inconsistency','verified_binary','interaction','review_depth','rating_extremity','avg_word_length','avg_sentence_length']),
      ('product_FE_rating',['inconsistency','verified_binary','interaction','review_depth','avg_word_length','avg_sentence_length'])]:
        print('Fitting',name,len(q),flush=True);start=time.time()
        X=pd.concat([q[terms].reset_index(drop=True),pd.get_dummies(q.year_month,prefix='month',drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
        if name=='product_FE_rating':X=pd.concat([X,pd.get_dummies(q.rating_num,prefix='rating',drop_first=True,dtype=float).reset_index(drop=True)],axis=1)
        b,cov,diag=concentrated(X.to_numpy(float),q.helpful_vote.to_numpy(float),q.product_group.to_numpy())
        diag.update({'converged':True,'seconds':time.time()-start})
        save(name,list(X.columns),b,cov,diag,'conditional_poisson')
        print('Done',name,'seconds',time.time()-start,flush=True)
    # Like-for-like comparisons on the sample entering the product-FE models.
    glm('matched_sample_M1',q,'inconsistency + '+controls)
    glm('matched_sample_M2',q,rhs)
    # Re-express key descriptive tables directly from the scored data.
    d.groupby('category').agg(n=('analysis_id','size'),products=('product_group','nunique'),verified_share=('verified_binary','mean'),mean_helpful=('helpful_vote','mean'),mean_inconsistency=('inconsistency','mean')).to_csv(out/'category_descriptives.csv')
    desc=d[['helpful_vote','any_helpful','rating_num','verified_binary','review_depth','avg_word_length','avg_sentence_length','inconsistency']].describe(percentiles=[.5]).T
    desc.to_csv(out/'descriptives.csv')
    pd.DataFrame({'verified_binary':[0,1],'mean_inconsistency':[d.loc[d.verified_binary==v].inconsistency.mean() for v in [0,1]]}).to_csv(out/'inconsistency_by_verification.csv',index=False)
    (out/'estimation_complete.json').write_text(json.dumps({'status':'complete','models':len(stats),'all_converged':all(s['converged'] for s in stats),'statsmodels_version':__import__('statsmodels').__version__},indent=2))
    print('ALL MODELS COMPLETE',len(stats),flush=True)

if __name__=='__main__':main()
