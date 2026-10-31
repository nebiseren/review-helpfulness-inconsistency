"""Verify saved scores and produce directional contrasts and reproducibility checks."""
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--private-scored',type=Path,help='Optional retained author input for score audit')
    a=ap.parse_args()
    root=a.root;r=root/'results'
    coef=pd.read_csv(r/'all_coefficients.csv');contr=pd.read_csv(r/'contrasts.csv')
    d=coef.loc[coef.model=='directional'].set_index('term')
    cov=pd.read_csv(r/'covariance/directional.csv',index_col=0)
    terms=list(d.index);rows=[]
    def add(name,mapping):
        v=np.array([mapping.get(t,0) for t in terms]);b=float(v@d.b.to_numpy());se=float(np.sqrt(v@cov.to_numpy()@v));lo=b-1.959963984540054*se;hi=b+1.959963984540054*se
        rows.append({'model':'directional','contrast':name,'b':b,'se':se,'p':float(2*norm.sf(abs(b/se))),'ci_low':lo,'ci_high':hi,'ratio':float(np.exp(b))})
    for term in ['text_more_positive','text_more_negative']:
        add('unverified_'+term,{term:1})
        add('verified_'+term,{term:1,term+':verified_binary':1})
    add('verified_directional_equality',{'text_more_positive':1,'text_more_negative':-1,'text_more_positive:verified_binary':1,'text_more_negative:verified_binary':-1})
    # Idempotent addition of computed contrasts to the exported table.
    names=[x['contrast'] for x in rows]
    contr=contr.loc[~((contr.model=='directional')&contr.contrast.isin(names))]
    contr=pd.concat([contr,pd.DataFrame(rows)],ignore_index=True)
    contr.to_csv(r/'contrasts.csv',index=False)
    pd.DataFrame(rows).to_csv(r/'directional_group_slopes.csv',index=False)
    secondary=coef.loc[coef.term.isin(['interaction','text_more_positive:verified_binary','text_more_negative:verified_binary','discrete_inconsistency:verified_binary']) & ~coef.model.isin(['M2'])].copy()
    secondary['q_BH_exploratory']=multipletests(secondary.p,method='fdr_bh')[1]
    secondary.to_csv(r/'secondary_interactions_BH.csv',index=False)
    public=pd.read_csv(r/'analysis_derived.csv.gz')
    score_audit={'status':'not_run','reason':'Optional private input was not supplied'}
    if a.private_scored is not None:
        sample=pd.read_csv(a.private_scored,keep_default_na=False)
        ix=np.unique(np.linspace(0,len(sample)-1,100,dtype=int))
        bundle=joblib.load(root/'models/text_model.joblib')
        X=bundle['vectorizer'].transform(sample.review_text.iloc[ix]);w=bundle['balanced_weight']
        pr=(1-w)*bundle['unweighted'].predict_proba(X)+w*bundle['balanced'].predict_proba(X)
        pred=bundle['calibrator'].predict(pr@bundle['unweighted'].classes_)
        error=float(np.max(abs(pred-sample.text_expected_star_calibrated.iloc[ix].to_numpy())))
        assert error<1e-12
        assert public.analysis_id.equals(sample.analysis_id)
        for col in ['helpful_vote','rating_num','inconsistency','text_expected_star_calibrated','text_more_positive','text_more_negative']:
            assert np.allclose(public[col],sample[col],rtol=0,atol=1e-12)
        score_audit={'status':'passed','rows':len(ix),'max_abs_difference':error}
    assert not {'user_id','review_text','review_title','asin','parent_asin'}&set(public.columns)
    diagnostics=pd.read_csv(r/'model_diagnostics.csv')
    assert len(diagnostics)==17 and diagnostics.converged.all()
    checks={'all_17_models_converged':True,'score_audit':score_audit,
      'derived_rows':len(public),'public_file_omits_direct_user_ids_and_review_text':True,
      'character_per_word_max_abs_error':float(np.max(abs(public.avg_word_length-public.char_count/public.word_count))),
      'words_per_sentence_max_abs_error':float(np.max(abs(public.avg_sentence_length-public.word_count/public.sentence_count))),
      'review_depth_max_abs_error':float(np.max(abs(public.review_depth-np.log1p(public.word_count)))),
      'reanalysis_date':'2026-10-30','original_historical_results_recovered':False,
      'statistical_scope':'New estimates from retained empirical data; no old coefficient or score carried forward.',
      'preregistration':'This reanalysis was not preregistered and began with knowledge of earlier results.'}
    (r/'public_release_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
