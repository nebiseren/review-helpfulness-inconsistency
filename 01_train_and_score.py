"""Train, calibrate and score the text-rating model using retained input files.

Model selection uses review ratings in product-disjoint development data only.
Helpful-vote outcomes are never supplied to the text estimators or calibrator.
Run from the analysis directory with Python and the pinned requirements.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse, json, hashlib, platform, time, warnings
import numpy as np
import pandas as pd
import scipy, sklearn, joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, confusion_matrix

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    for name in ['private','models','results','logs']:(out/name).mkdir(exist_ok=True)
    seed=20260908
    config={'analysis_type':'new documented reanalysis on retained empirical data',
      'seed':seed,'main_sample':'existing audited 479493 rows; rechecked below',
      'development_exclusion':'exclude products from all 480000 original main-sample rows',
      'placeholder_text_values':['','na','n/a','none','nan','null'],
      'duplicate_key':['category','parent_asin','asin','user_id','review_date','rating_num','helpful_vote','verified_binary','text'],
      'split':'GroupShuffleSplit 80% train / 10% validation / 10% test by parent_asin; validation split equally by product into mixture-selection and calibration subsets',
      'vectorizer':{'n_features':2**17,'ngram_range':[1,2],'lowercase':True,'strip_accents':'unicode','norm':'l2','alternate_sign':True,'dtype':'float64','analyzer':'word','token_pattern':r'(?u)\b\w\w+\b'},
      'classifier':{'loss':'log_loss','penalty':'l2','alpha':1e-5,'max_iter':100,'tol':1e-4,'n_iter_no_change':5,'shuffle':True,'random_state':seed,'learning_rate':'optimal','average':True,'early_stopping':False,'n_jobs':2},
      'mixture_grid':list(np.round(np.arange(0,.50001,.05),2)),
      'selection':'calibrators fitted on calibration half of validation; select mixture using calibrated MAE in separate selection half, ties choose lower balanced weight; refit chosen calibrator on full validation only; test used once',
      'original_settings_recovered':False}
    (out/'models/training_config.json').write_text(json.dumps(config,indent=2))
    mainpath=a.data_dir/'analysis_clean_unscored.csv.gz'
    original=a.data_dir/'final_480k_with_text_v4.csv.gz'
    pilotpath=a.data_dir/'pilot_analysis_base.csv.gz'
    base=pd.read_csv(original,usecols=['parent_asin'],keep_default_na=False)
    pilot=pd.read_csv(pilotpath,keep_default_na=False)
    pilot['source_pilot_row']=np.arange(len(pilot))
    p=pilot.loc[~pilot.parent_asin.isin(base.parent_asin)].copy()
    after_exclusion=len(p)
    bad=p.text.str.strip().str.lower().isin(config['placeholder_text_values'])
    bad_n=int(bad.sum());p=p.loc[~bad].copy()
    dup=p.duplicated(config['duplicate_key']);dup_n=int(dup.sum());p=p.loc[~dup].reset_index(drop=True)
    assert p.rating_num.between(1,5).all()
    group=p.parent_asin.to_numpy();idx=np.arange(len(p))
    tr,rest=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(idx,groups=group))
    va0,te0=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(rest,groups=group[rest]))
    va=rest[va0];te=rest[te0]
    c0,s0=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+2).split(va,groups=group[va]))
    cal=va[c0];sel=va[s0]
    sets=[set(group[x]) for x in [tr,va,te]]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert not set(group)&set(base.parent_asin)
    manifest=p[['source_pilot_row','parent_asin','rating_num','category']].copy()
    manifest['split']='';manifest.loc[tr,'split']='train';manifest.loc[va,'split']='validation';manifest.loc[te,'split']='test'
    manifest['validation_role']='';manifest.loc[cal,'validation_role']='calibration';manifest.loc[sel,'validation_role']='selection'
    manifest.to_csv(out/'private/development_split_manifest.csv.gz',index=False,compression='gzip')
    vector=HashingVectorizer(**{**config['vectorizer'],'ngram_range':(1,2),'dtype':np.float64})
    X=vector.transform(p.text);y=p.rating_num.to_numpy().astype(int)
    print('Development',len(p),'products',len(set(group)),'nnz',X.nnz,flush=True)
    models=[]
    for cw in [None,'balanced']:
        model=SGDClassifier(**config['classifier'],class_weight=cw)
        print('Training class_weight',cw,flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X[tr],y[tr])
        if any(issubclass(w.category,sklearn.exceptions.ConvergenceWarning) for w in caught):
            raise RuntimeError('SGD did not converge; inspect training before proceeding')
        models.append(model)
        print('epochs',model.n_iter_,flush=True)
    classes=models[0].classes_;assert np.array_equal(classes,np.arange(1,6))
    pv=[m.predict_proba(X[va]) for m in models]
    vamap={v:i for i,v in enumerate(va)}
    ci=np.array([vamap[v] for v in cal]);si=np.array([vamap[v] for v in sel])
    grid=[]
    for w in config['mixture_grid']:
        probs=(1-w)*pv[0]+w*pv[1];raw=probs@classes
        iso=IsotonicRegression(y_min=1,y_max=5,out_of_bounds='clip').fit(raw[ci],y[cal])
        grid.append({'balanced_weight':w,'selection_mae':mean_absolute_error(y[sel],iso.predict(raw[si]))})
    chosen=min(grid,key=lambda r:(r['selection_mae'],r['balanced_weight']))
    weight=chosen['balanced_weight'];valraw=((1-weight)*pv[0]+weight*pv[1])@classes
    iso=IsotonicRegression(y_min=1,y_max=5,out_of_bounds='clip').fit(valraw,y[va])
    pt=(1-weight)*models[0].predict_proba(X[te])+weight*models[1].predict_proba(X[te])
    argmax=classes[np.argmax(pt,axis=1)];raw=pt@classes;expected=iso.predict(raw)
    metrics={'development_n':len(p),'development_products':p.parent_asin.nunique(),
      'after_product_exclusion_n':after_exclusion,'placeholder_exclusions':bad_n,'duplicate_exclusions':dup_n,
      'splits':{name:{'reviews':len(ix),'products':len(set(group[ix]))} for name,ix in [('train',tr),('validation',va),('test',te),('validation_calibration',cal),('validation_selection',sel)]},
      'balanced_weight':weight,'selection_mae':chosen['selection_mae'],
      'test_exact_accuracy':accuracy_score(y[te],argmax),'test_within_one':float(np.mean(abs(y[te]-argmax)<=1)),
      'test_macro_f1':f1_score(y[te],argmax,average='macro'),'test_raw_mae':mean_absolute_error(y[te],raw),'test_calibrated_mae':mean_absolute_error(y[te],expected),
      'test_rating_counts':pd.Series(y[te]).value_counts().sort_index().to_dict(),
      'classifier_epochs':[m.n_iter_ for m in models],
      'source_hashes':{mainpath.name:sha(mainpath),pilotpath.name:sha(pilotpath),original.name:sha(original)},
      'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__}}
    pd.DataFrame(grid).to_csv(out/'results/mixture_selection.csv',index=False)
    pd.DataFrame(confusion_matrix(y[te],argmax,labels=classes),index=classes,columns=classes).to_csv(out/'results/test_confusion_matrix.csv')
    pd.DataFrame({'source_pilot_row':p.iloc[te].source_pilot_row,'observed_rating':y[te],'argmax':argmax,'raw_expected':raw,'calibrated_expected':expected}).to_csv(out/'private/test_predictions.csv.gz',index=False,compression='gzip')
    joblib.dump({'vectorizer':vector,'unweighted':models[0],'balanced':models[1],'balanced_weight':weight,'calibrator':iso,'config':config},out/'models/text_model.joblib',compress=3)
    del X,p,pilot,base,pv,pt
    d=pd.read_csv(mainpath,keep_default_na=False)
    assert len(d)==479493
    assert not d.review_text.str.strip().str.lower().isin(config['placeholder_text_values']).any()
    preds=[]
    for start in range(0,len(d),20000):
        x=vector.transform(d.review_text.iloc[start:start+20000])
        pr=(1-weight)*models[0].predict_proba(x)+weight*models[1].predict_proba(x)
        preds.append(np.column_stack([iso.predict(pr@classes),classes[pr.argmax(axis=1)],pr.max(axis=1)]))
        print('Scored',min(start+20000,len(d)),flush=True)
    scores=np.vstack(preds)
    for j,col in enumerate(['text_expected_star_calibrated','text_argmax_star','text_max_probability']):d[col]=scores[:,j]
    d['inconsistency']=abs(d.rating_num-d.text_expected_star_calibrated)
    signed=d.text_expected_star_calibrated-d.rating_num
    d['text_more_positive']=signed.clip(lower=0);d['text_more_negative']=(-signed).clip(lower=0)
    d['discrete_inconsistency']=abs(d.rating_num-d.text_argmax_star)
    d['analysis_id']=np.arange(1,len(d)+1)
    d.to_csv(out/'private/analysis_clean_scored_private.csv.gz',index=False,compression='gzip')
    public=['analysis_id','category','product_type','year','month','year_month','rating_num','rating_extremity','helpful_vote','any_helpful','verified_binary','word_count','sentence_count','char_count','review_depth','avg_word_length','avg_sentence_length','text_expected_star_calibrated','text_argmax_star','text_max_probability','inconsistency','text_more_positive','text_more_negative','discrete_inconsistency']
    pub=d[public].copy();pub['product_group']=pd.factorize(d.parent_asin,sort=True)[0]+1
    pub.to_csv(out/'results/analysis_derived.csv.gz',index=False,compression='gzip')
    metrics.update({'main_n':len(d),'main_products':d.parent_asin.nunique(),'scores_all_finite':bool(np.isfinite(scores).all()),'scores_min':float(scores[:,0].min()),'scores_max':float(scores[:,0].max()),'mean_inconsistency':float(d.inconsistency.mean())})
    (out/'results/text_model_metrics.json').write_text(json.dumps(metrics,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else float(v)))
    print(json.dumps(metrics,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else float(v)),flush=True)

if __name__=='__main__':main()
