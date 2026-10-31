"""Refit the text model from the public split manifest and recovered source text.

This alternative does not require private user identifiers or the historical
300000-row CSV. It checks the new model against the declared held-out metrics.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error,accuracy_score,f1_score

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--recovered-texts',type=Path,required=True);a=p.parse_args();root=a.root
    config=json.loads((root/'models/training_config.json').read_text())
    expected=json.loads((root/'results/text_model_metrics.json').read_text())
    manifest=pd.read_csv(root/'results/development_split_manifest.csv.gz',keep_default_na=False)
    text=pd.read_csv(a.recovered_texts,keep_default_na=False)
    text=text.loc[text.record_type=='development',['record_id','text']].rename(columns={'record_id':'source_pilot_row'})
    assert text.source_pilot_row.is_unique
    d=manifest.merge(text,on='source_pilot_row',how='left',validate='one_to_one',sort=False)
    if d.text.isna().any():raise RuntimeError('Some development texts are missing')
    tr=np.flatnonzero(d.split=='train');va=np.flatnonzero(d.split=='validation');te=np.flatnonzero(d.split=='test')
    cal=np.flatnonzero(d.validation_role=='calibration');sel=np.flatnonzero(d.validation_role=='selection')
    cfg={**config['vectorizer'],'ngram_range':(1,2),'dtype':np.float64}
    vector=HashingVectorizer(**cfg);X=vector.transform(d.text);y=d.rating_num.to_numpy(int)
    models=[]
    for weight in [None,'balanced']:
        model=SGDClassifier(**config['classifier'],class_weight=weight).fit(X[tr],y[tr]);models.append(model)
    classes=models[0].classes_;p0=models[0].predict_proba(X[va]);p1=models[1].predict_proba(X[va])
    vi={v:i for i,v in enumerate(va)};ci=np.array([vi[x] for x in cal]);si=np.array([vi[x] for x in sel])
    choices=[]
    for w in config['mixture_grid']:
        raw=((1-w)*p0+w*p1)@classes
        iso=IsotonicRegression(y_min=1,y_max=5,out_of_bounds='clip').fit(raw[ci],y[cal])
        choices.append((mean_absolute_error(y[sel],iso.predict(raw[si])),w))
    _,w=min(choices);raw=((1-w)*p0+w*p1)@classes
    iso=IsotonicRegression(y_min=1,y_max=5,out_of_bounds='clip').fit(raw,y[va])
    pt=(1-w)*models[0].predict_proba(X[te])+w*models[1].predict_proba(X[te]);yp=classes[pt.argmax(axis=1)]
    results={'balanced_weight':w,'test_calibrated_mae':mean_absolute_error(y[te],iso.predict(pt@classes)),
             'test_raw_mae':mean_absolute_error(y[te],pt@classes),'test_exact_accuracy':accuracy_score(y[te],yp),'test_macro_f1':f1_score(y[te],yp,average='macro')}
    differences={k:float(abs(v-expected[k])) for k,v in results.items()}
    if max(differences.values())>1e-10:raise RuntimeError('Retraining metrics differ: '+str(differences))
    joblib.dump({'vectorizer':vector,'unweighted':models[0],'balanced':models[1],'balanced_weight':w,'calibrator':iso,'config':config},root/'models/retrained_from_manifest.joblib',compress=3)
    (root/'results/manifest_retraining_check.json').write_text(json.dumps({'metrics':results,'absolute_differences':differences,'passed':True},indent=2))
    print('Manifest-based model retraining reproduced all checked held-out metrics.',differences,flush=True)

if __name__=='__main__':main()
