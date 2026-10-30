"""Compare saved text-model scores with matched original review bodies."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--recovered-texts', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum checked rows; zero checks all analysis rows')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    root = args.root
    derived = pd.read_csv(root / 'results/analysis_derived.csv.gz')
    recovered = pd.read_csv(args.recovered_texts, keep_default_na=False)
    recovered = recovered.loc[recovered.record_type == 'main', ['record_id', 'text']]
    recovered = recovered.rename(columns={'record_id': 'analysis_id'})
    if not recovered.analysis_id.is_unique:
        raise ValueError('Analysis record identifiers must be unique')
    if args.limit:
        indices = np.unique(np.linspace(0, len(derived) - 1,
                                       min(args.limit, len(derived)), dtype=int))
        derived = derived.iloc[indices].copy()
    data = derived.merge(recovered, on='analysis_id', how='left',
                         sort=False, validate='one_to_one')
    if data.text.isna().any():
        raise ValueError('Required original review bodies are missing')
    model = joblib.load(root / 'models/text_model.joblib')
    max_difference = 0.0
    for start in range(0, len(data), 20000):
        batch = data.iloc[start:start + 20000]
        features = model['vectorizer'].transform(batch.text)
        weight = model['balanced_weight']
        probabilities = ((1 - weight) * model['unweighted'].predict_proba(features)
                         + weight * model['balanced'].predict_proba(features))
        scores = model['calibrator'].predict(probabilities @ model['unweighted'].classes_)
        max_difference = max(max_difference, float(np.max(np.abs(
            scores - batch.text_expected_star_calibrated.to_numpy()))))
    result = {'rows_checked': len(data), 'maximum_absolute_score_difference': max_difference,
              'passed': max_difference < 1e-12}
    (root / 'results/source_score_check.json').write_text(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Saved and reconstructed scores differ: ' + str(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
