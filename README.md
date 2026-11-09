# Rating–text inconsistency, purchase verification and online review helpfulness

Analysis code for the article *Rating–text inconsistency, purchase verification and online review helpfulness* (Nebi Seren, Bursa Uludağ University).

The data files, fitted text model and full outputs are archived on Zenodo:
https://doi.org/10.5281/zenodo.23844936

## Repository contents

| Path | Purpose |
|---|---|
| `code/00_recover_source_texts.py` | Recovers matching review texts from the provider's original Amazon Reviews'23 category files using the source-matching manifest (optional; output is private) |
| `code/01_train_and_score.py` | Trains, calibrates and scores the text model from the author's retained inputs (documentation of the original run) |
| `code/02_estimate_models.py` | Estimates the 17 core regression models and saves coefficients and covariance matrices |
| `code/03_verify_and_summarise.py` | Computes group-specific contrasts and reproducibility checks |
| `code/04_tables_and_figures.py` | Builds the eight main tables and Figures 1–4; empirical figures use saved outputs |
| `code/06_retrain_from_source_manifest.py` | Retrains the text model from the public split manifest and recovered texts |
| `code/07_verify_source_scores.py` | Compares saved scores with scores recomputed from recovered texts |
| `code/08_measurement_sensitivity.py` | Runs four exploratory models and produces Tables S1–S3 |
| `code/09_draw_framework_and_flow.py` | Draws the analytical framework and sample-selection diagram |
| `code/10_interior_absolute_sensitivity.py` | Re-estimates pooled absolute-discrepancy M1/M2 for two-to-four-star reviews (post hoc) |
| `setup_repository.py` | Checks SHA-256 hashes of the Zenodo files and places them in the expected folders |
| `models/training_config.json` | Exact text-model settings |
| `data_dictionary.csv` | Variable definitions for `analysis_derived.csv.gz` |
| `MANIFEST_DICTIONARY.txt` | Field definitions for the two manifests |
| `reference_materials/` | Table 1 (selected literature); Figures 1–2 are drawn by `code/09_draw_framework_and_flow.py` |
| `requirements.txt` | Pinned Python dependencies |

## Reproducing the regressions

Python 3.12 was used.

1. Install the dependencies:
   `pip install -r requirements.txt`
2. Download `analysis_derived.csv.gz`, `source_matching_manifest.csv.gz`, `development_split_manifest.csv.gz`, `text_model.joblib` and `analysis_outputs.zip` from the current Zenodo version: https://doi.org/10.5281/zenodo.23844936 .
   Place all five files next to `setup_repository.py` without renaming or recompressing them.
   The next step verifies each file against its expected SHA-256 checksum.
3. Check and arrange the files:
   `python setup_repository.py`
4. Re-estimate the models:
   `python code/02_estimate_models.py --root .`
5. Compute contrasts and checks:
   `python code/03_verify_and_summarise.py --root .`
6. Rebuild the eight main tables and four figures:
   `python code/04_tables_and_figures.py --root .`
7. Run the four additional exploratory analyses and create Tables S1–S3:
   `python code/08_measurement_sensitivity.py --root .`

Table S1 also requires `results/test_predictions.csv.gz`, included in the archived outputs.
Figures 1–2 can be drawn separately with:
`python code/09_draw_framework_and_flow.py --root .`


## Additional post hoc rating-range analysis

The follow-up absolute-discrepancy sensitivity analysis was conducted after the original Zenodo v4 archive. It uses the same `analysis_derived.csv.gz` file, excludes one- and five-star reviews (n=95,185), and fits M1/M2 with the original rating-extremity control or with posted-rating indicators. Standard errors remain clustered by product.

Run: `python code/10_interior_absolute_sensitivity.py --root .`

Outputs: `results/additional_checks/interior_absolute_sensitivity.csv`. A copy of the coefficients and contrasts is available in this repository at `results/interior_absolute_sensitivity.csv`. The analysis is exploratory and does not independently validate the text-based discrepancy measure. The original Zenodo v4 code archive remains unchanged.

## Rebuilding the text model (optional)

Download the six Amazon Reviews'23 category review files (Beauty_and_Personal_Care, Books, CDs_and_Vinyl, Electronics, Industrial_and_Scientific, Office_Products) from https://amazon-reviews-2023.github.io/, then run:

```
python code/00_recover_source_texts.py --manifest results/source_matching_manifest.csv.gz --source-dir <folder with category files> --out private/recovered_texts.csv.gz
python code/06_retrain_from_source_manifest.py --root . --recovered-texts private/recovered_texts.csv.gz
python code/07_verify_source_scores.py --root . --recovered-texts private/recovered_texts.csv.gz
```

The recovered file contains third-party review text. Keep it private and do not redistribute it.

## Licences

- Python source code: MIT (see `LICENSE`).
- Author-created derived data, model artefacts, documentation and outputs: CC BY 4.0 (see `LICENSE_DERIVED_DATA.txt`).
- Original Amazon review content is excluded from these grants and remains subject to the provider's terms.

## Citation

Seren, N. (2026). Rating–text inconsistency, purchase verification and online review helpfulness [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23844936
