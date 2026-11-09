"""Post hoc 2–4-star sensitivity for the pooled absolute-discrepancy models.

Usage: python code/10_interior_absolute_sensitivity.py --root .
Requires results/analysis_derived.csv.gz from Zenodo v4.
Exploratory analyses; do not interpret as independent measurement validation.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.stats import norm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    root = ap.parse_args().root
    d = pd.read_csv(root / "results/analysis_derived.csv.gz")
    d = d.loc[d.rating_num.between(2, 4)].copy()
    d["interaction"] = d.inconsistency * d.verified_binary
    controls = ("verified_binary + review_depth + rating_extremity + "
                "avg_word_length + avg_sentence_length + C(category) + C(year_month)")
    starcontrols = ("verified_binary + review_depth + avg_word_length + "
                    "avg_sentence_length + C(category) + C(year_month) + C(rating_num)")
    models = {
        "interior_M1": "inconsistency + " + controls,
        "interior_M2": "inconsistency + interaction + " + controls,
        "interior_ratingFE_M1": "inconsistency + " + starcontrols,
        "interior_ratingFE_M2": "inconsistency + interaction + " + starcontrols
    }
    records = []
    for name, rhs in models.items():
        y, X = patsy.dmatrices("helpful_vote ~ " + rhs, d, return_type="dataframe")
        fit = sm.GLM(y, X, family=sm.families.Poisson()).fit(
            maxiter=150, tol=1e-9, use_t=False, cov_type="cluster",
            cov_kwds={"groups":d.product_group.to_numpy(), "use_correction":True})
        if not fit.converged:
            raise RuntimeError("Non-converged model: " + name)
        def add(term, b, se):
            records.append({
                "model":name, "term":term, "n":int(fit.nobs),
                "product_clusters":int(d.product_group.nunique()),
                "b":b, "se":se, "irr":float(np.exp(b)),
                "irr_ci_low":float(np.exp(b-1.95996398454*se)),
                "irr_ci_high":float(np.exp(b+1.95996398454*se)),
                "p":float(2*norm.sf(abs(b/se)))})
        for term in ("inconsistency","interaction"):
            if term in fit.params.index:
                add(term, float(fit.params[term]), float(fit.bse[term]))
        if "interaction" in fit.params.index:
            terms = ["inconsistency","interaction"]
            b = float(fit.params[terms].sum())
            se = float(np.sqrt(fit.cov_params().loc[terms,terms].to_numpy().sum()))
            add("verified_combined_slope",b,se)
    target = root/"results/additional_checks/interior_absolute_sensitivity.csv"
    target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(target,index=False)
    print("Saved",len(records),"results to",target)

if __name__ == "__main__":
    main()
