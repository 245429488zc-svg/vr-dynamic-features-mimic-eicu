#!/usr/bin/env python3
"""Full-cohort (association-framework) corrected-spec ladder + 5-fold CV on the REBUILT
round-2 cohort (n=10,349 complete cases).

Purpose: the Table 3 footnote cites the continuity ladder ("corrected specification")
with AUROC / CV / OOF-calibration values computed ON THE REBUILT COHORT. Produces:
  - apparent AUROCs (Base / Base + P/F / Base + P/F + VR; full spec, 18 covariates incl COPD/CHF)
  - 5-fold CV AUROCs (fold seed = 20260721, identical to the published internal-validation run)
  - optimism and out-of-fold calibration intercept/slope
Output: fullcohort_corrected_spec_cv_round2.csv + log.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
READY = OUTDIR / "mimic_analysis_ready_round2.csv"
RNG = 20260721
OUTCOME = "hospital_expire_flag"
BASE_FULL = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
             "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
             "bun", "creatinine", "wbc", "bicarbonate"]
PF, VR = ["pf_initial", "pf_min"], ["vr_initial", "vr_delta"]


def apparent(df, vars_):
    x = pd.DataFrame(StandardScaler().fit_transform(df[vars_]), columns=vars_, index=df.index)
    res = sm.Logit(df[OUTCOME].astype(int), sm.add_constant(x, has_constant="add")).fit(disp=False, maxiter=300)
    y = df[OUTCOME].astype(int)
    p = res.predict(sm.add_constant(x, has_constant="add"))
    return roc_auc_score(y, p), brier_score_loss(y, p.astype(float))


def cv_run(df, vars_, name, rows):
    y = df[OUTCOME].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(df))
    aucs_in, aucs_oof = [], []
    for tr_idx, te_idx in skf.split(df, y):
        dtr = df.iloc[tr_idx]
        m = LogisticRegression(penalty=None, max_iter=3000)
        m.fit(StandardScaler().fit(dtr[vars_]).transform(dtr[vars_]), y[tr_idx])
        aucs_in.append(roc_auc_score(y[tr_idx], m.predict_proba(StandardScaler().fit(dtr[vars_]).transform(dtr[vars_]))[:, 1]))
        sc = StandardScaler().fit(dtr[vars_])
        oof[te_idx] = m.predict_proba(sc.transform(df.iloc[te_idx][vars_]))[:, 1]
        aucs_oof.append(roc_auc_score(y[te_idx], oof[te_idx]))
    a_app, b_app = apparent(df, vars_)
    lg = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
    cres = sm.Logit(y.astype(float), sm.add_constant(lg)).fit(disp=False, maxiter=300)
    cp = np.asarray(cres.params, dtype=float)
    rows.append({"Framework": "Full cohort complete cases (corrected spec, association framework — REBUILT round-2 cohort)",
                 "Model": name, "N": len(df), "Deaths": int(y.sum()),
                 "AUROC apparent": round(a_app, 3), "AUROC cross-validated": round(float(np.mean(aucs_oof)), 3),
                 "Optimism": round(a_app - float(np.mean(aucs_oof)), 4),
                 "Brier apparent": round(b_app, 3),
                 "Calibration intercept (OOF)": round(float(cp[0]), 3), "Calibration slope (OOF)": round(float(cp[1]), 3)})


def main():
    ready = pd.read_csv(READY)
    cc = ready.dropna(subset=[OUTCOME] + BASE_FULL + PF + VR)
    print(f"重建口径全队列完整病例: n={len(cc)}, 死亡={int(cc[OUTCOME].astype(int).sum())}")
    rows = []
    for name, vars_ in [("Base", BASE_FULL), ("Base + P/F", BASE_FULL + PF), ("Base + P/F + VR", BASE_FULL + PF + VR)]:
        a, b = apparent(cc, vars_)
        print(f"[apparent] {name}: AUROC={a:.4f}, Brier={b:.3f}")
        cv_run(cc, vars_, name, rows)
        print(f"[CV] {name}: rows已记")
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "fullcohort_corrected_spec_cv_round2.csv", index=False)
    print(df.to_string(index=False))
    (OUTDIR / "fullcohort_corrected_spec_cv_log.md").write_text(
        "# 重建口径全队列（10,349）修正规格 ladder + 5-fold CV\n\nfold seed=20260721（与已发表内部验证一致）。\n\n" +
        df.to_string(index=False), encoding="utf-8")
    print("完成")


if __name__ == "__main__":
    main()
