#!/usr/bin/env python3
"""Task A (MIMIC internal), Round-2: the prediction ladder on the REBUILT landmark cohort.

Ladder: Base (18 covariates incl COPD/CHF) -> Base + P/F (20) -> Base + P/F + VR (22).
Framework: landmark cohort (alive & hospitalized at 48 h; outcome = death after 48 h).
Reports the apparent ladder fit, the 5-fold stratified CV (seed 20260721) and
out-of-sample calibration, plus the VR initial / VR delta ORs in the full model
(for continuity with the association tables).

Consumes: 43_第二轮重分析/landmark_cohort_round2.csv (from run_mimic_rebuild_round2.py).
"""
from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUTDIR = ROOT / "43_第二轮重分析"
OUTCOME = "hospital_expire_flag"
RNG = 20260721
BASE_A = [
    "age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
PF_48H = ["pf_initial", "pf_min"]
VR_48H = ["vr_initial", "vr_delta"]
FULL = BASE_A + PF_48H + VR_48H
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def scale_x(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    scaler = StandardScaler().fit(df[variables])
    return pd.DataFrame(scaler.transform(df[variables]), columns=variables, index=df.index)


def apparent_fit(df: pd.DataFrame, variables: list[str]) -> tuple[dict, dict]:
    import statsmodels.api as sm
    x = sm.add_constant(scale_x(df, variables), has_constant="add")
    y = df[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)
    out = {"N": len(df), "Deaths": int(y.sum()),
           "AUROC": f"{roc_auc_score(y, pred):.3f}",
           "Brier": f"{brier_score_loss(y, pred):.3f}"}
    ors = {}
    for v in ("vr_initial", "vr_delta") + (tuple(variables) if False else ()):
        if v in variables:
            beta, se = float(res.params[v]), float(res.bse[v])
            ors[v] = (math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se))
    return out, ors


def cv_model(df: pd.DataFrame, variables: list[str]) -> dict:
    y = df[OUTCOME].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(df))
    for tr, te in skf.split(df[variables], y):
        scaler = StandardScaler().fit(df.iloc[tr][variables])
        m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
        m.fit(scaler.transform(df.iloc[tr][variables]), y[tr])
        oof[te] = m.predict_proba(scaler.transform(df.iloc[te][variables]))[:, 1]
    scaler_full = StandardScaler().fit(df[variables])
    m_full = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m_full.fit(scaler_full.transform(df[variables]), y)
    p_app = m_full.predict_proba(scaler_full.transform(df[variables]))[:, 1]
    import statsmodels.api as sm
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
    res = sm.Logit(y.astype(float), sm.add_constant(logit)).fit(disp=False, maxiter=300)
    p = np.asarray(res.params, dtype=float)
    return {"auc_app": roc_auc_score(y, p_app), "auc_oof": roc_auc_score(y, oof),
            "optimism": roc_auc_score(y, p_app) - roc_auc_score(y, oof),
            "brier_app": brier_score_loss(y, p_app), "brier_oof": brier_score_loss(y, oof),
            "cal_int_oof": p[0], "cal_slope_oof": p[1]}


def main() -> None:
    lm = pd.read_csv(OUTDIR / "landmark_cohort_round2.csv")
    stamp(f"重建landmark队列: n={len(lm)}, 死亡={int(lm[OUTCOME].sum())}")
    rows = []
    for label, vars_ in [("Base", BASE_A), ("Base + P/F", BASE_A + PF_48H), ("Base + P/F + VR", FULL)]:
        app, _ = apparent_fit(lm, vars_)
        cv = cv_model(lm, vars_)
        rows.append({"Model": label, **app,
                     "AUROC (CV)": f"{cv['auc_oof']:.3f}", "Brier (CV)": f"{cv['brier_oof']:.3f}",
                     "Optimism": f"{cv['optimism']:.4f}",
                     "Calibration intercept (OOF)": f"{cv['cal_int_oof']:.3f}",
                     "Calibration slope (OOF)": f"{cv['cal_slope_oof']:.3f}"})
        stamp(f"{label}: apparent {app['AUROC']}, CV {cv['auc_oof']:.3f}, "
              f"optimism {cv['optimism']:.4f}, OOF校准 {cv['cal_int_oof']:.3f}/{cv['cal_slope_oof']:.3f}")
    _, ors = apparent_fit(lm, FULL)
    for v, (o, lo, hi) in ors.items():
        stamp(f"全模型中 {v}: OR {o:.3f} ({lo:.3f}-{hi:.3f})")
    pd.DataFrame(rows).to_csv(OUTDIR / "landmark_incremental_mimic.csv", index=False)
    (OUTDIR / "landmark_incremental_mimic_log.md").write_text(
        "# Task A - MIMIC landmark ladder（重建后）\n\n- 数据：mimic_analysis_ready_round2 的重建 landmark 队列\n"
        "- Ladder：Base(18含copd/chf) → Base+P/F → Base+P/F+VR；5折分层CV（seed=20260721）\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp("完成")


if __name__ == "__main__":
    main()