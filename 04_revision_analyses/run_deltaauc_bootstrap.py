#!/usr/bin/env python3
"""Bootstrap 95% CIs for ΔAUROC (R1.6 / R2.5).

Replicates the published Table 3 incremental analysis exactly
(12_Table3正式版/run_revised_incremental_pf_vr_pf_initial_min.py):
  - MIMIC internal: Base(A) vs Base(A)+P/F vs +VR, complete cases, refit per resample
  - MIMIC harmonized: Base(B) vs Base(B)+P/F vs +VR, complete cases, refit per resample
  - eICU external: MIMIC-fitted harmonized models applied with FIXED coefficients,
    eICU resampled (validation setting; no refitting)
2,000 stratified bootstrap resamples; percentile 95% CIs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
OUT = ROOT / "35_增量bootstrapCI"
OUT.mkdir(exist_ok=True)
OUTCOME = "hospital_expire_flag"
RNG = 20260923
B = 2000

BASE_A = ["age", "sex", "charlson", "sepsis3", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
BASE_B = ["age", "sex", "charlson", "sepsis3", "peep", "vt_pbw", "hr",
          "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def fit_auc(df: pd.DataFrame, vars_: list[str]) -> float:
    x = StandardScaler().fit_transform(df[vars_])
    y = df[OUTCOME].astype(int).values
    m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m.fit(x, y)
    return roc_auc_score(y, m.predict_proba(x)[:, 1])


def fixed_model(df: pd.DataFrame, vars_: list[str]):
    scaler = StandardScaler().fit(df[vars_])
    x = scaler.transform(df[vars_])
    y = df[OUTCOME].astype(int).values
    m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m.fit(x, y)
    return m, scaler, y


def auc_fixed(model, x: np.ndarray, y: np.ndarray) -> float:
    return roc_auc_score(y, model.predict_proba(x)[:, 1])


def boot_delta(df: pd.DataFrame, old_vars: list[str], new_vars: list[str],
               refit: bool, fixed: dict | None, rng: np.random.Generator) -> tuple[float, float]:
    """Returns (delta_point, deltas_from_bootstrap). refit=True: refit per resample.
    Stratified resampling (positives and negatives resampled separately)."""
    y = df[OUTCOME].astype(int).values
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    deltas = np.empty(B)
    mold = mnew = None
    if not refit:  # fixed MIMIC coefficients
        mold, _, _ = fixed["old"]
        mnew, _, _ = fixed["new"]
    for b in range(B):
        samp = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        sub = df.iloc[samp]
        if refit:
            a_old = fit_auc(sub, old_vars)
            a_new = fit_auc(sub, new_vars)
        else:
            a_old = roc_auc_score(sub[OUTCOME].astype(int).values,
                                  mold.predict_proba(fixed["old"][1].transform(sub[old_vars].values))[:, 1])
            a_new = roc_auc_score(sub[OUTCOME].astype(int).values,
                                  mnew.predict_proba(fixed["new"][1].transform(sub[new_vars].values))[:, 1])
        deltas[b] = a_new - a_old
    point = float(np.mean(deltas))
    return point, deltas


def main() -> None:
    rng = np.random.default_rng(RNG)
    mimic = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv")
    eicu = pd.read_csv(ROOT / "1_队列数据" / "eICU_analysis_ready.csv")

    rows = []
    # ---- Setting 1: MIMIC internal (refit per resample)
    a_vars = BASE_A + PF
    a_new = BASE_A + PF + VR
    df1 = mimic.dropna(subset=[OUTCOME] + a_new).copy()
    stamp(f"S1 MIMIC internal: n={len(df1)}, events={int(df1[OUTCOME].sum())}")
    point1 = fit_auc(df1, a_new) - fit_auc(df1, a_vars)
    _, d1 = boot_delta(df1, a_vars, a_new, True, None, rng)
    rows.append(("MIMIC-IV internal model", len(df1), int(df1[OUTCOME].sum()), point1, d1))

    # ---- Setting 2: MIMIC harmonized (refit per resample)
    b_vars = BASE_B + PF
    b_new = BASE_B + PF + VR
    df2 = mimic.dropna(subset=[OUTCOME] + b_new).copy()
    stamp(f"S2 MIMIC harmonized: n={len(df2)}, events={int(df2[OUTCOME].sum())}")
    point2 = fit_auc(df2, b_new) - fit_auc(df2, b_vars)
    _, d2 = boot_delta(df2, b_vars, b_new, True, None, rng)
    rows.append(("MIMIC-IV harmonized reduced model", len(df2), int(df2[OUTCOME].sum()), point2, d2))

    # ---- Setting 3: eICU external (fixed MIMIC coefficients)
    df3 = eicu.dropna(subset=[OUTCOME] + b_new).copy()
    stamp(f"S3 eICU external: n={len(df3)}, events={int(df3[OUTCOME].sum())}")
    mo, so, yo = fixed_model(df2, b_vars)
    mn, sn, yn = fixed_model(df2, b_new)
    fixed = {"old": (mo, so, yo), "new": (mn, sn, yn)}
    point3 = roc_auc_score(df3[OUTCOME].astype(int).values,
                           mn.predict_proba(sn.transform(df3[b_new].values))[:, 1]) - \
             roc_auc_score(df3[OUTCOME].astype(int).values,
                           mo.predict_proba(so.transform(df3[b_vars].values))[:, 1])
    a3_before = roc_auc_score(df3[OUTCOME].astype(int).values, mo.predict_proba(so.transform(df3[b_vars].values))[:, 1])
    a3_after = roc_auc_score(df3[OUTCOME].astype(int).values, mn.predict_proba(sn.transform(df3[b_new].values))[:, 1])
    stamp(f"S3 迁移点估计核对: eICU before={a3_before:.4f}, after={a3_after:.4f} (发表值 0.771/0.787)")
    _, d3 = boot_delta(df3, b_vars, b_new, False, fixed, rng)
    rows.append(("eICU external validation (transported)", len(df3), int(df3[OUTCOME].sum()), point3, d3))

    out = []
    for name, n, ev, point, d in rows:
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append({"Setting": name, "N": n, "Deaths": ev,
                    "Delta AUROC": f"{point:.4f}",
                    "Bootstrap 95% CI": f"{lo:.4f} to {hi:.4f}",
                    "Bootstrap SE": f"{np.std(d, ddof=1):.4f}"})
        stamp(f"{name}: ΔAUROC={point:.4f}, 95% CI {lo:.4f}-{hi:.4f}, SE={np.std(d, ddof=1):.4f}")
    pd.DataFrame(out).to_csv(OUT / "delta_auroc_bootstrap.csv", index=False)
    np.save(OUT / "delta_boot_values.npy", np.vstack([r[4] for r in rows]))
    stamp(f"已保存: {OUT / 'delta_auroc_bootstrap.csv'}")
    (OUT / "bootstrap_log.md").write_text(
        "# ΔAUROC bootstrap 95% CI 日志（R1.6/R2.5）\n\n"
        f"- 复现口径：12_Table3正式版/run_revised_incremental_pf_vr_pf_initial_min.py（BASE_A/BASE_B不含COPD/CHF；P/F=initial+min；VR=initial+delta）。\n"
        f"- Bootstrap：分层（按结局）有放回重采样 B={B}，百分位95%CI；推导设置每次重采样重新拟合模型，eICU为固定MIMIC系数的迁移评估（不重拟合）。\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()