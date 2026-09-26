#!/usr/bin/env python3
"""Bootstrap 95% CIs for category-free NRI with event/non-event components (R1.7).

Replicates the published Table 3 models exactly
(12_Table3正式版/run_revised_incremental_pf_vr_pf_initial_min.py):
old = Base + P/F, new = Base + P/F + VR (VR initial + VR delta), category-free NRI.
  - MIMIC internal / MIMIC harmonized: refit per stratified resample
  - eICU external: MIMIC-fitted coefficients fixed (transported), eICU resampled
2,000 resamples; percentile 95% CIs. Point estimates are full-sample values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
OUT = ROOT / "35_增量bootstrap"
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


def nri_components(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> tuple[float, float, float]:
    ev = y == 1
    de = p_new[ev] - p_old[ev]
    dn = p_new[~ev] - p_old[~ev]
    nri_event = float((de > 0).mean() - (de < 0).mean())
    nri_nonevent = float((dn < 0).mean() - (dn > 0).mean())
    return nri_event + nri_nonevent, nri_event, nri_nonevent


def fit_probs(df: pd.DataFrame, vars_: list[str]) -> np.ndarray:
    x = StandardScaler().fit_transform(df[vars_])
    y = df[OUTCOME].astype(int).values
    m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m.fit(x, y)
    return m.predict_proba(x)[:, 1]


def fit_fixed(df: pd.DataFrame, vars_: list[str]):
    scaler = StandardScaler().fit(df[vars_])
    x = scaler.transform(df[vars_])
    y = df[OUTCOME].astype(int).values
    m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m.fit(x, y)
    return m, scaler


def boot_nri(df: pd.DataFrame, old_vars: list[str], new_vars: list[str],
             refit: bool, fixed: dict | None, rng: np.random.Generator) -> dict:
    y = df[OUTCOME].astype(int).values
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    out = {k: np.empty(B) for k in ["nri", "nri_event", "nri_nonevent", "idi"]}
    for b in range(B):
        samp = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        sub = df.iloc[samp]
        ys = sub[OUTCOME].astype(int).values
        if refit:
            p_old = fit_probs(sub, old_vars)
            p_new = fit_probs(sub, new_vars)
        else:
            p_old = fixed["old"][0].predict_proba(fixed["old"][1].transform(sub[old_vars].values))[:, 1]
            p_new = fixed["new"][0].predict_proba(fixed["new"][1].transform(sub[new_vars].values))[:, 1]
        n, ne, no = nri_components(ys, p_old, p_new)
        out["nri"][b] = n
        out["nri_event"][b] = ne
        out["nri_nonevent"][b] = no
        out["idi"][b] = float((p_new[ys == 1] - p_old[ys == 1]).mean() - (p_new[ys == 0] - p_old[ys == 0]).mean())
    return out


def summarize(name: str, df: pd.DataFrame, old_vars: list[str], new_vars: list[str],
              refit: bool, fixed: dict | None, rng: np.random.Generator) -> None:
    y = df[OUTCOME].astype(int).values
    if refit:
        p_old, p_new = fit_probs(df, old_vars), fit_probs(df, new_vars)
    else:
        p_old = fixed["old"][0].predict_proba(fixed["old"][1].transform(df[old_vars].values))[:, 1]
        p_new = fixed["new"][0].predict_proba(fixed["new"][1].transform(df[new_vars].values))[:, 1]
    n_pt, ne_pt, no_pt = nri_components(y, p_old, p_new)
    idi_pt = float((p_new[y == 1] - p_old[y == 1]).mean() - (p_new[y == 0] - p_old[y == 0]).mean())
    bs = boot_nri(df, old_vars, new_vars, refit, fixed, rng)
    res = {}
    for key, bkey, pt in [("NRI", "nri", n_pt), ("NRI events", "nri_event", ne_pt),
                          ("NRI nonevents", "nri_nonevent", no_pt), ("IDI", "idi", idi_pt)]:
        lo, hi = np.percentile(bs[bkey], [2.5, 97.5])
        res[key] = f"{pt:.4f} ({lo:.4f} to {hi:.4f})"
        stamp(f"{name} | {key}: {pt:.4f}, 95% CI {lo:.4f}-{hi:.4f}")
    rows.append({"Setting": name, "N": len(df), "Deaths": int(y.sum()), **res})


rows = []


def main() -> None:
    rng = np.random.default_rng(RNG)
    mimic = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv")
    eicu = pd.read_csv(ROOT / "1_队列数据" / "eICU_analysis_ready.csv")

    a_vars, a_new = BASE_A + PF, BASE_A + PF + VR
    b_vars, b_new = BASE_B + PF, BASE_B + PF + VR

    df1 = mimic.dropna(subset=[OUTCOME] + a_new).copy()
    stamp(f"S1 MIMIC internal: n={len(df1)}, events={int(df1[OUTCOME].sum())}")
    summarize("MIMIC-IV internal model", df1, a_vars, a_new, True, None, rng)

    df2 = mimic.dropna(subset=[OUTCOME] + b_new).copy()
    stamp(f"S2 MIMIC harmonized: n={len(df2)}, events={int(df2[OUTCOME].sum())}")
    summarize("MIMIC-IV harmonized reduced model", df2, b_vars, b_new, True, None, rng)

    df3 = eicu.dropna(subset=[OUTCOME] + b_new).copy()
    stamp(f"S3 eICU external: n={len(df3)}, events={int(df3[OUTCOME].sum())}")
    mo, so = fit_fixed(df2, b_vars)
    mn, sn = fit_fixed(df2, b_new)
    fixed = {"old": (mo, so), "new": (mn, sn)}
    summarize("eICU external validation (transported)", df3, b_vars, b_new, False, fixed, rng)

    pd.DataFrame(rows).to_csv(OUT / "nri_bootstrap.csv", index=False)
    stamp(f"已保存: {OUT / 'nri_bootstrap.csv'}")
    (OUT / "nri_bootstrap_log.md").write_text(
        "# NRI bootstrap 95% CI 日志（R1.7）\n\n"
        "- 口径：category-free（continuous）NRI，old=Base+P/F，new=Base+P/F+VR，与已发表Table 3模型完全一致。\n"
        f"- Bootstrap：按结局分层有放回重采样 B={B}，百分位95%CI；推导设置每次重拟合，eICU为固定MIMIC系数迁移评估。\n\n" + "\n".join(LOG),
        encoding="utf-8")


if __name__ == "__main__":
    main()