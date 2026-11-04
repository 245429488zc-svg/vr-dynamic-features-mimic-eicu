#!/usr/bin/env python3
"""Round-2 revision (R2-3①, R2-1): internal validation rebuilt on the landmark framework.

Fixes relative to 39_内部验证/run_internal_validation.py (flagged by Reviewer 2, Comment 3):
  * BASE_A now includes COPD and CHF, matching the reported fully adjusted model
    specification (Tables 2/4; 18 covariates + P/F x2 + VR x2 = 22 variables).
  * The primary prediction analysis (ladder Base / Base+P/F / Base+P/F+VR) is now
    cross-validated on the 48-hour LANDMARK cohort (alive and hospitalized at 48 h;
    outcome = hospital death after 48 h), per Reviewer 2 Comment 1.
  * A corrected full-cohort CV (complete cases, outcome = hospital mortality) is also
    reported for continuity with the previously published numbers.

Replicates exactly the cohort construction of 31_48h_landmark分析/run_landmark_48h.py
(same seeds, same window rules) and the CV mechanics of 39号
(StratifiedKFold(5, shuffle=True, seed=20260721), StandardScaler per fold,
LogisticRegression(penalty=None, max_iter=3000)).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUTDIR = ROOT / "43_第二轮重分析"
OUTDIR.mkdir(exist_ok=True)
DATA = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
ADM = Path("/Volumes/tongtong/数据库/public/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/hosp/admissions.csv.gz")
ICU = Path("/Volumes/tongtong/数据库/public/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/icu/icustays.csv.gz")

OUTCOME = "hospital_expire_flag"
RNG = 20260721
BASE_A = [
    "age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
PF_48H = ["pf_initial", "pf_min"]
VR_48H = ["vr_initial", "vr_delta"]
FULL_VARS = BASE_A + PF_48H + VR_48H
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def sm_logit(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm
    res = sm.Logit(y, sm.add_constant(x)).fit(disp=False, maxiter=300)
    p = np.asarray(res.params, dtype=float)
    return float(p[0]), float(p[1])


def oof_model(df: pd.DataFrame, vars_: list[str]) -> dict:
    y = df[OUTCOME].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(df))
    for tr, te in skf.split(df[vars_], y):
        scaler = StandardScaler().fit(df.iloc[tr][vars_])
        m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
        m.fit(scaler.transform(df.iloc[tr][vars_]), y[tr])
        oof[te] = m.predict_proba(scaler.transform(df.iloc[te][vars_]))[:, 1]
    scaler_full = StandardScaler().fit(df[vars_])
    m_full = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m_full.fit(scaler_full.transform(df[vars_]), y)
    p_app = m_full.predict_proba(scaler_full.transform(df[vars_]))[:, 1]
    auc_app, auc_oof = roc_auc_score(y, p_app), roc_auc_score(y, oof)
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
    cal = sm_logit(y.astype(float), logit)
    return {"auc_app": auc_app, "auc_oof": auc_oof, "optimism": auc_app - auc_oof,
            "brier_app": brier_score_loss(y, p_app), "brier_oof": brier_score_loss(y, oof),
            "cal_int_app": 0.0, "cal_slope_app": 1.0,
            "cal_int_oof": cal[0], "cal_slope_oof": cal[1]}


def build_landmark_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim replication of 31号 landmark logic (cc minus in-window deaths minus early discharge)."""
    df = df.copy()
    df[OUTCOME] = df[OUTCOME].astype(int)
    icu = pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime"])
    adm = pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"])
    for col in ["intime", "outtime"]:
        icu[col] = pd.to_datetime(icu[col], errors="coerce")
    for col in ["dischtime", "deathtime"]:
        adm[col] = pd.to_datetime(adm[col], errors="coerce")
    df = df.merge(icu, on="stay_id", how="left")
    df = df.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    hour48 = df["intime"] + pd.Timedelta(hours=48)
    died = df[OUTCOME] == 1
    win_death = died & (
        (df["deathtime"].notna() & (df["deathtime"] <= hour48))
        | (df["deathtime"].isna() & df["dischtime"].notna() & (df["dischtime"] <= hour48))
    )
    early_discharge = (~died) & df["dischtime"].notna() & (df["dischtime"] <= hour48)
    cc = df.dropna(subset=[OUTCOME] + FULL_VARS).copy()
    stamp(f"完整病例: n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")
    lm = cc.loc[~win_death[cc.index]].copy()
    stamp(f"完整病例剔除48h内死亡: -{len(cc) - len(lm)}")
    before_ed = len(lm)
    lm = lm.loc[~((lm[OUTCOME] == 0) & lm["dischtime"].notna() &
                  (lm["dischtime"] <= hour48[lm.index]))].copy()
    stamp(f"再剔除48h内存活出院: -{before_ed - len(lm)}; "
          f"landmark队列: n={len(lm)}, 48h后死亡={int(lm[OUTCOME].sum())}")
    return lm


def run_ladder(df: pd.DataFrame, framework: str) -> list[dict]:
    rows = []
    for label, vars_ in [
        ("Base", BASE_A),
        ("Base + P/F", BASE_A + PF_48H),
        ("Base + P/F + VR", FULL_VARS),
    ]:
        r = oof_model(df, vars_)
        rows.append({"Framework": framework, "Model": label, "N": len(df),
                     "Deaths": int(df[OUTCOME].sum()),
                     "AUROC apparent": f"{r['auc_app']:.3f}",
                     "AUROC cross-validated": f"{r['auc_oof']:.3f}",
                     "Optimism": f"{r['optimism']:.4f}",
                     "Brier apparent": f"{r['brier_app']:.3f}",
                     "Brier cross-validated": f"{r['brier_oof']:.3f}",
                     "Calibration intercept (OOF)": f"{r['cal_int_oof']:.3f}",
                     "Calibration slope (OOF)": f"{r['cal_slope_oof']:.3f}"})
        stamp(f"[{framework}] {label}: apparent={r['auc_app']:.4f}, CV={r['auc_oof']:.4f}, "
              f"optimism={r['optimism']:.4f}; OOF校准 {r['cal_int_oof']:.3f}/{r['cal_slope_oof']:.3f}; "
              f"Brier {r['brier_app']:.3f}->{r['brier_oof']:.3f}")
    return rows


def main() -> None:
    df = pd.read_csv(DATA)
    stamp(f"分析集读入: n={len(df)}")
    cc = df.dropna(subset=[OUTCOME] + FULL_VARS).copy()

    lm = build_landmark_cohort(df)
    rows: list[dict] = []
    rows += run_ladder(lm, "Landmark cohort (prediction framework)")
    rows += run_ladder(cc, "Full cohort complete cases (corrected spec, association framework)")
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "landmark_cv_internal.csv", index=False)
    stamp(f"已保存: {OUTDIR / 'landmark_cv_internal.csv'}")
    (OUTDIR / "landmark_cv_log.md").write_text(
        "# Landmark框架内部验证日志（R2-1/R2-3①）\n\n"
        "- 修复：BASE_A 纳入 COPD/CHF（18协变量），与正文完全调整模型规格（22变量族）一致。\n"
        "- 预测框架：48h landmark 队列（复制31号构造逻辑与种子），结局=48h后院内死亡。\n"
        "- 对照框架：完整病例全队列（结局=院内死亡），供回复信报告一致性用。\n"
        "- CV：StratifiedKFold(5, shuffle, seed=20260721)；折外校准= logit(p_oof) 回归。\n\n" + "\n".join(LOG),
        encoding="utf-8")


if __name__ == "__main__":
    main()