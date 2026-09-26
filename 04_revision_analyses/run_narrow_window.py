#!/usr/bin/env python3
"""Narrow matching-window (|ABG–ventilator pair lag| <= 1 h) sensitivity
analysis (R2.9): VR features re-derived from pairs with pair_lag_hours <= 1,
joint fully adjusted model (VR initial + VR delta), compared with the primary
analysis (<= 2 h)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
OUT = ROOT / "41_窄窗口敏感性"
OUT.mkdir(exist_ok=True)
OUTCOME = "hospital_expire_flag"
BASE_A = ["age", "sex", "charlson", "sepsis3", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VARS = BASE_A + PF + ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def fit_logit(cc: pd.DataFrame, variables: list[str]) -> tuple[dict, pd.DataFrame]:
    x = sm.add_constant(pd.DataFrame(StandardScaler().fit_transform(cc[variables]),
                                     columns=variables, index=cc.index), has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)
    rows = []
    for var in ("vr_initial", "vr_delta"):
        beta, se = float(res.params[var]), float(res.bse[var])
        z = beta / se
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        rows.append({"variable": var, "OR_CI": f"{math.exp(beta):.3f} ({math.exp(beta-1.96*se):.3f}-{math.exp(beta+1.96*se):.3f})",
                     "p": "<0.001" if p < 0.001 else f"{p:.3g}"})
    return {"N": len(cc), "Deaths": int(y.sum()), "AUROC": f"{roc_auc_score(y, pred):.3f}",
            "Brier": f"{brier_score_loss(y, pred):.3f}"}, pd.DataFrame(rows)


def main() -> None:
    traj = pd.read_csv(ROOT / "22_研究归档_20260722" / "01_cohort_data" / "trajectory_mimic_48h.csv")
    traj = traj[traj["vr"].between(0.2, 10) & traj["pf"].between(30, 700)].copy()
    stamp(f"逐点(有效VR): {len(traj)}点/{traj['stay_id'].nunique()}人; pair_lag范围 {traj['pair_lag_hours'].min():.2f}~{traj['pair_lag_hours'].max():.2f} h")
    narrow = traj[traj["pair_lag_hours"].abs() <= 1.0].copy()
    stamp(f"|lag|≤1h 的点: {len(narrow)} ({len(narrow)/len(traj)*100:.1f}%)")

    cnt = narrow.groupby("stay_id").size()
    keep = set(cnt[cnt >= 2].index)
    rows = []
    for sid, g in narrow[narrow["stay_id"].isin(keep)].sort_values(["stay_id", "hour"]).groupby("stay_id", sort=False):
        v = g["vr"].to_numpy(float)
        rows.append({"stay_id": sid, "vr_initial": v[0], "vr_delta": v[-1] - v[0]})
    feat = pd.DataFrame(rows)
    stamp(f"窄窗口特征: {len(feat)}人 (≥2个≤1h配对点)")

    ard = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv")
    cc = ard.dropna(subset=[OUTCOME] + BASE_A + PF).drop(columns=["vr_initial", "vr_delta"]).merge(feat, on="stay_id", how="inner")
    stamp(f"窄窗口分析队列(完整病例∩≥2窄配对): n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")

    out, ors = fit_logit(cc, VARS)
    stamp(f"[±1h] N={out['N']}, 死亡={out['Deaths']}, AUROC={out['AUROC']}, Brier={out['Brier']}")
    for _, r in ors.iterrows():
        stamp(f"[±1h] {r.variable}: {r.OR_CI}, P={r.p}")
    stamp("对照(主分析±2h): VR initial 1.684 (1.545-1.835), VR delta 1.395 (1.296-1.502), AUROC 0.848")

    ors.to_csv(OUT / "narrow_window_OR.csv", index=False)
    cc[["stay_id", "vr_initial", "vr_delta"]].to_csv(OUT / "narrow_window_features.csv", index=False)
    (OUT / "narrow_window_log.md").write_text(
        "# ±1h窄窗口敏感性日志（R2.9）\n\n"
        "- 数据：trajectory_mimic_48h.csv，仅保留 |ABG–呼吸机配对时差(pair_lag_hours)| ≤ 1 h 的配对点。\n"
        "- 特征：≥2个窄配对点的患者重算 vr_initial/vr_delta；协变量与完整病例主分析一致。\n"
        "- 模型：与主分析相同的联合全调整模型（VR initial + VR delta），per 1-SD。\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()