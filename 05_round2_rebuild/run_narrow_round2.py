#!/usr/bin/env python3
"""±1h narrow-window sensitivity (R2.9) — Round-2 rebuilt world.

Patches vs 41_窄窗口敏感性/run_narrow_window.py:
  * points source = trajectory_mimic_48h_round2.csv (rule-B truncation applied);
  * covariates source = mimic_analysis_ready_round2.csv;
  * BASE_A now includes COPD and CHF (18 covariates), matching the reported fully
    adjusted specification (the round-2 code-manuscript alignment).
Feature rule unchanged: >=2 narrow pairs keep the stay; vr_initial/vr_delta re-derived
from narrow pairs only; joint fully adjusted model, per 1-SD.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

OUTDIR = Path("/Volumes/tongtong/论文/icuVR与死亡的关联/43_第二轮重分析")
OUTCOME = "hospital_expire_flag"
BASE_A = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
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
        rows.append({"variable": var,
                     "OR_CI": f"{math.exp(beta):.3f} ({math.exp(beta-1.96*se):.3f}-{math.exp(beta+1.96*se):.3f})",
                     "p": "<0.001" if p < 0.001 else f"{p:.3g}",
                     "N": len(cc), "Deaths": int(y.sum()), "AUROC": f"{roc_auc_score(y, pred):.3f}",
                     "Brier": f"{brier_score_loss(y, pred):.3f}"})
    return {"N": len(cc), "Deaths": int(y.sum()), "AUROC": f"{roc_auc_score(y, pred):.3f}",
            "Brier": f"{brier_score_loss(y, pred):.3f}"}, pd.DataFrame(rows)


def main() -> None:
    traj = pd.read_csv(OUTDIR / "trajectory_mimic_48h_round2.csv")
    stamp(f"round2逐点: {len(traj)}点/{traj['stay_id'].nunique()}人")
    narrow = traj[traj["pair_lag_hours"].abs() <= 1.0].copy()
    stamp(f"|lag|≤1h: {len(narrow)} ({len(narrow)/len(traj)*100:.1f}%)")
    cnt = narrow.groupby("stay_id").size()
    keep = set(cnt[cnt >= 2].index)
    rows = []
    for sid, g in narrow[narrow["stay_id"].isin(keep)].sort_values(["stay_id", "hour"]).groupby("stay_id", sort=False):
        v = g["vr"].to_numpy(float)
        rows.append({"stay_id": sid, "vr_initial": v[0], "vr_delta": v[-1] - v[0]})
    feat = pd.DataFrame(rows)
    stamp(f"窄窗口特征: {len(feat)}人 (≥2个≤1h配对点)")

    ard = pd.read_csv(OUTDIR / "mimic_analysis_ready_round2.csv")
    cc = ard.dropna(subset=[OUTCOME] + BASE_A + PF).drop(columns=["vr_initial", "vr_delta"]).merge(
        feat, on="stay_id", how="inner")
    stamp(f"窄窗口分析队列(完整病例∩≥2窄配对): n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")

    out, ors = fit_logit(cc, VARS)
    stamp(f"[±1h] N={out['N']}, 死亡={out['Deaths']}, AUROC={out['AUROC']}, Brier={out['Brier']}")

    # 对照：round-2 完整病例联合主分析
    rd_full = ard.dropna(subset=[OUTCOME] + BASE_A + PF + ["vr_initial", "vr_delta"])
    out2, ors2 = fit_logit(rd_full, VARS)
    stamp(f"[对照·round2主分析] N={out2['N']}, 死亡={out2['Deaths']}, AUROC={out2['AUROC']}")
    for _, r in ors2.iterrows():
        stamp(f"[对照·round2主分析] {r.variable}: {r.OR_CI}, P={r.p}")

    ors.to_csv(OUTDIR / "narrow_window_OR_round2.csv", index=False)
    cc[["stay_id", "vr_initial", "vr_delta"]].to_csv(OUTDIR / "narrow_window_features_round2.csv", index=False)
    (OUTDIR / "narrow_window_round2_log.md").write_text(
        "# ±1h窄窗口敏感性日志（round-2 重建世界）\n\n"
        "- 点源=规则B截断后的重建轨迹；协变量=重建完整病例；BASE_A含COPD/CHF（18协变量）。\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp("±1h round-2 完成")


if __name__ == "__main__":
    main()