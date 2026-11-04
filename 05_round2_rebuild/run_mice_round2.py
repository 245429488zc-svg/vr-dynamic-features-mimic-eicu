#!/usr/bin/env python3
"""MICE sensitivity analysis with the OUTCOME included in the imputation model
(R1.11 / R2.8).

Replicates the published Table 4 MICE pipeline exactly
(13_Table4正式版/make_table4_sensitivity.py: same data, same 22-variable analysis
set, m=10, burn-in=5, Rubin pooling, same RNG seed) with one change: the outcome
(hospital_expire_flag) is added to the imputation model, per the reviewers' request.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUT.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "mimic_analysis_ready_round2.csv"
OUTCOME = "hospital_expire_flag"
RNG = 20260721
MICE_M = 10

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


def p_from_z(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def mice_datasets(df: pd.DataFrame, variables: list[str], m: int, burnin: int = 5) -> list[pd.DataFrame]:
    data = MICEData(df[variables].apply(pd.to_numeric, errors="coerce"))
    np.random.seed(RNG)
    for _ in range(burnin):
        data.update_all(1)
    out = []
    for _ in range(m):
        data.update_all(1)
        imp = df.copy()
        imp[variables] = data.data[variables].copy()
        out.append(imp)
    return out


def fit_mice_outcome(df: pd.DataFrame, variables: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    parts, perf_rows = [], []
    for idx, imp in enumerate(mice_datasets(df, variables + [OUTCOME], MICE_M), start=1):
        x = sm.add_constant(pd.DataFrame(
            StandardScaler().fit_transform(imp[variables]), columns=variables, index=imp.index), has_constant="add")
        y = imp[OUTCOME].astype(int)
        res = sm.Logit(y, x).fit(disp=False, maxiter=300)
        pred = res.predict(x)
        perf_rows.append({"imputation": idx, "n": int(len(imp)), "events": int(y.sum()),
                          "AUROC": float(roc_auc_score(y, pred)), "Brier": float(brier_score_loss(y, pred))})
        rows = []
        for var in variables:
            rows.append({"imputation": idx, "variable": var,
                         "coef": float(res.params[var]), "se": float(res.bse[var])})
        parts.append(pd.DataFrame(rows))

    pooled = []
    for var in variables:
        coefs = np.array([p.loc[p.variable.eq(var), "coef"].iloc[0] for p in parts])
        ses = np.array([p.loc[p.variable.eq(var), "se"].iloc[0] for p in parts])
        qbar = float(coefs.mean())
        ubar = float(np.mean(ses ** 2))
        b = float(np.var(coefs, ddof=1))
        total = ubar + (1 + 1 / MICE_M) * b
        se = math.sqrt(total)
        pooled.append({"variable": var, "OR": math.exp(qbar),
                       "CI_lower": math.exp(qbar - 1.96 * se), "CI_upper": math.exp(qbar + 1.96 * se),
                       "p_value": p_from_z(qbar / se), "between_var": b})
    perf = pd.DataFrame(perf_rows)
    initial = next(r for r in pooled if r["variable"] == "vr_initial")
    delta = next(r for r in pooled if r["variable"] == "vr_delta")
    out = {"N": int(len(df)), "Deaths": int(df[OUTCOME].sum()),
           "AUROC": f"{perf.AUROC.mean():.3f}", "Brier": f"{perf.Brier.mean():.3f}"}
    return out, pd.DataFrame(pooled), perf, initial, delta


def fmt(o, lo, hi):
    return f"{o:.3f} ({lo:.3f}-{hi:.3f})"


def main() -> None:
    df = pd.read_csv(DATA)
    df[OUTCOME] = df[OUTCOME].astype(int)
    stamp(f"数据: n={len(df)}, 死亡={int(df[OUTCOME].sum())}")
    stamp(f"插补模型: {len(FULL_VARS) + 1}个变量（22个分析变量 + 结局hospital_expire_flag）, m={MICE_M}, burn-in=5, Rubin合并, seed={RNG}")

    out, pooled, perf, initial, delta = fit_mice_outcome(df, FULL_VARS)
    stamp(f"[含结局MICE] N={out['N']}, 死亡={out['Deaths']}, AUROC={out['AUROC']}, Brier={out['Brier']}")
    stamp(f"[含结局MICE] VR initial: {fmt(initial['OR'], initial['CI_lower'], initial['CI_upper'])}, P={initial['p_value']:.3g}")
    stamp(f"[含结局MICE] VR delta:   {fmt(delta['OR'], delta['CI_lower'], delta['CI_upper'])}, P={delta['p_value']:.3g}")

    perf.to_csv(OUT / "mice_outcome_iteration_performance.csv", index=False)
    pooled.to_csv(OUT / "mice_outcome_pooled_OR.csv", index=False)
    (OUT / "mice_outcome_log.md").write_text(
        "# 含结局MICE日志（R1.11/R2.8）\n\n"
        "- 与已发表Table 4 MICE管道完全一致（数据/22变量分析集/m=10/burn-in=5/Rubin合并/seed=20260721），唯一改动：结局变量纳入插补模型。\n"
        "- 已发表对照（不含结局）：VR initial 1.748 (1.619-1.888), VR delta 1.409 (1.319-1.506), AUROC 0.843, Brier 0.107。\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp(f"已保存: {OUT}")


if __name__ == "__main__":
    main()