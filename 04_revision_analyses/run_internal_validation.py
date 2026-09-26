#!/usr/bin/env python3
"""Internal validation via 5-fold stratified CV (R2.5): optimism-corrected
discrimination and out-of-fold calibration (intercept/slope) for Base+P/F and
Base+P/F+VR (complete-case, same spec as published Table 3 internal model).
Replicates the original CV setup (StratifiedKFold(5, shuffle=True, seed=20260721))."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
OUT = ROOT / "39_内部验证"
OUT.mkdir(exist_ok=True)
OUTCOME = "hospital_expire_flag"
RNG = 20260721
BASE_A = ["age", "sex", "charlson", "sepsis3", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def oof_model(df: pd.DataFrame, vars_: list[str]) -> dict:
    y = df[OUTCOME].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    oof = np.zeros(len(df))
    for tr, te in skf.split(df[vars_], y):
        scaler = StandardScaler().fit(df.iloc[tr][vars_])
        m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
        m.fit(scaler.transform(df.iloc[tr][vars_]), y[tr])
        oof[te] = m.predict_proba(scaler.transform(df.iloc[te][vars_]))[:, 1]
    # apparent (训练集全量) 与 OOF 指标
    scaler_full = StandardScaler().fit(df[vars_])
    m_full = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m_full.fit(scaler_full.transform(df[vars_]), y)
    p_app = m_full.predict_proba(scaler_full.transform(df[vars_]))[:, 1]
    auc_app, auc_oof = roc_auc_score(y, p_app), roc_auc_score(y, oof)
    brier_app, brier_oof = brier_score_loss(y, p_app), brier_score_loss(y, oof)
    # 折外校准：logit(p_oof) 对 y 的 logistic 回归 → 截距=校准截距, 系数=校准斜率
    logit = np.log(np.clip(oof, 1e-6, 1 - 1e-6) / (1 - np.clip(oof, 1e-6, 1 - 1e-6)))
    cal = sm_logit(y, logit)
    return {"auc_app": auc_app, "auc_oof": auc_oof, "optimism": auc_app - auc_oof,
            "brier_app": brier_app, "brier_oof": brier_oof,
            "cal_int_app": 0.0, "cal_slope_app": 1.0,
            "cal_int_oof": cal[0], "cal_slope_oof": cal[1]}


def sm_logit(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm
    res = sm.Logit(y, sm.add_constant(x)).fit(disp=False, maxiter=300)
    return float(res.params[0]), float(res.params[1])


def main() -> None:
    df = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv")
    cc = df.dropna(subset=[OUTCOME] + BASE_A + PF + VR).copy()
    stamp(f"完整病例: n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")
    rows = []
    for label, vars_ in [("Base + P/F", BASE_A + PF), ("Base + P/F + VR", BASE_A + PF + VR)]:
        r = oof_model(cc, vars_)
        rows.append({"Model": label, "N": len(cc), "Deaths": int(cc[OUTCOME].sum()),
                     "AUROC apparent": f"{r['auc_app']:.3f}", "AUROC cross-validated": f"{r['auc_oof']:.3f}",
                     "Optimism": f"{r['optimism']:.4f}",
                     "Brier apparent": f"{r['brier_app']:.3f}", "Brier cross-validated": f"{r['brier_oof']:.3f}",
                     "Calibration intercept (OOF)": f"{r['cal_int_oof']:.3f}",
                     "Calibration slope (OOF)": f"{r['cal_slope_oof']:.3f}"})
        stamp(f"{label}: apparent AUROC={r['auc_app']:.4f}, CV AUROC={r['auc_oof']:.4f}, "
              f"optimism={r['optimism']:.4f}; OOF校准截距={r['cal_int_oof']:.3f}, 斜率={r['cal_slope_oof']:.3f}; "
              f"Brier {r['brier_app']:.3f}->{r['brier_oof']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "internal_validation_cv.csv", index=False)
    stamp(f"已保存: {OUT / 'internal_validation_cv.csv'}")
    (OUT / "internal_validation_log.md").write_text(
        "# 内部验证日志（R2.5）\n\n"
        "- 口径：与已发表Table 3内部模型一致（BASE_A 16协变量+pf_initial/pf_min+vr_initial/vr_delta，完整病例n=10,791）。\n"
        "- 方法：5折分层交叉验证（shuffle, seed=20260721，与原CV一致）；折外预测汇总后计算AUROC与Brier；"
        "折外校准截距/斜率由 logit(p_oof) 对结局的logistic回归得到。\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    import statsmodels.api as sm  # noqa
    main()