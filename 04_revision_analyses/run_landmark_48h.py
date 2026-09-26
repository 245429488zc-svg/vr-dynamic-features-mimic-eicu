#!/usr/bin/env python3
"""48-hour landmark analysis for R1.1 / R2.2 (YRMED-D-26-02453 revision).

Replaces the proxy-based "Excluding deaths within 48 h" row (ICU LOS <= 2 d)
with an exact-timestamp landmark cohort on the complete-case base:
  - keep patients alive at hour 48 after ICU admission (deathtime > intime + 48 h)
  - keep patients still hospitalized at hour 48 (dischtime > intime + 48 h)
  - outcome: hospital_expire_flag (all remaining deaths occur after the landmark)
Model spec identical to the published Table 4 complete-case primary analysis.
Timestamps sourced from the official MIMIC-IV v3.1 hospital/ICU core tables.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
OUTDIR = ROOT / "31_48h_landmark分析"
OUTDIR.mkdir(exist_ok=True)
DATA = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
ADM = Path("./raw_data/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/hosp/admissions.csv.gz")
ICU = Path("./raw_data/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/icu/icustays.csv.gz")

OUTCOME = "hospital_expire_flag"
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


def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.3f} ({lo:.3f}-{hi:.3f})"


def scale_x(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    scaler = StandardScaler().fit(df[variables])
    return pd.DataFrame(scaler.transform(df[variables]), columns=variables, index=df.index)


def fit_logit(cc: pd.DataFrame, variables: list[str]) -> tuple[dict, pd.DataFrame]:
    x = sm.add_constant(scale_x(cc, variables), has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)
    rows = []
    for var in variables:
        beta = float(res.params[var])
        se = float(res.bse[var])
        z = beta / se
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
        rows.append({
            "variable": var,
            "OR": math.exp(beta),
            "CI_lower": math.exp(beta - 1.96 * se),
            "CI_upper": math.exp(beta + 1.96 * se),
            "p_value": p,
        })
    out = {
        "N": int(len(cc)),
        "Deaths": int(y.sum()),
        "AUROC": f"{roc_auc_score(y, pred):.3f}",
        "Brier": f"{brier_score_loss(y, pred):.3f}",
    }
    return out, pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA)
    df[OUTCOME] = df[OUTCOME].astype(int)
    stamp(f"分析集读入: n={len(df)}, 死亡={int(df[OUTCOME].sum())}")

    icu = pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime"])
    adm = pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"])
    for col in ["intime", "outtime"]:
        icu[col] = pd.to_datetime(icu[col], errors="coerce")
    for col in ["dischtime", "deathtime"]:
        adm[col] = pd.to_datetime(adm[col], errors="coerce")

    df = df.merge(icu, on="stay_id", how="left")
    df = df.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    stamp(f"时间戳合并: intime缺失={int(df['intime'].isna().sum())}, "
          f"dischtime缺失={int(df['dischtime'].isna().sum())}, "
          f"deathtime缺失={int(df['deathtime'].isna().sum())}")

    # 一致性核查：官方表死亡标志 vs 分析集死亡标志
    flag_mismatch = int((df["hospital_expire_flag"].astype(int) != df["hospital_expire_flag_adm"].astype(int)).sum())
    stamp(f"hospital_expire_flag 与官方 admissions 不一致条数: {flag_mismatch}")

    hour48 = df["intime"] + pd.Timedelta(hours=48)
    died = df[OUTCOME] == 1
    # 窗口内死亡：deathtime 在窗内；deathtime 缺失者用 dischtime 兜底（住院死亡以出院即死亡为准）
    win_death = died & (
        (df["deathtime"].notna() & (df["deathtime"] <= hour48))
        | (df["deathtime"].isna() & df["dischtime"].notna() & (df["dischtime"] <= hour48))
    )
    died_no_time = died & df["deathtime"].isna() & df["dischtime"].isna()
    stamp(f"全队列(13,145) 48h内死亡(精确时间戳) = {int(win_death.sum())}; "
          f"死亡且 deathtime/dischtime 均缺失 = {int(died_no_time.sum())}（按存活至48h处理并在正文如实说明，如非零）")

    early_discharge = (~died) & df["dischtime"].notna() & (df["dischtime"] <= hour48)
    stamp(f"全队列 存活且48h内出院(非事件、不再处于风险中) = {int(early_discharge.sum())}")

    cc = df.dropna(subset=[OUTCOME] + FULL_VARS).copy()
    stamp(f"完整病例基线: n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")

    primary_out, primary_or = fit_logit(cc, FULL_VARS)
    stamp(f"[对照] primary_cc 重现: n={primary_out['N']}, 死亡={primary_out['Deaths']}, "
          f"AUROC={primary_out['AUROC']}, Brier={primary_out['Brier']}")
    pv = primary_out and pd.concat([
        primary_or.loc[primary_or.variable.eq("vr_initial")],
        primary_or.loc[primary_or.variable.eq("vr_delta")],
    ])
    for _, r in pv.iterrows():
        stamp(f"[对照] {r.variable}: {fmt_or(r.OR, r.CI_lower, r.CI_upper)}, P={r.p_value:.3g}")

    lm = cc.loc[~win_death[cc.index]].copy()
    removed_death = len(cc) - len(lm)
    before_ed = len(lm)
    lm = lm.loc[~((lm[OUTCOME] == 0) & lm["dischtime"].notna() & (lm["dischtime"] <= hour48[lm.index]))]
    removed_ed = before_ed - len(lm)
    stamp(f"landmark 过滤: 剔除窗口内死亡 {removed_death}, 剔除48h内出院存活 {removed_ed}, "
          f"剩余 n={len(lm)}, 死亡={int(lm[OUTCOME].sum())}")

    lm_out, lm_or = fit_logit(lm, FULL_VARS)
    stamp(f"[landmark] n={lm_out['N']}, 死亡={lm_out['Deaths']}, "
          f"AUROC={lm_out['AUROC']}, Brier={lm_out['Brier']}")
    vr = pd.concat([lm_or.loc[lm_or.variable.eq("vr_initial")],
                    lm_or.loc[lm_or.variable.eq("vr_delta")]])
    for _, r in vr.iterrows():
        stamp(f"[landmark] {r.variable}: {fmt_or(r.OR, r.CI_lower, r.CI_upper)}, P={r.p_value:.3g}")

    summary = pd.DataFrame([
        {"analysis": "primary_cc", **primary_out},
        {"analysis": "landmark_48h_exact", **lm_out},
    ])
    or_all = pd.concat([
        primary_or.assign(analysis="primary_cc"),
        lm_or.assign(analysis="landmark_48h"),
    ], ignore_index=True)
    summary.to_csv(OUTDIR / "landmark_48h_summary.csv", index=False)
    or_all.to_csv(OUTDIR / "landmark_48h_OR_details.csv", index=False)
    (OUTDIR / "landmark_48h_generation_log.md").write_text(
        "# 48h landmark 分析日志（R1.1 / R2.2）\n\n"
        "- 基准：完整病例队列（与最终 Table 4 敏感性分析同基线）。\n"
        "- 时间戳来源：MIMIC-IV v3.1 官方 hosp/admissions.csv（dischtime/deathtime）与 icu/icustays.csv（intime/outtime）。\n"
        "- 定义：48h 时点存活（deathtime > intime+48h）且仍在院（dischtime > intime+48h）；结局为 48h 后院内死亡。\n"
        "- 模型与主分析完全一致（BASE_A + pf_initial/pf_min + vr_initial/vr_delta，队列内标准化，Logit，maxiter=300）。\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp(f"输出已保存到 {OUTDIR}")


if __name__ == "__main__":
    main()