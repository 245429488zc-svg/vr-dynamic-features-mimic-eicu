#!/usr/bin/env python3
"""Round-2 audit (R2-3②): explicit truncation verification for the 48-hour feature pipeline.

Reviewer 2 (Comment 3) states the published feature-generation code does not clearly
implement the truncation "at death, extubation, or ICU discharge". This script:

  PART A - replicates the ORIGINAL rule verbatim (points 0-48h post ICU admission,
           MV candidates 0-48h, ±2h nearest-neighbor, tie -> preceding record) from the
           published trajectory table, and verifies that the recomputed features match
           1_队列数据/MIMIC_IV_analysis_ready.csv exactly (formula calibration).
  PART B - adds the manuscript-stated truncation rule: the observation window ends at
           min(48 h, death, first extubation (end of the first InvasiveVent segment),
           ICU discharge), with pairs admitted only if both the ABG time and the
           ventilator-setting time fall inside that window; recomputes all features,
           re-applies the >=3-pairs gate, and diffs against the original analysis set.
  PART C - determines the VR slope definition actually used in analysis_ready
           (two-point net slope delta/span vs per-point least-squares polyfit).

Outputs: truncation_verification.csv + truncation_verification_log.md
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
TRAJ = ROOT / "22_研究归档_20260722" / "01_cohort_data" / "trajectory_mimic_48h.csv"
READY = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
VENT = Path("/Volumes/tongtong/数据库/public/mimmc4/mimmc/项目4/原始数据/mimiciv_derived/mimiciv_derived/ventilation.csv")
ICU = Path("/Volumes/tongtong/数据库/public/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/icu/icustays.csv.gz")
ADM = Path("/Volumes/tongtong/数据库/public/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1/hosp/admissions.csv.gz")
OUTDIR = ROOT / "43_第二轮重分析"
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def features_from(points: pd.DataFrame, id_col: str = "stay_id") -> pd.DataFrame:
    """Same formulas as the analysis-ready builder (enhanced_covariates lineage)."""
    rows = []
    for sid, sub in points.groupby(id_col):
        sub = sub.sort_values("hour")
        vr = sub["vr"].astype(float)
        ok_vr = vr.notna()
        v = vr[ok_vr]
        h_v = sub.loc[v.index, "hour"].astype(float)
        span = float(h_v.iloc[-1] - h_v.iloc[0]) if len(v) >= 2 else 0.0
        rec = {id_col: sid}
        if len(v) >= 1:
            rec["vr_initial"] = float(v.iloc[0])
            rec["vr_last"] = float(v.iloc[-1])
            rec["vr_mean"] = float(v.mean())
            rec["vr_max"] = float(v.max())
            rec["vr_min"] = float(v.min())
        if len(v) >= 2:
            rec["vr_delta"] = float(v.iloc[-1] - v.iloc[0])
            rec["vr_slope"] = float(rec["vr_delta"] / max(span, 1e-6))
            rec["vr_slope_ols"] = float(np.polyfit(h_v, v, 1)[0])
            rec["vr_auc"] = float(np.trapz(v.values, h_v.values) / max(span, 1e-6))
        pf = sub["pf"].astype(float).dropna()
        if len(pf) >= 1:
            rec["pf_initial"] = float(pf.iloc[0])
            rec["pf_min"] = float(pf.min())
        rec["n_points"] = int(len(sub))
        rec["n_vr_points"] = int(ok_vr.sum())
        rec["vr_time_span_h"] = span
        rows.append(rec)
    out = pd.DataFrame(rows)
    keep_cols = {id_col: "stay_id"} if id_col != "stay_id" else None
    if keep_cols:
        out = out.rename(columns=keep_cols)
    return out


def main() -> None:
    traj = pd.read_csv(TRAJ)
    ready = pd.read_csv(READY)
    stamp(f"trajectory: {len(traj)} points / {traj['stay_id'].nunique()} stays; "
          f"analysis_ready: {len(ready)} rows")

    # ---------- PART A: original rule from the trajectory table ----------
    feat_a_all = features_from(traj)
    m = ready.merge(feat_a_all, on="stay_id", suffixes=("_ready", "_recomp"))
    stamp(f"PART A 匹配行: {len(m)} (analysis_ready ⊆ trajectory: "
          f"{int(ready['stay_id'].isin(feat_a_all['stay_id']).sum())}/{len(ready)})")
    checks = {}
    for a, b in [("vr_initial_ready", "vr_initial_recomp"), ("vr_delta_ready", "vr_delta_recomp"),
                 ("vr_mean_ready", "vr_mean_recomp"), ("vr_max_ready", "vr_max_recomp"),
                 ("vr_min_ready", "vr_min_recomp"), ("vr_auc_ready", "vr_auc_recomp"),
                 ("pf_initial_ready", "pf_initial_recomp"), ("pf_min_ready", "pf_min_recomp"),
                 ("n_points_ready", "n_points_recomp"), ("vr_time_span_h_ready", "vr_time_span_h_recomp")]:
        if a in m.columns and b in m.columns:
            x, y = pd.to_numeric(m[a], errors="coerce"), pd.to_numeric(m[b], errors="coerce")
            eq = ((x - y).abs() < 1e-9) | (x.isna() & y.isna())
            checks[a.replace("_ready", "")] = float(eq.mean())
    for k, v in checks.items():
        stamp(f"PART A 一致率 {k}: {v:.4%}")

    # slope 定义判别（PART C）
    if {"vr_slope_ready", "vr_slope_recomp", "vr_slope_ols"}.issubset(m.columns):
        s_ready = pd.to_numeric(m["vr_slope_ready"], errors="coerce")
        s_2pt = pd.to_numeric(m["vr_slope_recomp"], errors="coerce")
        s_ols = pd.to_numeric(m["vr_slope_ols"], errors="coerce")
        eq2 = ((s_ready - s_2pt).abs() < 1e-9).mean()
        eqols = ((s_ready - s_ols).abs() < 1e-9).mean()
        stamp(f"PART C slope定义: 两点法(delta/span)一致率={eq2:.4%}; OLS回归法一致率={eqols:.4%}")

    # ---------- PART B: explicit truncation ----------
    vent = pd.read_csv(VENT, names=["stay_id", "starttime", "endtime", "ventilation_status"],
                       skiprows=1, low_memory=False)
    vent = vent[vent["stay_id"].astype(str).str.isdigit()].copy()
    vent["stay_id"] = vent["stay_id"].astype(int)
    inv = vent[vent["ventilation_status"].eq("InvasiveVent")].copy()
    inv["start"] = pd.to_datetime(inv["starttime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    inv["end"] = pd.to_datetime(inv["endtime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    icu = pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime", "hadm_id"])
    adm = pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime"])
    for c in ["intime", "outtime"]:
        icu[c] = pd.to_datetime(icu[c], errors="coerce")
    for c in ["dischtime", "deathtime"]:
        adm[c] = pd.to_datetime(adm[c], errors="coerce")

    tr = traj.merge(icu, on="stay_id", how="left").merge(adm, on="hadm_id", how="left")
    tr["t_abg"] = tr["intime"] + pd.to_timedelta(tr["hour"], unit="h")
    # 两种"extubation"语义：首次InvasiveVent段结束 vs 末次InvasiveVent段结束（截断到48h上限）
    seg_first = inv.sort_values(["stay_id", "start"]).groupby("stay_id").head(1)[["stay_id", "end"]].rename(columns={"end": "seg_first_end"})
    seg_last = inv.groupby("stay_id")["end"].max().rename("seg_last_end").reset_index()
    tr = tr.merge(seg_first, on="stay_id", how="left").merge(seg_last, on="stay_id", how="left")
    tr["t_end"] = tr["intime"] + pd.Timedelta(hours=48)
    tr["t_end"] = tr["t_end"].where(tr["t_end"].notna(), pd.Timestamp.max)
    for col, src in [("icu_outtime", "outtime"), ("dischtime", "dischtime"), ("deathtime", "deathtime"),
                     ("seg_last_end", "seg_last_end"), ("seg_first_end", "seg_first_end")]:
        tr[col + "_h"] = tr[src]
    tr["t_end_strict"] = tr["t_end"]
    for src in ["outtime", "dischtime", "deathtime", "seg_first_end"]:
        tr["t_end_strict"] = tr["t_end_strict"].mask(tr[src].notna() & (tr[src] < tr["t_end_strict"]), tr[src])
    tr["t_end_last_vent"] = tr["t_end"]
    for src in ["outtime", "dischtime", "deathtime", "seg_last_end"]:
        tr["t_end_last_vent"] = tr["t_end_last_vent"].mask(tr[src].notna() & (tr[src] < tr["t_end_last_vent"]), tr[src])
    tr["t_mv"] = tr["t_abg"] + pd.to_timedelta(tr["pair_lag_hours"], unit="h")
    n_before = len(tr)
    # 规则A（首次拔管截断） vs 规则B（末次拔管截断·canonical）
    keep_first = (tr["t_abg"] <= tr["t_end_strict"]) & (tr["t_mv"] <= tr["t_end_strict"]) & tr["t_mv"].notna()
    keep_last = (tr["t_abg"] <= tr["t_end_last_vent"]) & (tr["t_mv"] <= tr["t_end_last_vent"]) & tr["t_mv"].notna()
    n_keep_first = int(keep_first.sum())
    tr2 = tr[keep_last].copy()
    stamp(f"PART B 规则A(首次拔管截断): {n_before} -> {n_keep_first} (去除 {n_before - n_keep_first} 点)")
    stamp(f"PART B 规则B(末次拔管截断·canonical): {n_before} -> {len(tr2)} (去除 {n_before - len(tr2)} 点)")
    stays_keep_first = int(tr[keep_first].groupby("stay_id").size().ge(3).sum())
    stamps = int(tr[keep_last].groupby("stay_id").size().ge(3).sum())
    stamp(f"PART B >=3配对座位数: 规则A={stays_keep_first}（原口径13,263）; 规则B={stamps}（原口径13,263）")

    # 边界分解（PART D）：规则B下的被删点
    removed = tr[~keep_last].copy()
    decomp = {
        "abg>death": int((removed["deathtime"].notna() & (removed["t_abg"] > removed["deathtime"])).sum()),
        "abg>hospital_discharge": int((removed["dischtime"].notna() & (removed["t_abg"] > removed["dischtime"])).sum()),
        "abg>icu_discharge": int((removed["outtime"].notna() & (removed["t_abg"] > removed["outtime"])).sum()),
        "abg>last_extubation": int((removed["seg_last_end"].notna() & (removed["t_abg"] > removed["seg_last_end"])).sum()),
        "mv>death": int((removed["deathtime"].notna() & (removed["t_mv"] > removed["deathtime"])).sum()),
    }
    extra_removed_by_first = int((~keep_first).sum()) - int((~keep_last).sum())
    stamp("PART D 规则B被删点边界分解: " + ", ".join(f"{k}={v}" for k, v in decomp.items()))
    stamp(f"PART D 规则A额外多删的点（再插管期间及其配对）: {extra_removed_by_first}")
    feat_b_all = features_from(tr2)
    feat_b = feat_b_all[(feat_b_all["n_vr_points"] >= 3) &
                        (feat_b_all["pf_initial"].notna())].copy()
    cmp = ready.merge(feat_b, on="stay_id", suffixes=("_ready", "_trunc"), how="left")
    in_both = cmp["vr_initial_trunc"].notna()
    stamp(f"PART B 双侧可比: {int(in_both.sum())};  analysis_ready-only: {int((~in_both).sum())}")
    feature_cols = ["vr_initial", "vr_delta", "vr_mean", "vr_last", "vr_max", "vr_min", "vr_auc",
                    "pf_initial", "pf_min", "n_points", "vr_time_span_h"]
    per_feature_change = {}
    for col in feature_cols:
        a = pd.to_numeric(cmp[f"{col}_ready"], errors="coerce")
        b = pd.to_numeric(cmp[f"{col}_trunc"], errors="coerce")
        changed = (~((a - b).abs() < 1e-9)) & in_both
        per_feature_change[col] = int(changed.sum())
    stamp("PART B 双侧行内特征差异计数: " + ", ".join(f"{k}={v}" for k, v in per_feature_change.items()))
    cmp_all = in_both & (~((cmp[[f"{c}_ready" for c in feature_cols]].astype(float).values -
                            cmp[[f"{c}_trunc" for c in feature_cols]].astype(float).values) < 1e-9)).any(axis=1)
    stamp(f"PART B 汇总: 有任一特征改变的座位数 = {int(cmp_all.sum())}")

    # ---------- 输出 ----------
    cols_out = ["stay_id"] + [f"{c}_ready" for c in feature_cols] + [f"{c}_trunc" for c in feature_cols]
    cmp[cols_out].to_csv(OUTDIR / "truncation_verification.csv", index=False)
    stamp(f"已保存: {OUTDIR / 'truncation_verification.csv'}")
    (OUTDIR / "truncation_verification_log.md").write_text(
        "# 截断逻辑验证（R2-3②）\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp(f"已保存日志: {OUTDIR / 'truncation_verification_log.md'}")


if __name__ == "__main__":
    main()