#!/usr/bin/env python3
"""Round-2 rebuild (R2-3② + R2-1), MIMIC side.

Re-derives ALL trajectory features (vr_*, pf_*, paco2_initial, ph_initial, n_points,
vr_time_span_h) under the manuscript-stated truncation rule, now explicit:

    window_end(stay) = min(ICU admit + 48 h,
                           hospital death time,
                           hospital discharge time,
                           ICU discharge time,
                           last InvasiveVent segment end [terminal extubation])

A pair (ABG + nearest ventilator setting) is retained only if BOTH timestamps lie in
[ICU admit, window_end]. Cohort gate unchanged: >=3 valid paired P/F-VR points (both
pf_points and vr_points >= 3, identical to the original builder).

Covariates are carried over UNCHANGED from the original analysis_ready (their
definitions are independent of pairing/truncation). Outcomes are re-jointed from
OFFICIAL MIMIC-IV tables (admissions.hospital_expire_flag) with a consistency check
against the legacy flag.

Output: mimic_analysis_ready_round2.csv (+ truncation log, cohort summary log).
"""
from __future__ import annotations

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
OUTDIR.mkdir(exist_ok=True)
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def features_explicit(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in points.groupby("stay_id"):
        sub = sub.sort_values("hour")
        vr = sub["vr"].astype(float).dropna()
        h_v = sub.loc[vr.index, "hour"].astype(float)
        span = float(h_v.iloc[-1] - h_v.iloc[0]) if len(vr) >= 2 else 0.0
        rec = {"stay_id": int(sid), "hadm_id": int(sub["hadm_id"].iloc[0])}
        if len(vr) >= 1:
            rec["vr_initial"] = float(vr.iloc[0])
            rec["vr_last"] = float(vr.iloc[-1])
            rec["vr_mean"] = float(vr.mean())
            rec["vr_max"] = float(vr.max())
            rec["vr_min"] = float(vr.min())
        if len(vr) >= 2:
            rec["vr_delta"] = float(vr.iloc[-1] - vr.iloc[0])
            rec["vr_slope"] = float(rec["vr_delta"] / max(span, 1e-6))
            rec["vr_slope_ols"] = float(np.polyfit(h_v, vr, 1)[0])
            rec["vr_auc"] = float(np.trapz(vr.values, h_v.values) / max(span, 1e-6))
        pf = sub["pf"].astype(float).dropna()
        if len(pf) >= 1:
            rec["pf_initial"] = float(pf.iloc[0])
            rec["pf_min"] = float(pf.min())
            rec["pf_last"] = float(pf.iloc[-1])
            rec["pf_mean"] = float(pf.mean())
        if len(pf) >= 2:
            rec["pf_delta"] = float(pf.iloc[-1] - pf.iloc[0])
        pc = sub["pco2"].astype(float).dropna()
        ph = sub["ph"].astype(float).dropna()
        if len(pc) >= 1:
            rec["paco2_initial"] = float(pc.iloc[0])
        if len(ph) >= 1:
            rec["ph_initial"] = float(ph.iloc[0])
        rec["n_points"] = int(len(sub))
        rec["vr_points"] = int(len(vr))
        rec["pf_points"] = int(len(pf))
        rec["vr_time_span_h"] = span
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    traj = pd.read_csv(TRAJ)
    ready = pd.read_csv(READY)
    stamp(f"输入: trajectory {len(traj)}点/{traj['stay_id'].nunique()}座; analysis_ready {len(ready)}行")

    # ---- window construction ----
    vent = pd.read_csv(VENT, names=["stay_id", "starttime", "endtime", "ventilation_status"],
                       skiprows=1, low_memory=False)
    vent = vent[vent["stay_id"].astype(str).str.isdigit()].copy()
    vent["stay_id"] = vent["stay_id"].astype(int)
    inv = vent[vent["ventilation_status"].eq("InvasiveVent")].copy()
    inv["start"] = pd.to_datetime(inv["starttime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    inv["end"] = pd.to_datetime(inv["endtime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    seg_last = inv.groupby("stay_id")["end"].max().rename("seg_last_end").reset_index()

    icu = pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime", "hadm_id"])
    adm = pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"])
    for c in ["intime", "outtime"]:
        icu[c] = pd.to_datetime(icu[c], errors="coerce")
    for c in ["dischtime", "deathtime"]:
        adm[c] = pd.to_datetime(adm[c], errors="coerce")

    tr = traj.merge(icu, on="stay_id", how="left").merge(adm, on="hadm_id", how="left") \
             .merge(seg_last, on="stay_id", how="left")
    tr["t_abg"] = tr["intime"] + pd.to_timedelta(tr["hour"], unit="h")
    tr["t_mv"] = tr["t_abg"] + pd.to_timedelta(tr["pair_lag_hours"], unit="h")
    t_end = tr["intime"] + pd.Timedelta(hours=48)
    for src in ["outtime", "dischtime", "deathtime", "seg_last_end"]:
        t_end = t_end.mask(tr[src].notna() & (tr[src] < t_end), tr[src])
    tr["t_end"] = t_end
    keep = tr["t_mv"].notna() & (tr["t_abg"] <= tr["t_end"]) & (tr["t_mv"] <= tr["t_end"])
    stamp(f"点过滤: {len(tr)} -> {int(keep.sum())} (删除 {int((~keep).sum())})")
    tr[keep].to_csv(OUTDIR / "trajectory_mimic_48h_round2.csv", index=False)
    stamp(f"已保存点级表(规则B): trajectory_mimic_48h_round2.csv")

    # ---- features under explicit truncation ----
    feat = features_explicit(tr[keep])
    feat = feat[(feat["vr_points"] >= 3) & (feat["pf_points"] >= 3)].copy()
    stamp(f"显式截断后 >=3配对座位(资格集): {len(feat)}（原 13,263）")

    # ---- outcome from OFFICIAL tables ----
    feat = feat.merge(adm[["hadm_id", "hospital_expire_flag"]].rename(
        columns={"hospital_expire_flag": "hexp_official"}), on="hadm_id", how="left")
    # official outcome 也需逐 stay 对齐（hadm 可重复入院——取该 stay 所属 hadm；首stay每hadm唯一）
    legacy = traj.groupby("stay_id")["hospital_expire_flag"].max()
    feat["hexp_legacy"] = feat["stay_id"].map(legacy)
    def row_mism(r) -> bool:
        if pd.isna(r["hexp_official"]) or pd.isna(r["hexp_legacy"]):
            return True
        return int(r["hexp_official"]) != int(r["hexp_legacy"])

    mism = feat.apply(row_mism, axis=1).sum()
    stamp(f"官方结局缺失/不一致: {int(mism)}（缺失即本轮排除口径不变：无官方结局的座位排除）")
    feat = feat[feat["hexp_official"].notna()].copy()
    feat["hospital_expire_flag"] = feat["hexp_official"].astype(int)
    assoc = feat.dropna(subset=["hospital_expire_flag"])
    stamp(f"新关联队列(有官方结局+资格): {len(assoc)}, 死亡={int(assoc['hospital_expire_flag'].sum())} "
          f"({assoc['hospital_expire_flag'].mean():.1%})；对比旧 13,145/2,579(19.6%)")

    # ---- assemble new analysis_ready ----
    feat_cols_new = [c for c in feat.columns if c != "hospital_expire_flag"] + ["hospital_expire_flag"]
    new_feat = feat[feat_cols_new].copy()
    drop_from_old = ["n_points", "vr_initial", "vr_max", "vr_mean", "vr_last", "vr_min", "vr_auc",
                     "vr_delta", "vr_slope", "vr_time_span_h", "pf_initial", "pf_min", "pf_mean",
                     "pf_last", "pf_delta", "pf_slope", "paco2_initial", "ph_initial",
                     "hospital_expire_flag", "hadm_id", "subject_id"]
    old_rest = ready.drop(columns=[c for c in drop_from_old if c in ready.columns])
    new_ready = new_feat.merge(old_rest, on="stay_id", how="left")
    new_ready.to_csv(OUTDIR / "mimic_analysis_ready_round2.csv", index=False)
    stamp(f"已保存: mimic_analysis_ready_round2.csv ({len(new_ready)}行×{new_ready.shape[1]}列)")

    # ---- landmark cohort (prediction framework) ----
    base_cols = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
                 "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
                 "bun", "creatinine", "wbc", "bicarbonate"]
    pf_cols, vr_cols = ["pf_initial", "pf_min"], ["vr_initial", "vr_delta"]
    full_vars = base_cols + pf_cols + vr_cols
    nr = new_ready.copy()
    nr["hospital_expire_flag"] = nr["hospital_expire_flag"].astype(int)
    cc = nr.dropna(subset=["hospital_expire_flag"] + full_vars).copy()
    stamp(f"新完整病例(cc): {len(cc)}, 死亡={int(cc['hospital_expire_flag'].sum())}")
    cc = cc.merge(icu[["stay_id", "intime", "outtime"]], on="stay_id", how="left", suffixes=("", "_icu"))
    cc = cc.merge(adm[["hadm_id", "dischtime", "deathtime"]], on="hadm_id", how="left")
    hour48 = cc["intime"] + pd.Timedelta(hours=48)
    died = cc["hospital_expire_flag"] == 1
    win_death = died & (
        (cc["deathtime"].notna() & (cc["deathtime"] <= hour48))
        | (cc["deathtime"].isna() & cc["dischtime"].notna() & (cc["dischtime"] <= hour48)))
    early_dis = (~died) & cc["dischtime"].notna() & (cc["dischtime"] <= hour48)
    lm = cc.loc[~win_death].copy()
    lm = lm.loc[~((lm["hospital_expire_flag"] == 0) & lm["dischtime"].notna() &
                  (lm["dischtime"] <= hour48[lm.index]))].copy()
    stamp(f"新landmark队列: n={len(lm)}, 48h后死亡={int(lm['hospital_expire_flag'].sum())} "
          f"（旧框架: 10,263/1,725）")
    lm[["stay_id", "hospital_expire_flag"] + full_vars].to_csv(OUTDIR / "landmark_cohort_round2.csv", index=False)
    assoc[["stay_id", "hospital_expire_flag"]].to_csv(OUTDIR / "association_cohort_round2_ids.csv", index=False)

    (OUTDIR / "mimic_rebuild_log.md").write_text(
        "# MIMIC 规则B特征重建日志（R2-3②/R2-1）\n\n"
        "- 截断规则：min(入科+48h, 死亡时间, 住院出院, ICU出科, 末次有创通气段结束)。\n"
        "- 配对保留条件：ABG与MV时刻均在窗内；资格门：pf_points>=3 且 vr_points>=3。\n"
        "- 结局：官方 admissions.hospital_expire_flag 重挂。\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("完成")


if __name__ == "__main__":
    main()