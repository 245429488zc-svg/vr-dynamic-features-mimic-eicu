#!/usr/bin/env python3
"""Build analysis-ready data for the ICU early P/F + VR dynamics study.

Phase 1 data construction only. No modeling.
Does not modify raw data. All outputs go under outputs/pf_vr_dynamic_analysis/.

Key changes vs feasibility script:
  - P/F and VR pairing uses nearest-neighbor within +/-2h (not same-hour).
  - Each blood-gas point keeps its own timestamp; minute ventilation is matched
    by the closest preceding or nearest within window record.
  - Produces analysis-ready patient-level and point-level datasets, variable
    dictionary, missingness summary, point-count summary, cohort flow, and a
    run log.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "原始数据"
OUT = ROOT / "outputs" / "pf_vr_dynamic_analysis"

PAIR_WINDOW_H = 2.0  # ±2 hours nearest-neighbor pairing
PF_MIN, PF_MAX = 30, 700
VR_MIN, VR_MAX = 0.2, 10.0
HEIGHT_MIN, HEIGHT_MAX = 120, 230
MV_MIN, MV_MAX = 1, 30  # minute volume L/min
PCO2_MIN, PCO2_MAX = 10, 150
FIO2_MIN, FIO2_MAX = 0.21, 1.0
AGE_MIN = 18


# ---------------------------------------------------------------- helpers
def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def norm_fio2(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return pd.Series(np.where(y > 1.0, y / 100.0, y), index=y.index)


def pbw_from_height_sex(height_cm: pd.Series, sex: pd.Series) -> pd.Series:
    h = pd.to_numeric(height_cm, errors="coerce")
    s = sex.astype(str).str.upper().str[0]
    male = 50.0 + 0.91 * (h - 152.4)
    female = 45.5 + 0.91 * (h - 152.4)
    out = np.where(s.eq("M"), male, np.where(s.eq("F"), female, np.nan))
    return pd.Series(out, index=height_cm.index).where((h >= HEIGHT_MIN) & (h <= HEIGHT_MAX))


def hours_between(t: pd.Series, ref: pd.Series) -> pd.Series:
    return (pd.to_datetime(t, errors="coerce") - pd.to_datetime(ref, errors="coerce")).dt.total_seconds() / 3600.0


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- MIMIC
def build_mimic(log: list[str]) -> dict[str, pd.DataFrame]:
    log.append(f"[{now_stamp()}] MIMIC-IV: loading derived tables...")
    base = RAW / "mimiciv_derived" / "mimiciv_derived"

    detail = pd.read_csv(base / "icustay_detail.csv")
    detail["icu_intime"] = pd.to_datetime(detail["icu_intime"], dayfirst=True, errors="coerce")
    detail["icu_outtime"] = pd.to_datetime(detail["icu_outtime"], dayfirst=True, errors="coerce")
    detail["admittime"] = pd.to_datetime(detail["admittime"], dayfirst=True, errors="coerce")
    detail["dischtime"] = pd.to_datetime(detail["dischtime"], dayfirst=True, errors="coerce")
    n_total = int(detail["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV total ICU stays: {n_total}")

    # adults + first ICU stay
    detail = detail[
        (detail["first_icu_stay"].astype(str).str.lower().eq("t"))
        & (pd.to_numeric(detail["admission_age"], errors="coerce") >= AGE_MIN)
    ].copy()
    n_adult_first = int(detail["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV adult first ICU stays: {n_adult_first}")

    # early invasive ventilation 0-24h
    vent = pd.read_csv(base / "ventilation.csv")
    vent["starttime"] = pd.to_datetime(vent["starttime"], dayfirst=True, errors="coerce")
    vent["endtime"] = pd.to_datetime(vent["endtime"], dayfirst=True, errors="coerce")
    invasive = vent[vent["ventilation_status"].eq("InvasiveVent")].merge(
        detail[["stay_id", "subject_id", "hadm_id", "icu_intime"]],
        on="stay_id", how="inner",
    )
    invasive["start_hour"] = (invasive["starttime"] - invasive["icu_intime"]).dt.total_seconds() / 3600
    invasive["end_hour"] = (invasive["endtime"] - invasive["icu_intime"]).dt.total_seconds() / 3600
    early_ids = invasive[(invasive["start_hour"] <= 24) & (invasive["end_hour"] >= 0)]["stay_id"].drop_duplicates()
    cohort = detail[detail["stay_id"].isin(early_ids)].copy()
    n_early = int(cohort["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV early invasive vent cohort: {n_early}")

    # height/weight/PBW
    height = pd.read_csv(base / "first_day_height.csv", usecols=["stay_id", "height"])
    weight = pd.read_csv(base / "first_day_weight.csv", usecols=["stay_id", "weight"])
    cohort = cohort.merge(height, on="stay_id", how="left").merge(weight, on="stay_id", how="left")
    cohort["pbw"] = pbw_from_height_sex(cohort["height"], cohort["gender"])
    n_pbw = int(cohort["pbw"].notna().sum())
    log.append(f"[{now_stamp()}] MIMIC-IV has PBW: {n_pbw}/{n_early}")

    # mortality
    cohort["hospital_expire_flag"] = pd.to_numeric(cohort["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)

    # ---- blood gas points
    bg = pd.read_csv(base / "bg.csv", usecols=["subject_id", "hadm_id", "charttime", "specimen", "po2", "pco2", "fio2", "pao2fio2ratio", "ph", "lactate", "bicarbonate", "baseexcess"])
    bg["charttime"] = pd.to_datetime(bg["charttime"], dayfirst=True, errors="coerce")
    bg = bg.merge(cohort[["stay_id", "subject_id", "hadm_id", "icu_intime"]], on=["subject_id", "hadm_id"], how="inner")
    bg["hour"] = (bg["charttime"] - bg["icu_intime"]).dt.total_seconds() / 3600
    bg = bg[(bg["hour"] >= 0) & (bg["hour"] <= 24)].copy()
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2fio2ratio"], errors="coerce")
    calc_pf = pd.to_numeric(bg["po2"], errors="coerce") / bg["fio2_norm"]
    bg.loc[bg["pf"].isna(), "pf"] = calc_pf
    bg["has_pf_point"] = bg["pf"].between(PF_MIN, PF_MAX)
    bg["has_pco2_point"] = pd.to_numeric(bg["pco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)
    n_pf = int(bg[bg["has_pf_point"]]["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV has >=1 P/F point (0-24h): {n_pf}")

    # ---- minute ventilation points
    vs = pd.read_csv(base / "ventilator_setting.csv", usecols=["stay_id", "charttime", "minute_volume"])
    vs["charttime"] = pd.to_datetime(vs["charttime"], dayfirst=True, errors="coerce")
    vs = vs.merge(cohort[["stay_id", "icu_intime"]], on="stay_id", how="inner")
    vs["hour"] = (vs["charttime"] - vs["icu_intime"]).dt.total_seconds() / 3600
    vs = vs[(vs["hour"] >= 0) & (vs["hour"] <= 24)].copy()
    vs["minute_volume"] = pd.to_numeric(vs["minute_volume"], errors="coerce")
    vs = vs[vs["minute_volume"].between(MV_MIN, MV_MAX)]
    n_mv = int(vs["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV has >=1 minute ventilation point: {n_mv}")

    # ---- nearest-neighbor pairing within ±2h
    bg_valid = bg[bg["has_pf_point"] & bg["has_pco2_point"]]
    paired = nearest_neighbor_pair(bg_valid, vs, id_col="stay_id", time_col="hour")
    paired = paired.merge(cohort[["stay_id", "pbw"]], on="stay_id", how="left")
    paired["pbw_use"] = paired["pbw"]
    denom = paired["pbw_use"] * 100 * 37.5 / 1000
    paired["vr"] = paired["minute_volume"] * pd.to_numeric(paired["pco2"], errors="coerce") / denom
    paired["has_vr_point"] = paired["vr"].between(VR_MIN, VR_MAX) & paired["pf"].between(PF_MIN, PF_MAX)
    n_vr = int(paired[paired["has_vr_point"]]["stay_id"].nunique())
    log.append(f"[{now_stamp()}] MIMIC-IV has >=1 VR point (±2h NN): {n_vr}")

    return {
        "cohort": cohort,
        "bg_points": bg[bg["has_pf_point"]][["stay_id", "hour", "pf", "pco2", "ph", "lactate", "bicarbonate", "baseexcess", "specimen"]],
        "vr_points": paired[paired["has_vr_point"]][["stay_id", "hour", "pf", "pco2", "ph", "minute_volume", "vr", "pair_lag_hours"]],
        "n_total": n_total, "n_adult_first": n_adult_first, "n_early": n_early, "n_pbw": n_pbw, "n_pf": n_pf, "n_mv": n_mv, "n_vr": n_vr,
    }


# ---------------------------------------------------------------- eICU
def build_eicu(log: list[str]) -> dict[str, pd.DataFrame]:
    log.append(f"[{now_stamp()}] eICU: loading tables...")
    base = RAW / "eICU" / "eICU"

    detail = pd.read_csv(base / "icustay_detail.csv")
    detail = detail[pd.to_numeric(detail["age"], errors="coerce").ge(AGE_MIN)].copy()
    n_total = int(detail["patientunitstayid"].nunique())
    log.append(f"[{now_stamp()}] eICU total adult ICU stays: {n_total}")

    # gender: eICU icustay_detail uses 1/0 numeric or F/M; coerce robustly
    g = detail["gender"].astype(str).str.upper().str[0]
    detail["gender_chr"] = np.where(g.eq("M"), "M", np.where(g.eq("F"), "F", np.where(g.eq("1"), "M", np.where(g.eq("0"), "F", ""))))
    detail["pbw"] = pbw_from_height_sex(detail["admissionheight"], detail["gender_chr"])
    n_pbw = int(detail["pbw"].notna().sum())
    log.append(f"[{now_stamp()}] eICU has PBW (all adult): {n_pbw}/{n_total}")

    # early invasive ventilation 0-24h
    events = pd.read_csv(base / "ventilation_events.csv")
    mv = events[events["event"].eq("mechvent start")]
    early_ids = mv[pd.to_numeric(mv["hrs"], errors="coerce").between(0, 24)]["patientunitstayid"].drop_duplicates()
    cohort = detail[detail["patientunitstayid"].isin(early_ids)].copy()
    n_early = int(cohort["patientunitstayid"].nunique())
    log.append(f"[{now_stamp()}] eICU early invasive vent cohort: {n_early}")

    # mortality: hosp_mort typically 0/1
    cohort["hospital_expire_flag"] = pd.to_numeric(cohort["hosp_mort"], errors="coerce").fillna(0).astype(int)

    # ---- blood gas points
    bg = pd.read_csv(base / "pivoted_bg.csv", usecols=["patientunitstayid", "chartoffset", "fio2", "pao2", "paco2", "ph", "peep"])
    bg = bg[bg["patientunitstayid"].isin(cohort["patientunitstayid"])]
    bg["hour"] = pd.to_numeric(bg["chartoffset"], errors="coerce") / 60
    bg = bg[(bg["hour"] >= 0) & (bg["hour"] <= 24)].copy()
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2"], errors="coerce") / bg["fio2_norm"]
    bg["has_pf_point"] = bg["pf"].between(PF_MIN, PF_MAX)
    bg["has_pco2_point"] = pd.to_numeric(bg["paco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)
    n_pf = int(bg[bg["has_pf_point"]]["patientunitstayid"].nunique())
    log.append(f"[{now_stamp()}] eICU has >=1 P/F point: {n_pf}")

    # ---- minute ventilation from respiratorycharting
    target_labels = {"Exhaled MV", "Adult Con Pt/Vent MinuteVentil", "Measured Ve", "MVe"}
    chunks = []
    for chunk in pd.read_csv(
        base / "respiratorycharting.csv",
        usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"],
        dtype={"respchartvalue": "string"},
        chunksize=1_000_000,
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(cohort["patientunitstayid"]) & chunk["respchartvaluelabel"].isin(target_labels)]
        if chunk.empty:
            continue
        chunk["hour"] = pd.to_numeric(chunk["respchartoffset"], errors="coerce") / 60
        chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 24)].copy()
        chunk["minute_volume"] = pd.to_numeric(chunk["respchartvalue"], errors="coerce")
        chunks.append(chunk[["patientunitstayid", "hour", "minute_volume"]])
    if chunks:
        mv_chart = pd.concat(chunks, ignore_index=True)
        mv_chart = mv_chart[mv_chart["minute_volume"].between(MV_MIN, MV_MAX)]
    else:
        mv_chart = pd.DataFrame(columns=["patientunitstayid", "hour", "minute_volume"])
    n_mv = int(mv_chart["patientunitstayid"].nunique()) if len(mv_chart) else 0
    log.append(f"[{now_stamp()}] eICU has >=1 minute ventilation point: {n_mv}")

    # ---- nearest-neighbor pairing ±2h
    bg_valid = bg[bg["has_pf_point"] & bg["has_pco2_point"]]
    paired = nearest_neighbor_pair(bg_valid, mv_chart, id_col="patientunitstayid", time_col="hour")
    paired = paired.merge(cohort[["patientunitstayid", "pbw"]], on="patientunitstayid", how="left")
    denom = paired["pbw"] * 100 * 37.5 / 1000
    paired["vr"] = paired["minute_volume"] * pd.to_numeric(paired["paco2"], errors="coerce") / denom
    paired["has_vr_point"] = paired["vr"].between(VR_MIN, VR_MAX) & paired["pf"].between(PF_MIN, PF_MAX)
    n_vr = int(paired[paired["has_vr_point"]]["patientunitstayid"].nunique())
    log.append(f"[{now_stamp()}] eICU has >=1 VR point (±2h NN): {n_vr}")

    return {
        "cohort": cohort,
        "bg_points": bg[bg["has_pf_point"]][["patientunitstayid", "hour", "pf", "paco2", "ph", "peep"]],
        "vr_points": paired[paired["has_vr_point"]][["patientunitstayid", "hour", "pf", "paco2", "ph", "minute_volume", "vr", "pair_lag_hours"]],
        "n_total": n_total, "n_early": n_early, "n_pbw": n_pbw, "n_pf": n_pf, "n_mv": n_mv, "n_vr": n_vr,
    }


# ---------------------------------------------------------------- pairing
def nearest_neighbor_pair(bg: pd.DataFrame, mv: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    """For each blood-gas point, find the nearest minute ventilation record
    within ±PAIR_WINDOW_H. If two are equidistant, prefer the earlier record
    (minute ventilation preceding blood gas)."""
    if bg.empty or mv.empty:
        return bg.assign(minute_volume=np.nan, pair_lag_hours=np.nan)

    # rename mv time for clarity
    mv = mv.rename(columns={time_col: "mv_hour"}).copy()
    bg = bg.copy()
    bg_key = bg[id_col].values
    bg_t = bg[time_col].values

    out_chunks = []
    for sid, sub_bg in bg.groupby(id_col):
        sub_mv = mv[mv[id_col] == sid]
        if sub_mv.empty:
            continue
        mv_t = sub_mv["mv_hour"].values
        # for each bg point, find nearest mv point
        for idx, row in sub_bg.iterrows():
            t = row[time_col]
            diffs = mv_t - t
            abs_d = np.abs(diffs)
            j = np.argmin(abs_d)
            lag = diffs[j]
            if abs(lag) > PAIR_WINDOW_H:
                continue
            # tie-break: prefer earlier mv (lag<=0)
            ties = np.where(abs_d == abs_d[j])[0]
            if len(ties) > 1:
                neg = ties[diffs[ties] <= 0]
                if len(neg) > 0:
                    j = neg[-1]
            new = row.copy()
            new["minute_volume"] = sub_mv.iloc[j]["minute_volume"].item() if hasattr(sub_mv.iloc[j]["minute_volume"], "item") else float(sub_mv.iloc[j]["minute_volume"])
            new["mv_hour"] = float(mv_t[j])
            new["pair_lag_hours"] = float(lag)
            out_chunks.append(new)
    if not out_chunks:
        return bg.assign(minute_volume=np.nan, pair_lag_hours=np.nan)
    return pd.DataFrame(out_chunks)


# ---------------------------------------------------------------- analysis-ready
def make_analysis_ready(points: pd.DataFrame, id_col: str, cohort: pd.DataFrame, mort_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build patient-level analysis dataset with single-point, dynamic and
    count features from a long points frame containing both P/F and VR.

    `points` must have columns: id_col, hour, pf, vr (vr may be NaN), pco2, ph,
    minute_volume, pair_lag_hours.
    """
    pf_pts = points.dropna(subset=["pf"])
    vr_pts = points.dropna(subset=["vr"])

    rows = []
    for sid, sub in points.groupby(id_col):
        sub_pf = sub.dropna(subset=["pf"]).sort_values("hour")
        sub_vr = sub.dropna(subset=["vr"]).sort_values("hour")
        rec = {
            id_col: sid,
            "pf_points": int(len(sub_pf)),
            "vr_points": int(len(sub_vr)),
            "pf_initial": float(sub_pf["pf"].iloc[0]) if len(sub_pf) else np.nan,
            "pf_min": float(sub_pf["pf"].min()) if len(sub_pf) else np.nan,
            "pf_mean": float(sub_pf["pf"].mean()) if len(sub_pf) else np.nan,
            "pf_last": float(sub_pf["pf"].iloc[-1]) if len(sub_pf) else np.nan,
            "vr_initial": float(sub_vr["vr"].iloc[0]) if len(sub_vr) else np.nan,
            "vr_max": float(sub_vr["vr"].max()) if len(sub_vr) else np.nan,
            "vr_mean": float(sub_vr["vr"].mean()) if len(sub_vr) else np.nan,
            "vr_last": float(sub_vr["vr"].iloc[-1]) if len(sub_vr) else np.nan,
            "paco2_initial": float(sub_pf["paco2"].iloc[0]) if len(sub_pf) and not pd.isna(sub_pf["paco2"].iloc[0]) else np.nan,
            "ph_initial": float(sub_pf["ph"].iloc[0]) if len(sub_pf) and not pd.isna(sub_pf["ph"].iloc[0]) else np.nan,
        }
        if len(sub_pf) >= 2:
            dt = sub_pf["hour"].iloc[-1] - sub_pf["hour"].iloc[0]
            rec["pf_delta"] = float(sub_pf["pf"].iloc[-1] - sub_pf["pf"].iloc[0])
            rec["pf_slope"] = float(rec["pf_delta"] / dt) if dt > 0 else np.nan
            rec["pf_time_window_h"] = float(dt)
        if len(sub_vr) >= 2:
            dt = sub_vr["hour"].iloc[-1] - sub_vr["hour"].iloc[0]
            rec["vr_delta"] = float(sub_vr["vr"].iloc[-1] - sub_vr["vr"].iloc[0])
            rec["vr_slope"] = float(rec["vr_delta"] / dt) if dt > 0 else np.nan
            rec["vr_time_window_h"] = float(dt)
        if len(sub_vr) >= 2 and len(sub_pf) >= 2:
            rec["pf_improved"] = bool(rec["pf_delta"] > 0)
            rec["vr_improved"] = bool(rec["vr_delta"] < 0)
        rows.append(rec)
    feat = pd.DataFrame(rows)
    # joint dynamic typology
    def _typ(r):
        if pd.isna(r.get("pf_improved")) or pd.isna(r.get("vr_improved")):
            return np.nan
        if r["pf_improved"] and r["vr_improved"]:
            return "pf_up_vr_down"
        if r["pf_improved"] and not r["vr_improved"]:
            return "pf_up_vr_not_down"
        if not r["pf_improved"] and r["vr_improved"]:
            return "pf_not_up_vr_down"
        return "pf_not_up_vr_not_down"
    if len(feat):
        feat["dynamic_type"] = feat.apply(_typ, axis=1)
    feat = feat.merge(cohort[[id_col, mort_col]], on=id_col, how="left")
    return feat, points


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["variable", "n_missing", "percent_missing"])
    out = []
    for c in df.columns:
        m = int(df[c].isna().sum())
        out.append({"variable": c, "n_missing": m, "percent_missing": round(m / n * 100, 2)})
    return pd.DataFrame(out)


def cohort_flow_rows(db: str, counts: dict) -> pd.DataFrame:
    rows = [
        {"database": db, "step": "1_total_icu_stays", "n": counts["n_total"]},
        {"database": db, "step": "2_adult_first_icu_stay" if "n_adult_first" in counts else "2_adult_icu_stays", "n": counts.get("n_adult_first", counts["n_total"])},
        {"database": db, "step": "3_early_invasive_vent_0_24h", "n": counts["n_early"]},
        {"database": db, "step": "4_has_pbw", "n": counts["n_pbw"]},
        {"database": db, "step": "5_has_pf_point", "n": counts["n_pf"]},
        {"database": db, "step": "6_has_minute_ventilation", "n": counts["n_mv"]},
        {"database": db, "step": "7_has_vr_point_pair2h", "n": counts["n_vr"]},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_out()
    log: list[str] = [f"[{now_stamp()}] Start phase 1 data build."]

    m = build_mimic(log)
    e = build_eicu(log)

    # ---- combine long points for analysis-ready builder
    m_points = m["vr_points"].rename(columns={"stay_id": "id", "pco2": "paco2"}).assign(database="MIMIC-IV")
    m_points = m_points.rename(columns={"id": "stay_id"})
    e_points = e["vr_points"].rename(columns={"patientunitstayid": "stay_id"}).assign(database="eICU")
    # unify columns
    cols = ["stay_id", "hour", "pf", "paco2", "ph", "minute_volume", "vr", "pair_lag_hours"]
    m_points = m_points[cols + ["database"]]
    e_points = e_points[cols + ["database"]]

    m_feat, m_long = make_analysis_ready(m_points, "stay_id", m["cohort"].rename(columns={"stay_id": "stay_id"}), "hospital_expire_flag")
    e_feat, e_long = make_analysis_ready(e_points, "stay_id", e["cohort"].rename(columns={"patientunitstayid": "stay_id"}), "hospital_expire_flag")
    m_feat["database"] = "MIMIC-IV"; e_feat["database"] = "eICU"

    # ---- point-count summary
    def _count_summary(feat: pd.DataFrame, db: str) -> pd.DataFrame:
        rows = []
        for k, cond in [
            (">=1 pf+vr", (feat["pf_points"] >= 1) & (feat["vr_points"] >= 1)),
            (">=2 pf+vr", (feat["pf_points"] >= 2) & (feat["vr_points"] >= 2)),
            (">=3 pf+vr", (feat["pf_points"] >= 3) & (feat["vr_points"] >= 3)),
            (">=4 pf+vr", (feat["pf_points"] >= 4) & (feat["vr_points"] >= 4)),
        ]:
            s = feat[cond]
            rows.append({
                "database": db,
                "subset": k,
                "n": int(len(s)),
                "deaths": int(s["hospital_expire_flag"].sum()),
                "mortality_percent": round(s["hospital_expire_flag"].mean() * 100, 1) if len(s) else np.nan,
            })
        return pd.DataFrame(rows)
    counts_df = pd.concat([_count_summary(m_feat, "MIMIC-IV"), _count_summary(e_feat, "eICU")], ignore_index=True)
    log.append(f"[{now_stamp()}] Point-count summary:\n{counts_df.to_string(index=False)}")

    # ---- flow
    flow = pd.concat([
        cohort_flow_rows("MIMIC-IV", m),
        cohort_flow_rows("eICU", e),
    ], ignore_index=True)
    # add point-count subsets to flow for visibility
    extra = []
    for db, feat in [("MIMIC-IV", m_feat), ("eICU", e_feat)]:
        for label, cond in [
            ("8_has_any_pf_vr", (feat["pf_points"] >= 1) & (feat["vr_points"] >= 1)),
            ("9_has_2plus_pf_vr", (feat["pf_points"] >= 2) & (feat["vr_points"] >= 2)),
            ("10_has_3plus_pf_vr", (feat["pf_points"] >= 3) & (feat["vr_points"] >= 3)),
            ("11_has_4plus_pf_vr", (feat["pf_points"] >= 4) & (feat["vr_points"] >= 4)),
        ]:
            extra.append({"database": db, "step": label, "n": int(cond.sum())})
    flow = pd.concat([flow, pd.DataFrame(extra)], ignore_index=True)

    # ---- missingness (on combined feature set)
    feat_all = pd.concat([m_feat, e_feat], ignore_index=True)
    miss = missingness(feat_all.drop(columns=["dynamic_type", "database"]))
    miss.insert(0, "database", "combined")

    # ---- variable dictionary
    var_dict = pd.DataFrame([
        {"variable": "pf_points", "definition": "number of valid P/F points in ICU hour 0-24", "source": "blood gas"},
        {"variable": "vr_points", "definition": "number of valid VR points in ICU hour 0-24 (±2h NN pair)", "source": "blood gas + ventilator_setting/respiratorycharting"},
        {"variable": "pf_initial", "definition": "first valid P/F value in 0-24h", "source": "bg.pao2fio2ratio or po2/fio2"},
        {"variable": "pf_min", "definition": "minimum valid P/F in 0-24h", "source": "bg"},
        {"variable": "pf_mean", "definition": "mean valid P/F in 0-24h", "source": "bg"},
        {"variable": "pf_last", "definition": "last valid P/F in 0-24h", "source": "bg"},
        {"variable": "pf_delta", "definition": "last P/F - first P/F (>=2 pts only)", "source": "bg"},
        {"variable": "pf_slope", "definition": "pf_delta / time_window_h (>=2 pts only)", "source": "bg"},
        {"variable": "vr_initial", "definition": "first valid VR in 0-24h", "source": "bg + mv"},
        {"variable": "vr_max", "definition": "maximum valid VR in 0-24h", "source": "bg + mv"},
        {"variable": "vr_mean", "definition": "mean valid VR in 0-24h", "source": "bg + mv"},
        {"variable": "vr_last", "definition": "last valid VR in 0-24h", "source": "bg + mv"},
        {"variable": "vr_delta", "definition": "last VR - first VR (>=2 pts only)", "source": "bg + mv"},
        {"variable": "vr_slope", "definition": "vr_delta / time_window_h (>=2 pts only)", "source": "bg + mv"},
        {"variable": "paco2_initial", "definition": "first PaCO2 at first P/F point", "source": "bg"},
        {"variable": "ph_initial", "definition": "first pH at first P/F point", "source": "bg"},
        {"variable": "dynamic_type", "definition": "joint dynamic typology (pf_up/down x vr_down/up) for >=2 pts", "source": "derived"},
        {"variable": "hospital_expire_flag", "definition": "hospital mortality (0/1)", "source": "admissions/icustay_detail"},
    ])
    var_dict["range_or_unit"] = ""
    var_dict.loc[var_dict["variable"].str.startswith("pf"), "range_or_unit"] = f"{PF_MIN}-{PF_MAX}"
    var_dict.loc[var_dict["variable"].str.startswith("vr"), "range_or_unit"] = f"{VR_MIN}-{VR_MAX}"

    # ---- save
    flow.to_csv(OUT / "cohort_flow_combined.csv", index=False)
    counts_df.to_csv(OUT / "pf_vr_point_count_summary.csv", index=False)
    m_feat.to_csv(OUT / "mimic_analysis_ready.csv", index=False)
    e_feat.to_csv(OUT / "eicu_analysis_ready.csv", index=False)
    m_long.to_csv(OUT / "mimic_points_long.csv", index=False)
    e_long.to_csv(OUT / "eicu_points_long.csv", index=False)
    var_dict.to_csv(OUT / "variable_dictionary.csv", index=False)
    miss.to_csv(OUT / "missingness_summary.csv", index=False)
    pd.concat([m_feat, e_feat], ignore_index=True).to_csv(OUT / "combined_analysis_ready.csv", index=False)

    # exclude logs (patients excluded before reaching each step)
    excl_m = m["cohort"][~m["cohort"]["stay_id"].isin(m_feat[m_feat["pf_points"] >= 1]["stay_id"])][["stay_id", "admission_age", "gender", "height", "pbw", "hospital_expire_flag"]].copy()
    excl_m["reason"] = "no_pf_point_or_no_vr_point"
    excl_e = e["cohort"][~e["cohort"]["patientunitstayid"].isin(e_feat[e_feat["pf_points"] >= 1]["stay_id"])][["patientunitstayid", "age", "gender_chr", "admissionheight", "pbw", "hospital_expire_flag"]].copy()
    excl_e["reason"] = "no_pf_point_or_no_vr_point"
    excl_m.to_csv(OUT / "excluded_mimic.csv", index=False)
    excl_e.to_csv(OUT / "excluded_eicu.csv", index=False)

    # ---- baseline table (any pf+vr subset)
    def _table1(feat: pd.DataFrame, cohort: pd.DataFrame, db: str) -> pd.DataFrame:
        sub = feat[(feat["pf_points"] >= 1) & (feat["vr_points"] >= 1)].copy()
        if db == "MIMIC-IV":
            base = cohort[cohort["stay_id"].isin(sub["stay_id"])][["stay_id", "admission_age", "gender", "height", "weight", "pbw", "hospital_expire_flag"]]
            base = base.rename(columns={"admission_age": "age", "gender": "sex"})
        else:
            base = cohort[cohort["patientunitstayid"].isin(sub["stay_id"])][["patientunitstayid", "age", "gender_chr", "admissionheight", "admissionweight", "pbw", "hospital_expire_flag"]]
            base = base.rename(columns={"patientunitstayid": "stay_id", "gender_chr": "sex", "admissionheight": "height", "admissionweight": "weight"})
        base["database"] = db
        return base
    t1 = pd.concat([_table1(m_feat, m["cohort"], "MIMIC-IV"), _table1(e_feat, e["cohort"], "eICU")], ignore_index=True)
    t1.to_csv(OUT / "table1_baseline.csv", index=False)

    # ---- log
    log.append(f"[{now_stamp()}] Done. Output directory: {OUT}")
    (OUT / "analysis_log.md").write_text("\n".join(log), encoding="utf-8")

    # ---- console summary
    print("\n=== Phase 1 build summary ===")
    print(flow.to_string(index=False))
    print("\n=== Point-count summary ===")
    print(counts_df.to_string(index=False))
    print("\n=== Variable dictionary ===")
    print(var_dict[["variable", "definition"]].to_string(index=False))
    print(f"\nLog written to {OUT/'analysis_log.md'}")


if __name__ == "__main__":
    main()