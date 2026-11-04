#!/usr/bin/env python3
"""Build 0-48h P/F+VR trajectory data with ±2h nearest-neighbor pairing.
Extends the 24h window to 48h to increase point density and trajectory stability.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "原始数据"
OUT = ROOT / "outputs" / "pf_vr_dynamic_analysis"
MIMIC_DERIVED = RAW / "mimiciv_derived" / "mimiciv_derived"
EICU = RAW / "eICU" / "eICU"
WINDOW_H = 48
PAIR_WINDOW = 2.0
MIN_POINTS = 3
PF_MIN, PF_MAX = 30, 700
VR_MIN, VR_MAX = 0.2, 10
MV_MIN, MV_MAX = 1, 30
PCO2_MIN, PCO2_MAX = 10, 150


def norm_fio2(x):
    y = pd.to_numeric(x, errors="coerce")
    return pd.Series(np.where(y > 1.0, y / 100.0, y), index=y.index)


def pbw_from_height_sex(h, s):
    h = pd.to_numeric(h, errors="coerce")
    s = s.astype(str).str.upper().str[0]
    male = 50.0 + 0.91 * (h - 152.4)
    female = 45.5 + 0.91 * (h - 152.4)
    out = np.where(s.eq("M"), male, np.where(s.eq("F"), female, np.nan))
    return pd.Series(out, index=h.index).where((h >= 120) & (h <= 230))


def nearest_neighbor_pair(bg, mv, id_col, time_col, window=PAIR_WINDOW):
    """For each bg point, find nearest mv within ±window hours."""
    if bg.empty or mv.empty:
        return bg.assign(minute_volume=np.nan, pair_lag_hours=np.nan)
    out = []
    for sid, sub_bg in bg.groupby(id_col):
        sub_mv = mv[mv[id_col] == sid]
        if sub_mv.empty:
            continue
        mv_t = sub_mv[time_col].values
        for _, row in sub_bg.iterrows():
            t = row[time_col]
            diffs = mv_t - t
            abs_d = np.abs(diffs)
            j = int(np.argmin(abs_d))
            if abs_d[j] > window:
                continue
            ties = np.where(abs_d == abs_d[j])[0]
            if len(ties) > 1:
                neg = ties[diffs[ties] <= 0]
                if len(neg) > 0:
                    j = int(neg[-1])
            new = row.copy()
            new["minute_volume"] = float(sub_mv.iloc[j]["minute_volume"])
            new["pair_lag_hours"] = float(diffs[j])
            out.append(new)
    return pd.DataFrame(out) if out else bg.assign(minute_volume=np.nan, pair_lag_hours=np.nan)






RAW_RAW_DIR = str(ROOT / "原始数据")  # PhysioNet flat exports; override if placed elsewhere


def build_mimic_48h():
    detail = pd.read_csv(MIMIC_DERIVED / "icustay_detail.csv")
    detail["icu_intime"] = pd.to_datetime(detail["icu_intime"], dayfirst=True, errors="coerce")
    detail = detail[(detail["first_icu_stay"].astype(str).str.lower().eq("t")) &
                    (pd.to_numeric(detail["admission_age"], errors="coerce") >= 18)].copy()

    vent = pd.read_csv(MIMIC_DERIVED / "ventilation.csv")
    vent["starttime"] = pd.to_datetime(vent["starttime"], dayfirst=True, errors="coerce")
    vent["endtime"] = pd.to_datetime(vent["endtime"], dayfirst=True, errors="coerce")
    inv = vent[vent["ventilation_status"] == "InvasiveVent"].merge(
        detail[["stay_id", "icu_intime"]], on="stay_id", how="inner")
    inv["start_h"] = (inv["starttime"] - inv["icu_intime"]).dt.total_seconds() / 3600
    inv["end_h"] = (inv["endtime"] - inv["icu_intime"]).dt.total_seconds() / 3600
    early_ids = inv[(inv["start_h"] <= 24) & (inv["end_h"] >= 0)]["stay_id"].drop_duplicates()
    cohort = detail[detail["stay_id"].isin(early_ids)].copy()

    height = pd.read_csv(MIMIC_DERIVED / "first_day_height.csv", usecols=["stay_id", "height"])
    cohort = cohort.merge(height, on="stay_id", how="left")
    cohort["pbw"] = pbw_from_height_sex(cohort["height"], cohort["gender"])

    # bg 0-48h
    bg = pd.read_csv(MIMIC_DERIVED / "bg.csv",
                     usecols=["subject_id", "hadm_id", "charttime", "po2", "pco2", "fio2", "pao2fio2ratio", "ph"])
    bg["charttime"] = pd.to_datetime(bg["charttime"], dayfirst=True, errors="coerce")
    bg = bg.merge(cohort[["stay_id", "subject_id", "hadm_id", "icu_intime", "pbw"]],
                  on=["subject_id", "hadm_id"], how="inner")
    bg["hour"] = (bg["charttime"] - bg["icu_intime"]).dt.total_seconds() / 3600
    bg = bg[(bg["hour"] >= 0) & (bg["hour"] <= WINDOW_H)].copy()
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2fio2ratio"], errors="coerce")
    calc = pd.to_numeric(bg["po2"], errors="coerce") / bg["fio2_norm"]
    bg.loc[bg["pf"].isna(), "pf"] = calc
    bg["has_pf"] = bg["pf"].between(PF_MIN, PF_MAX)
    bg["has_pco2"] = pd.to_numeric(bg["pco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)

    # minute ventilation 0-48h
    vs = pd.read_csv(MIMIC_DERIVED / "ventilator_setting.csv", usecols=["stay_id", "charttime", "minute_volume"])
    vs["charttime"] = pd.to_datetime(vs["charttime"], dayfirst=True, errors="coerce")
    vs = vs.merge(cohort[["stay_id", "icu_intime"]], on="stay_id", how="inner")
    vs["hour"] = (vs["charttime"] - vs["icu_intime"]).dt.total_seconds() / 3600
    vs = vs[(vs["hour"] >= 0) & (vs["hour"] <= WINDOW_H)].copy()
    vs["minute_volume"] = pd.to_numeric(vs["minute_volume"], errors="coerce")
    vs = vs[vs["minute_volume"].between(MV_MIN, MV_MAX)]

    # NN pairing
    bg_valid = bg[bg["has_pf"] & bg["has_pco2"]]
    paired = nearest_neighbor_pair(bg_valid, vs, "stay_id", "hour")
    paired = paired.merge(cohort[["stay_id", "pbw"]], on="stay_id", how="left", suffixes=("", "_c"))

    # ---- EXPLICIT TRUNCATION (manuscript-stated; round-2 implementation) ----
    # The observation window ends at the earliest of (a) 48 h after ICU admission,
    # (b) hospital death, (c) ICU (unit) discharge, (d) hospital discharge, and
    # (e) terminal extubation (the end of the LAST InvasiveVent segment with
    # starttime within/before the window). A pair is retained only if BOTH its
    # ABG timestamp and its ventilator-setting timestamp fall inside the window.
    core = Path(RAW_RAW_DIR) if RAW_RAW_DIR else None
    if core is not None:
        icu_t = pd.read_csv(core / "icustays.csv")             if (core / "icustays.csv").exists() else None
        adm_t = pd.read_csv(core / "admissions.csv", usecols=["hadm_id", "dischtime", "deathtime"])             if (core / "admissions.csv").exists() else None
    else:
        icu_t = adm_t = None
    if icu_t is not None and adm_t is not None:
        seg_end = inv.groupby("stay_id")["endtime"].max().rename("vent_end").reset_index()
        pb = (paired.merge(icu_t[["stay_id", "intime", "outtime", "hadm_id"]], on="stay_id", how="left")
                    .merge(adm_t, on="hadm_id", how="left")
                    .merge(seg_end, on="stay_id", how="left"))
        for c in ["intime", "outtime", "dischtime", "deathtime", "vent_end"]:
            pb[c] = pd.to_datetime(pb[c], errors="coerce")
        t_abg = pb["intime"] + pd.to_timedelta(pb["hour"], unit="h")
        t_mv = pb["intime"] + pd.to_timedelta(pb["hour"] + pb["pair_lag_hours"], unit="h")
        t_end = pb["intime"] + pd.Timedelta(hours=48)
        for _src in ["outtime", "dischtime", "deathtime", "vent_end"]:
            _v = pb[_src]
            t_end = t_end.mask(_v.notna() & (_v < t_end), _v)
        _keep = (t_abg <= t_end) & (t_mv <= t_end) & t_mv.notna()
        paired = pb.loc[_keep, paired.columns].copy()

    # ---- EXPLICIT TRUNCATION (manuscript-stated; round-2 implementation) ----
    # The observation window ends at the earliest of (a) 48 h after ICU admission,
    # (b) hospital death, (c) ICU (unit) discharge, (d) hospital discharge, and
    # (e) terminal extubation (the end of the LAST InvasiveVent segment with
    # starttime within/before the window). A pair is retained only if BOTH its
    # ABG timestamp and its ventilator-setting timestamp fall inside the window.
    core = Path(RAW_RAW_DIR) if RAW_RAW_DIR else None
    if core is not None:
        icu_t = pd.read_csv(core / "icustays.csv")             if (core / "icustays.csv").exists() else None
        adm_t = pd.read_csv(core / "admissions.csv", usecols=["hadm_id", "dischtime", "deathtime"])             if (core / "admissions.csv").exists() else None
    else:
        icu_t = adm_t = None
    if icu_t is not None and adm_t is not None:
        seg_end = inv.groupby("stay_id")["endtime"].max().rename("vent_end").reset_index()
        pb = (paired.merge(icu_t[["stay_id", "intime", "outtime", "hadm_id"]], on="stay_id", how="left")
                    .merge(adm_t, on="hadm_id", how="left")
                    .merge(seg_end, on="stay_id", how="left"))
        for c in ["intime", "outtime", "dischtime", "deathtime", "vent_end"]:
            pb[c] = pd.to_datetime(pb[c], errors="coerce")
        t_abg = pb["intime"] + pd.to_timedelta(pb["hour"], unit="h")
        t_mv = pb["intime"] + pd.to_timedelta(pb["hour"] + pb["pair_lag_hours"], unit="h")
        t_end = pb["intime"] + pd.Timedelta(hours=48)
        for _src in ["outtime", "dischtime", "deathtime", "vent_end"]:
            _v = pb[_src]
            t_end = t_end.mask(_v.notna() & (_v < t_end), _v)
        _keep = (t_abg <= t_end) & (t_mv <= t_end) & t_mv.notna()
        paired = pb.loc[_keep, paired.columns].copy()
    paired["pbw"] = paired["pbw"].fillna(paired["pbw_c"])
    denom = paired["pbw"] * 100 * 37.5 / 1000
    paired["vr"] = paired["minute_volume"] * pd.to_numeric(paired["pco2"], errors="coerce") / denom
    paired["has_vr"] = paired["vr"].between(VR_MIN, VR_MAX) & paired["pf"].between(PF_MIN, PF_MAX)

    pts = paired[paired["has_vr"]][["stay_id", "hour", "pf", "pco2", "ph", "minute_volume", "vr", "pair_lag_hours"]].copy()
    counts = pts.groupby("stay_id").size()
    keep = counts[counts >= MIN_POINTS].index
    pts = pts[pts["stay_id"].isin(keep)].copy()

    # merge outcomes
    out = pd.read_csv(OUT / "mimic_outcome_vent7d.csv")
    pts = pts.merge(out[["stay_id", "vent_7d", "hospital_expire_flag", "los_icu"]], on="stay_id", how="left")
    pts["los_icu_days"] = pd.to_numeric(pts["los_icu"], errors="coerce")
    pts["icu_los_ge7d"] = (pts["los_icu_days"] >= 7).astype(int)
    pts["database"] = "MIMIC-IV"
    print(f"MIMIC 0-48h: {pts['stay_id'].nunique()} stays with >=3 P/F+VR points, {len(pts)} rows")
    return pts


def build_eicu_48h():
    det = pd.read_csv(EICU / "icustay_detail.csv")
    det = det[pd.to_numeric(det["age"], errors="coerce") >= 18].copy()
    det = det.rename(columns={"patientunitstayid": "stay_id", "hosp_mort": "hospital_expire_flag"})
    det["hospital_expire_flag"] = pd.to_numeric(det["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    g = pd.to_numeric(det["gender"], errors="coerce")
    det["sex"] = np.where(g.eq(1), 1, np.where(g.eq(0), 0, np.nan))
    det["pbw"] = pbw_from_height_sex(det["admissionheight"], pd.Series(np.where(g.eq(1), "M", np.where(g.eq(0), "F", "")), index=det.index))

    events = pd.read_csv(EICU / "ventilation_events.csv")
    mv = events[events["event"].eq("mechvent start")]
    early_ids = mv[pd.to_numeric(mv["hrs"], errors="coerce").between(0, 24)]["patientunitstayid"].drop_duplicates()
    cohort = det[det["stay_id"].isin(early_ids)].copy()

    bg = pd.read_csv(EICU / "pivoted_bg.csv", usecols=["patientunitstayid", "chartoffset", "fio2", "pao2", "paco2", "ph"])
    bg = bg[bg["patientunitstayid"].isin(cohort["stay_id"])]
    bg["hour"] = pd.to_numeric(bg["chartoffset"], errors="coerce") / 60
    bg = bg[(bg["hour"] >= 0) & (bg["hour"] <= WINDOW_H)].copy()
    bg = bg.rename(columns={"patientunitstayid": "stay_id"})
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2"], errors="coerce") / bg["fio2_norm"]
    bg["has_pf"] = bg["pf"].between(PF_MIN, PF_MAX)
    bg["has_pco2"] = pd.to_numeric(bg["paco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)

    target_labels = {"Exhaled MV", "Adult Con Pt/Vent MinuteVentil", "Measured Ve", "MVe"}
    chunks = []
    for chunk in pd.read_csv(EICU / "respiratorycharting.csv",
                             usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"],
                             dtype={"respchartvalue": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(cohort["stay_id"]) &
                      chunk["respchartvaluelabel"].isin(target_labels)]
        if chunk.empty:
            continue
        chunk["hour"] = pd.to_numeric(chunk["respchartoffset"], errors="coerce") / 60
        chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= WINDOW_H)].copy()
        chunk["minute_volume"] = pd.to_numeric(chunk["respchartvalue"], errors="coerce")
        chunks.append(chunk[["patientunitstayid", "hour", "minute_volume"]])
    mv_chart = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["patientunitstayid", "hour", "minute_volume"])
    mv_chart = mv_chart.rename(columns={"patientunitstayid": "stay_id"})
    mv_chart = mv_chart[mv_chart["minute_volume"].between(MV_MIN, MV_MAX)]

    bg_valid = bg[bg["has_pf"] & bg["has_pco2"]]
    paired = nearest_neighbor_pair(bg_valid, mv_chart, "stay_id", "hour")
    paired = paired.merge(cohort[["stay_id", "pbw"]], on="stay_id", how="left")

    # ---- EXPLICIT TRUNCATION for eICU (round-2; all quantities in HOURS) ----
    # window end = earliest of (unit admit + 48 h, ICU discharge, hospital
    # discharge/death, terminal mechanvent end). Terminal semantics: if the last
    # mechvent event is a "start", ventilation is ongoing (no vent cutoff); if it
    # is an "end", the window ends at that time.
    ev = pd.read_csv(RAW / "eICU" / "eICU" / "ventilation_events.csv",
                     usecols=["patientunitstayid", "event", "hrs"], low_memory=False)
    ev = ev.rename(columns={"patientunitstayid": "stay_id"})
    ev["hrs"] = pd.to_numeric(ev["hrs"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    e_start = ev[ev["event"].eq("mechvent start")].groupby("stay_id")["hrs"].max().rename("e_start")
    e_end = ev[ev["event"].eq("mechvent end")].groupby("stay_id")["hrs"].max().rename("e_end")
    pat = pd.read_csv(RAW / "eICU" / "eICU" / "patient.csv",
                      usecols=["patientunitstayid", "hospitaldischargeoffset", "unitdischargeoffset"],
                      low_memory=False).rename(columns={"patientunitstayid": "stay_id"})
    pb = (paired.merge(e_start.reset_index(), on="stay_id", how="left")
                .merge(e_end.reset_index(), on="stay_id", how="left")
                .merge(pat[["stay_id", "hospitaldischargeoffset", "unitdischargeoffset"]],
                       on="stay_id", how="left"))
    pb["t_mv_h"] = pb["hour"] + pd.to_numeric(pb["pair_lag_hours"], errors="coerce")
    t_end = pd.DataFrame({"end": pb["hour"] + WINDOW_H})
    for src in ["hospitaldischargeoffset", "unitdischargeoffset"]:
        v = pd.to_numeric(pb[src], errors="coerce") / 60.0
        t_end["end"] = t_end["end"].mask(v.notna() & (v < t_end["end"]), v)
    ongoing = (pb["e_start"].notna()) & (pb["e_start"].fillna(-1e9) > pb["e_end"].fillna(-1e9))
    vent_end_h = pb["e_end"]
    t_end["end"] = t_end["end"].mask((~ongoing) & vent_end_h.notna() & (vent_end_h < t_end["end"]), vent_end_h)
    _keep = (pd.to_numeric(pb["pair_lag_hours"], errors="coerce").notna()) & \
            (pb["hour"] <= t_end["end"]) & (pb["t_mv_h"] <= t_end["end"])
    paired = pb.loc[_keep, paired.columns].copy()
    print(f"[truncation-eICU] pairs retained: {len(paired)} of {len(pb)}")

    denom = paired["pbw"] * 100 * 37.5 / 1000
    paired["vr"] = paired["minute_volume"] * pd.to_numeric(paired["paco2"], errors="coerce") / denom
    paired["has_vr"] = paired["vr"].between(VR_MIN, VR_MAX) & paired["pf"].between(PF_MIN, PF_MAX)

    pts = paired[paired["has_vr"]][["stay_id", "hour", "pf", "paco2", "ph", "minute_volume", "vr", "pair_lag_hours"]].copy()
    pts = pts.rename(columns={"paco2": "pco2"})
    counts = pts.groupby("stay_id").size()
    keep = counts[counts >= MIN_POINTS].index
    pts = pts[pts["stay_id"].isin(keep)].copy()

    out = pd.read_csv(OUT / "eicu_outcome_vent7d.csv")
    pts = pts.merge(out[["stay_id", "vent_7d", "hospital_expire_flag", "actualiculos"]], on="stay_id", how="left")
    pts["los_icu_days"] = pd.to_numeric(pts["actualiculos"], errors="coerce")
    pts["icu_los_ge7d"] = (pts["los_icu_days"] >= 7).astype(int)
    pts["database"] = "eICU"
    print(f"eICU 0-48h: {pts['stay_id'].nunique()} stays with >=3 P/F+VR points, {len(pts)} rows")
    return pts


def main():
    m = build_mimic_48h()
    e = build_eicu_48h()
    m.to_csv(OUT / "trajectory_mimic_48h.csv", index=False)
    e.to_csv(OUT / "trajectory_eicu_48h.csv", index=False)
    print(f"\nSaved: trajectory_mimic_48h.csv ({len(m)} rows), trajectory_eicu_48h.csv ({len(e)} rows)")


if __name__ == "__main__":
    main()