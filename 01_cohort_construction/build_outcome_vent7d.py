#!/usr/bin/env python3
"""Build the primary outcome: prolonged invasive mechanical ventilation >=7 days.

Builds per-stay vent_7d (0/1) before trajectory construction.
Rules:
  - MIMIC-IV: sum durations of all InvasiveVent segments per stay (from
    ventilation.csv). vent_7d = total_invasive_vent_hours >= 168.
  - eICU: use apachepatientresult.actualventdays >= 7.
  - Does NOT modify phase 1 P/F+VR features.
  - This secondary outcome is not used to select predictors in the 48-h
    hospital-mortality landmark model.
  - The inclusion criterion "invasive vent started within 0-24h" is NOT used
    as a predictor.

Outputs:
  outputs/pf_vr_dynamic_analysis/mimic_outcome_vent7d.csv
  outputs/pf_vr_dynamic_analysis/eicu_outcome_vent7d.csv
  outputs/pf_vr_dynamic_analysis/outcome_summary.csv
  outputs/pf_vr_dynamic_analysis/build_outcome_log.md
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "原始数据"
OUT = ROOT / "outputs" / "pf_vr_dynamic_analysis"
MIMIC_DERIVED = RAW / "mimiciv_derived" / "mimiciv_derived"
EICU = RAW / "eICU" / "eICU"
VENT_7D_H = 168  # hours


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_mimic_outcome() -> pd.DataFrame:
    """stay_id, inv_vent_h, vent_7d, plus hosp mortality for sensitivity."""
    v = pd.read_csv(MIMIC_DERIVED / "ventilation.csv")
    v["starttime"] = pd.to_datetime(v["starttime"], dayfirst=True, errors="coerce")
    v["endtime"] = pd.to_datetime(v["endtime"], dayfirst=True, errors="coerce")
    inv = v[v["ventilation_status"] == "InvasiveVent"].copy()
    inv["dur_h"] = (inv["endtime"] - inv["starttime"]).dt.total_seconds() / 3600
    # clip negative durations to 0 (data errors)
    inv["dur_h"] = inv["dur_h"].clip(lower=0)
    grp = inv.groupby("stay_id").agg(
        inv_vent_h=("dur_h", "sum"),
        inv_segments=("dur_h", "size"),
    ).reset_index()
    grp["vent_7d"] = (grp["inv_vent_h"] >= VENT_7D_H).astype(int)

    # hospital mortality for later sensitivity (exclude early death)
    det = pd.read_csv(MIMIC_DERIVED / "icustay_detail.csv",
                      usecols=["stay_id", "los_icu", "hospital_expire_flag"])
    grp = grp.merge(det, on="stay_id", how="left")
    grp["los_icu"] = pd.to_numeric(grp["los_icu"], errors="coerce")
    return grp


def build_eicu_outcome() -> pd.DataFrame:
    ap = pd.read_csv(EICU / "apachepatientresult.csv",
                     usecols=["patientunitstayid", "actualventdays", "actualiculos"])
    ap = ap.dropna(subset=["patientunitstayid"]).drop_duplicates("patientunitstayid")
    ap = ap.rename(columns={"patientunitstayid": "stay_id"})
    ap["actualventdays"] = pd.to_numeric(ap["actualventdays"], errors="coerce")
    ap["vent_7d"] = (ap["actualventdays"] >= 7).astype(int)
    ap["actualiculos"] = pd.to_numeric(ap["actualiculos"], errors="coerce")
    # mortality for sensitivity
    det = pd.read_csv(EICU / "icustay_detail.csv",
                      usecols=["patientunitstayid", "hosp_mort"])
    det = det.rename(columns={"patientunitstayid": "stay_id", "hosp_mort": "hospital_expire_flag"})
    ap = ap.merge(det, on="stay_id", how="left")
    return ap[["stay_id", "actualventdays", "vent_7d", "actualiculos", "hospital_expire_flag"]]


def main() -> None:
    log = [f"[{now()}] Build vent_7d outcome."]
    OUT.mkdir(parents=True, exist_ok=True)

    m_out = build_mimic_outcome()
    e_out = build_eicu_outcome()
    log.append(f"[{now()}] MIMIC outcome rows: {len(m_out)} | vent_7d events: {int(m_out['vent_7d'].sum())} ({m_out['vent_7d'].mean()*100:.1f}%)")
    log.append(f"[{now()}] eICU outcome rows: {len(e_out)} | vent_7d events: {int(e_out['vent_7d'].sum())} ({e_out['vent_7d'].mean()*100:.1f}%)")

    m_out.to_csv(OUT / "mimic_outcome_vent7d.csv", index=False)
    e_out.to_csv(OUT / "eicu_outcome_vent7d.csv", index=False)

    summary = pd.DataFrame([
        {"database": "MIMIC-IV", "cohort": "all_with_outcome", "n": len(m_out),
         "events": int(m_out["vent_7d"].sum()), "event_rate_percent": round(m_out["vent_7d"].mean()*100, 1)},
        {"database": "eICU", "cohort": "all_with_outcome", "n": len(e_out),
         "events": int(e_out["vent_7d"].sum()), "event_rate_percent": round(e_out["vent_7d"].mean()*100, 1)},
    ])
    summary.to_csv(OUT / "outcome_summary.csv", index=False)
    log.append(f"[{now()}] Outcome summary:\n{summary.to_string(index=False)}")
    (OUT / "build_outcome_log.md").write_text("\n".join(log), encoding="utf-8")

    print("\n=== Outcome summary ===")
    print(summary.to_string(index=False))
    print(f"\nOutputs: mimic_outcome_vent7d.csv, eicu_outcome_vent7d.csv")


if __name__ == "__main__":
    main()
