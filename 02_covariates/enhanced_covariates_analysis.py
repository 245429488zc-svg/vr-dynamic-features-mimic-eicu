#!/usr/bin/env python3
"""Build enhanced covariate dataset with severity scores, ventilator parameters,
labs, vitals, BMI. Merge with existing Option B features and redo model ladder
to test VR's incremental value OVER SOFA/APACHE.

Key question: Does VR remain significant after adjusting for SOFA + vent params + labs?

Outputs:
  outputs/pf_vr_dynamic_analysis/enhanced/mimic_enhanced.csv
  outputs/pf_vr_dynamic_analysis/enhanced/eicu_enhanced.csv
  outputs/pf_vr_dynamic_analysis/enhanced/model_ladder_results.csv
  outputs/pf_vr_dynamic_analysis/enhanced/or_results.csv
  outputs/pf_vr_dynamic_analysis/enhanced/delong_idi_nri.csv
  outputs/pf_vr_dynamic_analysis/enhanced/table1_*.csv
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import cross_val_predict
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "原始数据"
OUT = ROOT / "outputs" / "pf_vr_dynamic_analysis"
MIM = RAW / "mimiciv_derived" / "mimiciv_derived"
EICU = RAW / "eICU" / "eICU"
RES = OUT / "enhanced"
RES.mkdir(parents=True, exist_ok=True)
RNG = 2026


def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- race grouping
def group_race_mimic(race: str) -> str:
    if not isinstance(race, str): return "Other/Unknown"
    r = race.upper()
    if any(x in r for x in ["WHITE", "PORTUGUESE", "BRAZILIAN"]): return "White"
    if any(x in r for x in ["BLACK", "AFRICAN AMERICAN", "CAPE VERDEAN", "CARIBBEAN"]): return "Black"
    if "ASIAN" in r: return "Asian"
    if "HISPANIC" in r or "LATINO" in r: return "Hispanic/Latino"
    return "Other/Unknown"

def group_race_eicu(race: str) -> str:
    if not isinstance(race, str): return "Other/Unknown"
    r = race.lower()
    if "caucasian" in r or "white" in r: return "White"
    if "african" in r or "black" in r: return "Black"
    if "asian" in r: return "Asian"
    if "hispanic" in r: return "Hispanic/Latino"
    return "Other/Unknown"


# ---------------------------------------------------------------- MIMIC covariates
def build_mimic_covariates(stay_ids):
    """Build enhanced covariates for given MIMIC stay_ids."""
    # SOFA 24h
    sofa = pd.read_csv(MIM / "sofa.csv", usecols=["stay_id", "sofa_24hours"])
    sofa = sofa.rename(columns={"sofa_24hours": "sofa"})
    sofa["sofa"] = pd.to_numeric(sofa["sofa"], errors="coerce")
    sofa = sofa.groupby("stay_id", as_index=False)["sofa"].median()

    # APSIII
    apsiii = pd.read_csv(MIM / "apsiii.csv", usecols=["stay_id", "apsiii"])
    apsiii["apsiii"] = pd.to_numeric(apsiii["apsiii"], errors="coerce")

    # Charlson + comorbidities
    charl = pd.read_csv(MIM / "charlson.csv")
    charl = charl.rename(columns={"charlson_comorbidity_index": "charlson",
                                  "chronic_pulmonary_disease": "copd",
                                  "congestive_heart_failure": "chf"})
    charl = charl[["subject_id", "hadm_id", "charlson", "copd", "chf"]].copy()
    charl["charlson"] = pd.to_numeric(charl["charlson"], errors="coerce")

    # Sepsis-3
    sep = pd.read_csv(MIM / "sepsis3.csv", usecols=["stay_id", "sepsis3"])
    sep["sepsis3"] = (sep["sepsis3"].astype(str).str.lower().isin(["t", "true", "1"])).astype(int)
    sep = sep.groupby("stay_id", as_index=False)["sepsis3"].max()

    # Demographics + race
    det = pd.read_csv(MIM / "icustay_detail.csv",
                      usecols=["stay_id", "subject_id", "hadm_id", "gender", "admission_age", "race",
                               "hospital_expire_flag", "los_icu"])
    det = det.rename(columns={"admission_age": "age"})
    det["sex"] = det["gender"].map({"M": 1, "F": 0}).astype("Int64")
    det["race_grouped"] = det["race"].apply(group_race_mimic)

    # Weight/height -> BMI
    wt = pd.read_csv(MIM / "first_day_weight.csv", usecols=["stay_id", "weight"])
    ht = pd.read_csv(MIM / "first_day_height.csv", usecols=["stay_id", "height"])
    det = det.merge(wt, on="stay_id", how="left").merge(ht, on="stay_id", how="left")
    det["bmi"] = det["weight"] / (det["height"] / 100) ** 2
    det.loc[(det["bmi"] < 12) | (det["bmi"] > 80), "bmi"] = np.nan

    # Vitals (first day) - temperature already included
    vit = pd.read_csv(MIM / "first_day_vitalsign.csv",
                      usecols=["stay_id", "heart_rate_mean", "mbp_mean", "resp_rate_mean", "spo2_mean", "temperature_mean"])
    vit = vit.rename(columns={"heart_rate_mean": "hr", "mbp_mean": "map", "resp_rate_mean": "rr_vitalsign",
                              "spo2_mean": "spo2", "temperature_mean": "temp"})

    # Lactate from first_day_bg (not in first_day_lab)
    bg_lab = pd.read_csv(MIM / "first_day_bg.csv", usecols=["stay_id", "lactate_max"])
    bg_lab = bg_lab.rename(columns={"lactate_max": "lactate"})

    # Labs (first day) - key ones
    lab = pd.read_csv(MIM / "first_day_lab.csv",
                      usecols=["stay_id", "bun_max", "creatinine_max", "bilirubin_total_max",
                               "albumin_min", "platelets_min", "wbc_max", "hemoglobin_min",
                               "sodium_max", "potassium_max", "bicarbonate_min"])
    lab = lab.rename(columns={"bun_max": "bun", "creatinine_max": "creatinine",
                              "bilirubin_total_max": "bilirubin", "albumin_min": "albumin",
                              "platelets_min": "platelets", "wbc_max": "wbc",
                              "hemoglobin_min": "hemoglobin", "sodium_max": "sodium",
                              "potassium_max": "potassium", "bicarbonate_min": "bicarbonate"})
    lab = lab.merge(bg_lab, on="stay_id", how="left")

    # Ventilator params (0-48h mean): PEEP, plateau pressure, Vt, FiO2
    vs = pd.read_csv(MIM / "ventilator_setting.csv",
                     usecols=["stay_id", "charttime", "peep", "plateau_pressure", "tidal_volume_set", "fio2"])
    vs["charttime"] = pd.to_datetime(vs["charttime"], dayfirst=True, errors="coerce")
    intime = det[["stay_id", "icu_intime"]].copy() if "icu_intime" in det.columns else None
    # need icu_intime - reload
    det2 = pd.read_csv(MIM / "icustay_detail.csv", usecols=["stay_id", "icu_intime"])
    det2["icu_intime"] = pd.to_datetime(det2["icu_intime"], dayfirst=True, errors="coerce")
    vs = vs.merge(det2, on="stay_id", how="left")
    vs["hour"] = (vs["charttime"] - vs["icu_intime"]).dt.total_seconds() / 3600
    vs = vs[(vs["hour"] >= 0) & (vs["hour"] <= 48)]
    vs_agg = vs.groupby("stay_id").agg(
        peep=("peep", "mean"),
        plateau_pressure=("plateau_pressure", "mean"),
        vt_set=("tidal_volume_set", "mean"),
        fio2_vent=("fio2", "mean"),
    ).reset_index()
    # driving pressure = plateau - peep
    vs_agg["driving_pressure"] = vs_agg["plateau_pressure"] - vs_agg["peep"]
    # Vt/PBW (need PBW from height/sex)
    vs_agg = vs_agg.merge(det[["stay_id", "height", "gender"]], on="stay_id", how="left")
    h = pd.to_numeric(vs_agg["height"], errors="coerce")
    s = vs_agg["gender"].astype(str).str.upper().str[0]
    pbw = np.where(s.eq("M"), 50 + 0.91 * (h - 152.4),
                   np.where(s.eq("F"), 45.5 + 0.91 * (h - 152.4), np.nan))
    vs_agg["vt_pbw"] = vs_agg["vt_set"] / pbw
    vs_agg = vs_agg.drop(columns=["height", "gender"])

    # Vasopressor (norepi equivalent dose in first 48h)
    norepi = pd.read_csv(MIM / "norepinephrine_equivalent_dose.csv",
                         usecols=["stay_id", "starttime"])
    norepi["starttime"] = pd.to_datetime(norepi["starttime"], dayfirst=True, errors="coerce")
    norepi = norepi.merge(det2, on="stay_id", how="left")
    norepi["hour"] = (norepi["starttime"] - norepi["icu_intime"]).dt.total_seconds() / 3600
    norepi_48 = norepi[(norepi["hour"] >= 0) & (norepi["hour"] <= 48)]
    vaso_flag = norepi_48.groupby("stay_id").size().rename("vasopressor").reset_index()
    vaso_flag["vasopressor"] = (vaso_flag["vasopressor"] > 0).astype(int)

    # RRT
    rrt = pd.read_csv(MIM / "rrt.csv", usecols=["stay_id", "charttime", "dialysis_present"])
    rrt["charttime"] = pd.to_datetime(rrt["charttime"], dayfirst=True, errors="coerce")
    rrt = rrt.merge(det2, on="stay_id", how="left")
    rrt["hour"] = (rrt["charttime"] - rrt["icu_intime"]).dt.total_seconds() / 3600
    rrt_48 = rrt[(rrt["hour"] >= 0) & (rrt["hour"] <= 48) & (rrt["dialysis_present"] == 1)]
    rrt_flag = rrt_48.groupby("stay_id").size().rename("rrt").reset_index()
    rrt_flag["rrt"] = (rrt_flag["rrt"] > 0).astype(int)

    # Merge all
    df = det[["stay_id", "subject_id", "hadm_id", "age", "sex", "race_grouped", "hospital_expire_flag", "los_icu", "bmi"]] \
        .merge(sofa, on="stay_id", how="left") \
        .merge(apsiii, on="stay_id", how="left") \
        .merge(charl, on=["subject_id", "hadm_id"], how="left") \
        .merge(sep, on="stay_id", how="left") \
        .merge(vit, on="stay_id", how="left") \
        .merge(lab, on="stay_id", how="left") \
        .merge(vs_agg[["stay_id", "peep", "plateau_pressure", "driving_pressure", "vt_pbw", "fio2_vent"]], on="stay_id", how="left") \
        .merge(vaso_flag, on="stay_id", how="left") \
        .merge(rrt_flag, on="stay_id", how="left")
    df["sepsis3"] = df["sepsis3"].fillna(0)
    df["vasopressor"] = df["vasopressor"].fillna(0).astype(int)
    df["rrt"] = df["rrt"].fillna(0).astype(int)
    df = df[df["stay_id"].isin(stay_ids)].copy()
    return df


# ---------------------------------------------------------------- eICU covariates
def build_eicu_covariates(stay_ids):
    det = pd.read_csv(EICU / "icustay_detail.csv")
    det = det.rename(columns={"patientunitstayid": "stay_id", "apache_iv": "apache_iv",
                              "hosp_mort": "hospital_expire_flag", "ethnicity": "race_raw"})
    det["age"] = pd.to_numeric(det["age"], errors="coerce")
    g = pd.to_numeric(det["gender"], errors="coerce")
    det["sex"] = np.where(g.eq(1), 1, np.where(g.eq(0), 0, np.nan))
    det["race_grouped"] = det["race_raw"].apply(group_race_eicu)
    det["hospital_expire_flag"] = pd.to_numeric(det["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    det["bmi"] = det["admissionweight"] / (det["admissionheight"] / 100) ** 2
    det.loc[(det["bmi"] < 12) | (det["bmi"] > 80), "bmi"] = np.nan

    charl = pd.read_csv(EICU / "charlson.csv")
    charl = charl.rename(columns={"patientunitstayid": "stay_id", "final_charlson_score": "charlson",
                                  "copd1": "copd", "chf1": "chf"})
    charl = charl[["stay_id", "charlson", "copd", "chf"]].copy()
    charl["charlson"] = pd.to_numeric(charl["charlson"], errors="coerce")

    adm = pd.read_csv(EICU / "admissiondx.csv", usecols=["patientunitstayid", "admitdxname"])
    sep_ids = adm[adm["admitdxname"].str.contains("sepsis|Sepsis", case=False, na=False)]["patientunitstayid"].unique()
    det["sepsis3"] = det["stay_id"].isin(sep_ids).astype(int)

    # Labs first day (add lactate)
    lab = pd.read_csv(EICU / "labsfirstday.csv",
                      usecols=["patientunitstayid", "bun_max", "creatinine_max", "bilirubin_max",
                               "albumin_min", "wbc_max", "bicarbonate_min", "lactate_max"])
    lab = lab.rename(columns={"patientunitstayid": "stay_id", "bun_max": "bun",
                              "creatinine_max": "creatinine", "bilirubin_max": "bilirubin",
                              "albumin_min": "albumin", "wbc_max": "wbc",
                              "bicarbonate_min": "bicarbonate", "lactate_max": "lactate"})

    # Vitals: use apacheapsvar (has HR, MAP, temp, RR, pH, etc at admission)
    aps = pd.read_csv(EICU / "apacheapsvar.csv",
                      usecols=["patientunitstayid", "heartrate", "meanbp", "temperature", "respiratoryrate"])
    aps = aps.rename(columns={"patientunitstayid": "stay_id", "heartrate": "hr",
                              "meanbp": "map", "temperature": "temp",
                              "respiratoryrate": "rr_vitalsign"})

    # Vent params from respiratorycharting (0-48h)
    target_peep = {"PEEP", "PEEP/CPAP"}
    target_plat = {"Plateau Pressure"}
    target_vt = {"Tidal Volume (set)", "Set Vt", "Adult Con Setting Set Vt"}
    chunks = []
    for chunk in pd.read_csv(EICU / "respiratorycharting.csv",
                             usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"],
                             dtype={"respchartvalue": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(stay_ids)]
        if chunk.empty: continue
        chunk["hour"] = pd.to_numeric(chunk["respchartoffset"], errors="coerce") / 60
        chunk = chunk[(chunk["hour"] >= 0) & (chunk["hour"] <= 48)]
        chunks.append(chunk)
    if chunks:
        rc = pd.concat(chunks, ignore_index=True)
        rc["val"] = pd.to_numeric(rc["respchartvalue"], errors="coerce")
        peep = rc[rc["respchartvaluelabel"].isin(target_peep)].groupby("patientunitstayid")["val"].mean().rename("peep").reset_index()
        plat = rc[rc["respchartvaluelabel"].isin(target_plat)].groupby("patientunitstayid")["val"].mean().rename("plateau_pressure").reset_index()
        vt = rc[rc["respchartvaluelabel"].isin(target_vt)].groupby("patientunitstayid")["val"].mean().rename("vt_set").reset_index()
        vent = peep.rename(columns={"patientunitstayid":"stay_id"}) \
            .merge(plat.rename(columns={"patientunitstayid":"stay_id"}), on="stay_id", how="outer") \
            .merge(vt.rename(columns={"patientunitstayid":"stay_id"}), on="stay_id", how="outer")
        vent["driving_pressure"] = vent["plateau_pressure"] - vent["peep"]
        # Vt/PBW
        vent = vent.merge(det[["stay_id", "admissionheight", "gender"]], on="stay_id", how="left")
        h = pd.to_numeric(vent["admissionheight"], errors="coerce")
        g2 = pd.to_numeric(vent["gender"], errors="coerce")
        pbw = np.where(g2.eq(1), 50 + 0.91 * (h - 152.4),
                       np.where(g2.eq(0), 45.5 + 0.91 * (h - 152.4), np.nan))
        vent["vt_pbw"] = vent["vt_set"] / pbw
        vent = vent.drop(columns=["admissionheight", "gender"])
    else:
        vent = pd.DataFrame(columns=["stay_id", "peep", "plateau_pressure", "driving_pressure", "vt_pbw"])

    # Vasopressor + RRT
    inf = pd.read_csv(EICU / "infusiondrug.csv", usecols=["patientunitstayid", "drugname"])
    inf = inf.rename(columns={"patientunitstayid": "stay_id"})
    vaso_ids = inf[inf["drugname"].str.contains("norepinephrine|epinephrine|dopamine|phenylephrine|vasopressin", case=False, na=False)]["stay_id"].unique()
    det["vasopressor"] = det["stay_id"].isin(vaso_ids).astype(int)
    aps_dial = pd.read_csv(EICU / "apacheapsvar.csv", usecols=["patientunitstayid", "dialysis"])
    aps_dial = aps_dial.rename(columns={"patientunitstayid": "stay_id"}).drop_duplicates("stay_id")
    det = det.merge(aps_dial, on="stay_id", how="left")
    det["rrt"] = pd.to_numeric(det["dialysis"], errors="coerce").fillna(0).astype(int)
    det = det.drop(columns=["dialysis"])

    df = det[["stay_id", "age", "sex", "race_grouped", "hospital_expire_flag", "apache_iv", "bmi",
              "sepsis3", "vasopressor", "rrt", "icu_los_hours"]] \
        .merge(charl, on="stay_id", how="left") \
        .merge(lab, on="stay_id", how="left") \
        .merge(aps, on="stay_id", how="left") \
        .merge(vent, on="stay_id", how="left")
    df = df[df["stay_id"].isin(stay_ids)].copy()
    return df


# ---------------------------------------------------------------- model ladder
def impute_scale(train, vars_, test=None):
    X = train[vars_].apply(pd.to_numeric, errors="coerce")
    med = X.median()
    med = med.fillna(0)
    X = X.fillna(med)
    sc = StandardScaler().fit(X)
    Xs = pd.DataFrame(sc.transform(X), columns=vars_, index=X.index)
    if test is not None:
        Xt = test[vars_].apply(pd.to_numeric, errors="coerce").fillna(med)
        Xts = pd.DataFrame(sc.transform(Xt), columns=vars_, index=Xt.index)
        return Xs, Xts, med, sc
    return Xs, None, med, sc


def metrics(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    return {"auroc": round(roc_auc_score(y, p), 4) if len(np.unique(y)) > 1 else np.nan,
            "brier": round(brier_score_loss(y, p), 4)}


def signif(p):
    if p is None or np.isnan(p): return np.nan
    if p < 0.001: return f"{p:.2e}"
    return round(p, 4)


def run_model_ladder(mimic, eicu, log):
    """New model ladder with severity + vent params."""
    # Aligned across DBs (no SOFA/APACHE)
    base = ["age", "sex", "charlson", "sepsis3"]
    severity_aligned = []  # none (SOFA/APACHE not comparable)
    vent_aligned = ["peep", "driving_pressure", "vt_pbw"]
    labs_aligned = ["bun", "creatinine", "bilirubin", "albumin", "wbc", "bicarbonate", "hr", "map", "lactate", "temp"]
    vr_single = ["vr_initial", "vr_max", "vr_mean"]
    vr_dyn = ["vr_slope", "vr_delta", "vr_auc"]
    pf_feat = ["pf_initial", "pf_min", "pf_mean", "pf_slope"]

    # Cross-DB validated models
    models = {
        "M1_base": base,
        "M2_base_plus_vent_labs": base + vent_aligned + labs_aligned,
        "M3_M2_plus_VR": base + vent_aligned + labs_aligned + vr_single + vr_dyn,
        "M4_M3_plus_PF": base + vent_aligned + labs_aligned + vr_single + vr_dyn + pf_feat,
    }

    # MIMIC-internal models with SOFA (key analysis!)
    mimic_models = {
        "M5_mimic_SOFA_only": ["sofa"],
        "M6_mimic_base_plus_SOFA": base + ["sofa"],
        "M7_mimic_base_plus_SOFA_plus_vent_labs": base + ["sofa"] + vent_aligned + labs_aligned,
        "M8_mimic_M7_plus_VR": base + ["sofa"] + vent_aligned + labs_aligned + vr_single + vr_dyn,
        "M9_mimic_M8_plus_PF": base + ["sofa"] + vent_aligned + labs_aligned + vr_single + vr_dyn + pf_feat,
    }

    y_m = mimic["hospital_expire_flag"].astype(int).values
    y_e = eicu["hospital_expire_flag"].astype(int).values

    perf_rows = []
    or_rows = []
    all_probs = {}

    # Cross-DB models
    for name, vars_ in models.items():
        log.append(f"[{now()}] Fitting {name} (cross-DB)...")
        X_tr, X_te, med, sc = impute_scale(mimic, vars_, eicu)
        lr = LogisticRegression(penalty=None, max_iter=2000, random_state=RNG).fit(X_tr.values, y_m)
        p_m = lr.predict_proba(X_tr.values)[:, 1]
        # eICU with intercept recalibration
        p_e_raw = lr.predict_proba(X_te.values)[:, 1]
        logit_e = np.log(np.clip(p_e_raw, 1e-6, 1-1e-6) / (1-np.clip(p_e_raw, 1e-6, 1-1e-6)))
        logit_e = logit_e + (np.log(y_e.mean() / (1-y_e.mean())) - logit_e.mean())
        p_e = 1 / (1 + np.exp(-logit_e))
        m_m = metrics(y_m, p_m); m_e = metrics(y_e, p_e)
        # CV
        try:
            p_cv = cross_val_predict(LogisticRegression(penalty=None, max_iter=2000, random_state=RNG),
                                     X_tr.values, y_m, cv=5, method="predict_proba")[:, 1]
            m_cv = metrics(y_m, p_cv)
        except: m_cv = {"auroc": np.nan, "brier": np.nan}
        perf_rows.append({"model": name, "type": "cross_db", "n_vars": len(vars_),
                          "mimic_auroc": m_m["auroc"], "mimic_brier": m_m["brier"],
                          "mimic_cv_auroc": m_cv["auroc"],
                          "eicu_auroc": m_e["auroc"], "eicu_brier": m_e["brier"]})
        all_probs[name] = {"m": p_m, "e": p_e}
        # OR from statsmodels
        X_sm = sm.add_constant(X_tr)
        sm_m = sm.Logit(y_m, X_sm.values).fit(disp=0, maxiter=200)
        for i, v in enumerate(vars_):
            or_rows.append({"model": name, "variable": v,
                            "OR_per_SD": round(np.exp(sm_m.params[i+1]), 4),
                            "CI_lower": round(np.exp(sm_m.params[i+1] - 1.96*sm_m.bse[i+1]), 4),
                            "CI_upper": round(np.exp(sm_m.params[i+1] + 1.96*sm_m.bse[i+1]), 4),
                            "p_value": signif(sm_m.pvalues[i+1])})
        log.append(f"  {name}: MIMIC AUROC={m_m['auroc']} eICU AUROC={m_e['auroc']}")

    # MIMIC-internal models with SOFA
    for name, vars_ in mimic_models.items():
        log.append(f"[{now()}] Fitting {name} (MIMIC-internal with SOFA)...")
        X_tr, _, med, sc = impute_scale(mimic, vars_)
        lr = LogisticRegression(penalty=None, max_iter=2000, random_state=RNG).fit(X_tr.values, y_m)
        p_m = lr.predict_proba(X_tr.values)[:, 1]
        m_m = metrics(y_m, p_m)
        try:
            p_cv = cross_val_predict(LogisticRegression(penalty=None, max_iter=2000, random_state=RNG),
                                     X_tr.values, y_m, cv=5, method="predict_proba")[:, 1]
            m_cv = metrics(y_m, p_cv)
        except: m_cv = {"auroc": np.nan, "brier": np.nan}
        perf_rows.append({"model": name, "type": "mimic_internal", "n_vars": len(vars_),
                          "mimic_auroc": m_m["auroc"], "mimic_brier": m_m["brier"],
                          "mimic_cv_auroc": m_cv["auroc"],
                          "eicu_auroc": np.nan, "eicu_brier": np.nan})
        all_probs[name] = {"m": p_m, "e": None}
        X_sm = sm.add_constant(X_tr)
        sm_m = sm.Logit(y_m, X_sm.values).fit(disp=0, maxiter=200)
        for i, v in enumerate(vars_):
            or_rows.append({"model": name, "variable": v,
                            "OR_per_SD": round(np.exp(sm_m.params[i+1]), 4),
                            "CI_lower": round(np.exp(sm_m.params[i+1] - 1.96*sm_m.bse[i+1]), 4),
                            "CI_upper": round(np.exp(sm_m.params[i+1] + 1.96*sm_m.bse[i+1]), 4),
                            "p_value": signif(sm_m.pvalues[i+1])})
        log.append(f"  {name}: MIMIC AUROC={m_m['auroc']} CV={m_cv['auroc']}")

    return pd.DataFrame(perf_rows), pd.DataFrame(or_rows), all_probs


# ---------------------------------------------------------------- DeLong + IDI/NRI
def delong_test(y, a, b):
    y = np.asarray(y).astype(int); a = np.asarray(a); b = np.asarray(b)
    x1a, x0a = a[y==1], a[y==0]; x1b, x0b = b[y==1], b[y==0]
    n1, n0 = len(x1a), len(x0a)
    if n1 < 2 or n0 < 2: return np.nan, np.nan, np.nan, np.nan
    def struct(x1, x0):
        v10 = np.array([((xi > x0).sum() + 0.5*(xi == x0).sum())/n0 for xi in x1])
        v01 = np.array([((x1 > x0j).sum() + 0.5*(x1 == x0j).sum())/n1 for x0j in x0])
        return v10, v01
    v10a, v01a = struct(x1a, x0a); v10b, v01b = struct(x1b, x0b)
    auc_a, auc_b = v10a.mean(), v10b.mean()
    var_a = np.var(v10a, ddof=1)/n1 + np.var(v01a, ddof=1)/n0
    var_b = np.var(v10b, ddof=1)/n1 + np.var(v01b, ddof=1)/n0
    cov = np.cov(v10a, v10b, ddof=1)[0,1]/n1 + np.cov(v01a, v01b, ddof=1)[0,1]/n0
    vd = var_a + var_b - 2*cov
    if vd <= 0: return 0, 1, auc_a, auc_b
    z = (auc_a - auc_b)/np.sqrt(vd)
    return z, 2*(1-stats.norm.cdf(abs(z))), auc_a, auc_b


def idi_nri(y, new, old):
    y = np.asarray(y).astype(int); new = np.asarray(new); old = np.asarray(old)
    idi = ((new[y==1].mean() - new[y==0].mean()) - (old[y==1].mean() - old[y==0].mean()))
    diff = new - old
    nri = ((diff[y==1] > 0).sum() - (diff[y==1] < 0).sum())/max(y.sum(),1) + \
          ((diff[y==0] < 0).sum() - (diff[y==0] > 0).sum())/max((len(y)-y.sum()),1)
    return {"IDI": round(idi, 4), "NRI": round(nri, 4)}


# ---------------------------------------------------------------- main
def aggregate_features_from_long(long_csv, id_col="stay_id"):
    """Aggregate VR/PF features from 48h long-format data (≥2 points)."""
    long = pd.read_csv(long_csv)
    rows = []
    for sid, sub in long.groupby(id_col):
        sub = sub.sort_values("hour")
        vr = sub["vr"].dropna()
        pf = sub["pf"].dropna()
        rec = {id_col: sid, "n_points": len(sub)}
        # VR features (only initial + delta for reduced model)
        if len(vr) > 0:
            rec["vr_initial"] = float(vr.iloc[0])
            rec["vr_max"] = float(vr.max())
            rec["vr_mean"] = float(vr.mean())
            rec["vr_last"] = float(vr.iloc[-1])
            if len(vr) >= 2:
                rec["vr_delta"] = float(vr.iloc[-1] - vr.iloc[0])
                dt = sub.dropna(subset=["vr"])["hour"].iloc[-1] - sub.dropna(subset=["vr"])["hour"].iloc[0]
                rec["vr_slope"] = float(rec["vr_delta"] / dt) if dt > 0 else np.nan
                rec["vr_auc"] = float(np.trapz(vr.values, sub.dropna(subset=["vr"])["hour"].values) / max(dt, 1e-6))
            else:
                rec["vr_delta"] = 0.0
                rec["vr_slope"] = np.nan
                rec["vr_auc"] = float(vr.iloc[0])
        # PF features
        if len(pf) > 0:
            rec["pf_initial"] = float(pf.iloc[0])
            rec["pf_min"] = float(pf.min())
            rec["pf_mean"] = float(pf.mean())
            if len(pf) >= 2:
                dt = sub.dropna(subset=["pf"])["hour"].iloc[-1] - sub.dropna(subset=["pf"])["hour"].iloc[0]
                rec["pf_slope"] = float((pf.iloc[-1] - pf.iloc[0]) / dt) if dt > 0 else np.nan
            else:
                rec["pf_slope"] = np.nan
        # other
        for c in ["paco2_initial", "ph_initial"]:
            pass
        if "pco2" in sub.columns:
            rec["paco2_initial"] = float(sub["pco2"].iloc[0]) if pd.notna(sub["pco2"].iloc[0]) else np.nan
        elif "paco2" in sub.columns:
            rec["paco2_initial"] = float(sub["paco2"].iloc[0]) if pd.notna(sub["paco2"].iloc[0]) else np.nan
        if "ph" in sub.columns:
            rec["ph_initial"] = float(sub["ph"].iloc[0]) if pd.notna(sub["ph"].iloc[0]) else np.nan
        # outcomes
        for c in ["hospital_expire_flag", "vent_7d", "icu_los_ge7d", "los_icu_days"]:
            if c in long.columns:
                rec[c] = sub[c].iloc[0] if pd.notna(sub[c].iloc[0]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

    # Load features from 48h ≥2 points long data
    m_feat = aggregate_features_from_long(OUT / "trajectory_mimic_48h.csv", "stay_id")
    e_feat = aggregate_features_from_long(OUT / "trajectory_eicu_48h.csv", "stay_id")
    log.append(f"[{now()}] Loaded features from 48h ≥2pts: MIMIC={len(m_feat)}, eICU={len(e_feat)}")

    # Build enhanced covariates
    log.append(f"[{now()}] Building MIMIC enhanced covariates...")
    m_cov = build_mimic_covariates(set(m_feat["stay_id"]))
    log.append(f"[{now()}] MIMIC cov: {len(m_cov)} stays, SOFA missing={m_cov['sofa'].isna().sum()}")

    log.append(f"[{now()}] Building eICU enhanced covariates...")
    e_cov = build_eicu_covariates(set(e_feat["stay_id"]))
    log.append(f"[{now()}] eICU cov: {len(e_cov)} stays, APACHE missing={e_cov['apache_iv'].isna().sum()}")

    # Merge features + covariates
    mimic = m_feat.merge(m_cov.drop(columns=["hospital_expire_flag"], errors="ignore"), on="stay_id", how="left", suffixes=("", "_cov"))
    eicu = e_feat.merge(e_cov.drop(columns=["hospital_expire_flag"], errors="ignore"), on="stay_id", how="left", suffixes=("", "_cov"))
    # prefer original hospital_expire_flag
    for c in ["age", "sex", "charlson", "sepsis3"]:
        if f"{c}_cov" in mimic.columns:
            mimic[c] = mimic[c].fillna(mimic[f"{c}_cov"])
            mimic = mimic.drop(columns=[f"{c}_cov"])
            eicu[c] = eicu[c].fillna(eicu[f"{c}_cov"])
            eicu = eicu.drop(columns=[f"{c}_cov"])
    mimic = mimic.dropna(subset=["hospital_expire_flag"]).copy()
    eicu = eicu.dropna(subset=["hospital_expire_flag"]).copy()
    mimic["hospital_expire_flag"] = mimic["hospital_expire_flag"].astype(int)
    eicu["hospital_expire_flag"] = eicu["hospital_expire_flag"].astype(int)

    mimic.to_csv(RES / "mimic_enhanced.csv", index=False)
    eicu.to_csv(RES / "eicu_enhanced.csv", index=False)
    log.append(f"[{now()}] Saved enhanced data: MIMIC={len(mimic)}, eICU={len(eicu)}")

    # Table 1 with grouped race
    log.append(f"[{now()}] Building Table 1 with grouped race...")
    def table1(df, db):
        rows = []
        df = df.copy(); df["mort"] = df["hospital_expire_flag"].astype(int)
        alive = df[df["mort"]==0]; dead = df[df["mort"]==1]
        cont = ["age","bmi","sofa" if "sofa" in df.columns else "apache_iv","charlson",
                "vr_initial","vr_max","vr_mean","vr_delta","vr_slope",
                "pf_initial","pf_min","pf_mean","paco2_initial","ph_initial",
                "peep","driving_pressure","vt_pbw","plateau_pressure",
                "hr","map","temp","lactate","bun","creatinine","bilirubin","albumin","wbc","bicarbonate","n_points"]
        for v in cont:
            if v not in df.columns: continue
            a = pd.to_numeric(alive[v], errors="coerce").dropna()
            d = pd.to_numeric(dead[v], errors="coerce").dropna()
            if len(a)>0 and len(d)>0:
                _, p = stats.ttest_ind(a, d, equal_var=False)
                if np.isnan(p): _, p = stats.mannwhitneyu(a, d)
            else: p = np.nan
            rows.append({"db":db,"variable":v,"type":"cont",
                         "alive":f"{a.mean():.2f} ± {a.std():.2f}" if len(a) else "",
                         "dead":f"{d.mean():.2f} ± {d.std():.2f}" if len(d) else "",
                         "p":signif(p)})
        cat = ["sex","sepsis3","vasopressor","rrt","race_grouped"]
        for v in cat:
            if v not in df.columns: continue
            ct = pd.crosstab(df[v].fillna("NA"), df["mort"])
            if ct.shape[0]>1 and ct.shape[1]>1:
                _, p, _, _ = stats.chi2_contingency(ct)
            else: p = np.nan
            for cat_v in df[v].dropna().unique():
                na = ((df[v]==cat_v)&(df["mort"]==0)).sum()
                nd = ((df[v]==cat_v)&(df["mort"]==1)).sum()
                rows.append({"db":db,"variable":v,"type":"cat","category":str(cat_v),
                             "alive_n":int(na),"alive_pct":f"{na/max(1,len(alive))*100:.1f}%",
                             "dead_n":int(nd),"dead_pct":f"{nd/max(1,len(dead))*100:.1f}%","p":signif(p)})
        return pd.DataFrame(rows)
    t1m = table1(mimic, "MIMIC-IV"); t1e = table1(eicu, "eICU")
    t1m.to_csv(RES / "table1_mimic.csv", index=False)
    t1e.to_csv(RES / "table1_eicu.csv", index=False)
    log.append(f"[{now()}] Table 1 saved.")

    # Model ladder
    log.append(f"[{now()}] Running model ladder...")
    perf, or_df, probs = run_model_ladder(mimic, eicu, log)
    perf.to_csv(RES / "model_ladder_results.csv", index=False)
    or_df.to_csv(RES / "or_results.csv", index=False)

    # DeLong + IDI/NRI for key comparisons
    log.append(f"[{now()}] DeLong + IDI/NRI...")
    y_m = mimic["hospital_expire_flag"].astype(int).values
    y_e = eicu["hospital_expire_flag"].astype(int).values
    dl_rows = []; idi_rows = []
    # MIMIC internal: M7 (base+SOFA+vent+labs) vs M8 (+VR) vs M9 (+VR+PF)
    for (a, b, label) in [("M7_mimic_base_plus_SOFA_plus_vent_labs", "M8_mimic_M7_plus_VR", "VR over SOFA+vent+labs"),
                          ("M8_mimic_M7_plus_VR", "M9_mimic_M8_plus_PF", "PF over VR+SOFA"),
                          ("M6_mimic_base_plus_SOFA", "M8_mimic_M7_plus_VR", "VR over base+SOFA")]:
        if a in probs and b in probs:
            z, p, auc_a, auc_b = delong_test(y_m, probs[a]["m"], probs[b]["m"])
            dl_rows.append({"db":"MIMIC","comparison":label,"auc_a":round(auc_a,4),"auc_b":round(auc_b,4),"z":round(z,3),"p":signif(p)})
            r = idi_nri(y_m, probs[b]["m"], probs[a]["m"])
            idi_rows.append({"db":"MIMIC","comparison":label,"IDI":r["IDI"],"NRI":r["NRI"]})
    # Cross-DB: M2 vs M3 (VR over base+vent+labs)
    for (a, b, label) in [("M2_base_plus_vent_labs", "M3_M2_plus_VR", "VR over base+vent+labs (cross-DB)"),
                          ("M3_M2_plus_VR", "M4_M3_plus_PF", "PF over VR (cross-DB)")]:
        if a in probs and b in probs and probs[a]["e"] is not None and probs[b]["e"] is not None:
            z, p, auc_a, auc_b = delong_test(y_e, probs[a]["e"], probs[b]["e"])
            dl_rows.append({"db":"eICU","comparison":label,"auc_a":round(auc_a,4),"auc_b":round(auc_b,4),"z":round(z,3),"p":signif(p)})
            r = idi_nri(y_e, probs[b]["e"], probs[a]["e"])
            idi_rows.append({"db":"eICU","comparison":label,"IDI":r["IDI"],"NRI":r["NRI"]})
    pd.DataFrame(dl_rows).to_csv(RES / "delong_results.csv", index=False)
    pd.DataFrame(idi_rows).to_csv(RES / "idi_nri_results.csv", index=False)

    log.append(f"[{now()}] Done.")
    (RES / "log.md").write_text("\n".join(log), encoding="utf-8")

    print("\n=== Model ladder ===")
    print(perf.to_string(index=False))
    print("\n=== DeLong ===")
    print(pd.read_csv(RES/"delong_results.csv").to_string(index=False))
    print("\n=== IDI/NRI ===")
    print(pd.read_csv(RES/"idi_nri_results.csv").to_string(index=False))
    print("\n=== Key OR: M8 (base+SOFA+vent+labs+VR) ===")
    m8 = or_df[or_df["model"]=="M8_mimic_M7_plus_VR"]
    print(m8[["variable","OR_per_SD","CI_lower","CI_upper","p_value"]].to_string(index=False))
    print(f"\nOutputs in {RES}")


if __name__ == "__main__":
    main()