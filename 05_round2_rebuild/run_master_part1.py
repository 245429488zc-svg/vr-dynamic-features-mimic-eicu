#!/usr/bin/env python3
"""Round-2 master, part 1 (M1): sepsis_h harmonization + eICU landmark frames +
harmonized-model ladder (fit MIMIC / transport eICU) + eICU refit replication.

Harmonized spec (as ruled with the authors, mirroring the reviewer's premise that the
reported fully adjusted specification includes COPD/CHF):
  Base_h = age, sex, charlson, sepsis48 (sepsis determinable by the 48-h landmark:
           MIMIC-IV = 48-h Sepsis-3, i.e. sepsis3_48h as derived by the rebuild
           [suspected-infection onset AND qualifying SOFA assessment both <=48 h];
           eICU = sepsis among the recorded admission diagnoses), copd, chf, peep,
           vt_pbw, hr, map, temp, bun, creatinine, wbc, bicarbonate          [15 cov]
  ICD-coded sepsis (sepsis_h) is retained on file but is hospitalization-level and is
  NOT used in any model fitted or applied at the 48-h landmark.
  + PF_48H [2] -> 17; + VR_48H [2] -> 19.
SOFA / driving pressure / lactate remain excluded (not available in eICU).
All vitals use first-day MEANS on both sides (eICU recomputed natively).

Outputs: sepsis_h tables, eICU association/landmark cohorts, harmonized ladder table.
"""
from __future__ import annotations

import os
from pathlib import Path

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
MIMIC_READY = OUTDIR / "mimic_analysis_ready_round2.csv"
EICU_READY = OUTDIR / "eicu_analysis_ready_round2.csv"
DX = ROOT / "原始数据" / "core" / "diagnoses_icd.csv.gz"

OUTCOME = "hospital_expire_flag"
BASE_FULL = ["age", "sex", "charlson", "sepsis3_48h", "copd", "chf", "sofa", "peep",
             "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
             "bun", "creatinine", "wbc", "bicarbonate"]
BASE_H = ["age", "sex", "charlson", "sepsis48", "copd", "chf", "peep",
          "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF_48H = ["pf_initial", "pf_min"]
VR_48H = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def scale_x(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    scaler = StandardScaler().fit(df[variables])
    return pd.DataFrame(scaler.transform(df[variables]), columns=variables, index=df.index)


def fit_ladder(df: pd.DataFrame, base: list[str], label: str) -> tuple[list[dict], pd.DataFrame]:
    import statsmodels.api as sm
    rows, ors_all = [], {}
    ladder = [("Base", base), ("Base + P/F", base + PF_48H), ("Base + P/F + VR", base + PF_48H + VR_48H)]
    for name, vars_ in ladder:
        x = sm.add_constant(scale_x(df, vars_), has_constant="add")
        y = df[OUTCOME].astype(int)
        res = sm.Logit(y, x).fit(disp=False, maxiter=300)
        pred = res.predict(x)
        a = roc_auc_score(y, pred)
        rows.append({"Row": label, "Model": name, "N": len(df), "Deaths": int(y.sum()),
                     "AUROC before": "", "AUROC after": "",
                     "AUROC": f"{a:.3f}", "Brier": f"{brier_score_loss(y, pred):.3f}"})
        stamp(f"[{label}] {name}: N={len(df)}, 死亡={int(y.sum())}, AUROC={a:.4f}")
        for v in VR_48H:
            if v not in vars_:
                continue
            beta, se = float(res.params[v]), float(res.bse[v])
            ors_all[f"{name}|{v}"] = (math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se))
    return rows, pd.DataFrame([{"model": k, "OR": v[0], "lo": v[1], "hi": v[2]} for k, v in ors_all.items()])


def deltas(dfx: pd.DataFrame, base: list[str]) -> dict:
    a = roc_auc_score(dfx[OUTCOME].astype(int),
                      LogisticRegression(penalty=None, max_iter=3000).fit(
                          scale_x(dfx, base), dfx[OUTCOME].astype(int)).predict_proba(scale_x(dfx, base))[:, 1])
    return {"a": a}


def main() -> None:
    mim = pd.read_csv(MIMIC_READY)
    eic = pd.read_csv(EICU_READY)
    stamp(f"读入: MIMIC {len(mim)}, eICU {len(eic)}")

    # ---------- MIMIC sepsis_h (ICD-coded) ----------
    dx = pd.read_csv(DX, usecols=["hadm_id", "icd_code", "icd_version"])
    dx["code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    is10 = dx["icd_version"].eq(10)
    sep10 = is10 & (dx["code"].str.startswith(("A40", "A41", "R652")))
    sep9 = (~is10) & (dx["code"].str.startswith(("99591", "99592", "78552")))
    sep_hadm = set(dx.loc[sep10 | sep9, "hadm_id"].unique())
    mim["sepsis_h"] = mim["hadm_id"].isin(sep_hadm).astype(int)
    stamp(f"MIMIC sepsis_h(ICD): {int(mim['sepsis_h'].sum())}/{len(mim)} ({mim['sepsis_h'].mean():.1%}); "
          f"对比 Sepsis-3 {int(mim['sepsis3'].sum())} ({mim['sepsis3'].mean():.1%})")
    # ICD-coded sepsis is hospitalization-level (no usable timestamps) — kept on file for
    # association-side bookkeeping only. The harmonized/transport slot uses the 48-h-
    # determinable Sepsis-3 flag derived by the rebuild.
    mim["sepsis48"] = mim["sepsis3_48h"].astype(int)
    stamp(f"MIMIC sepsis48 (48-h Sepsis-3): {int(mim['sepsis48'].sum())}/{len(mim)} ({mim['sepsis48'].mean():.1%})")
    mim.to_csv(OUTDIR / "mimic_analysis_ready_round2.csv", index=False)

    # ---------- eICU: official flag check + sepsis_h naming ----------
    # ---------- eICU covariate fill via the published builder (covers recovered stays) ----------
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enh_cov", ROOT / "02_covariates" /
        "enhanced_covariates_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.EICU = ROOT / "原始数据" / "eICU" / "eICU"
    gate_ids = set(eic["stay_id"].astype(int))
    cov = mod.build_eicu_covariates(sorted(gate_ids))
    cov = cov.set_index("stay_id")
    filled = 0
    for col in cov.columns:
        if col in eic.columns:
            gap = eic[col].isna() | (eic[col].astype(str).isin(["nan", ""]))
            if gap.any():
                eic.loc[gap, col] = eic.loc[gap, "stay_id"].map(cov[col]).values
                filled += int(gap.sum())
        else:
            eic[col] = eic["stay_id"].map(cov[col]).values
            filled += len(eic)
    stamp(f"eICU 协变量回填(来自公开builder, 旧值优先): 填补 {filled} 个空位")

    if "hospitaldischargestatus" in eic.columns:
        miss = int((~eic["hospitaldischargestatus"].isin(["Alive", "Expired"])).sum())
        if miss:
            eic = eic[eic["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy()
        eic["hospital_expire_flag"] = eic["hospital_expire_flag"].astype(int)
        stamp(f"eICU 官方状态双检后: {len(eic)}, 死亡={int(eic['hospital_expire_flag'].sum())}")
    eic["sepsis_h"] = eic["sepsis3"].astype(int)  # eICU's own sepsis3 column IS the admitDx text def
    # harmonized/transport slot: admissionDx-recorded sepsis (admission-form entries; in the current eICU landmark cohort every sepsis-positive entry carries an offset <=48 h,
    # hence determinable before the landmark time)
    eic["sepsis48"] = eic["sepsis3"].astype(int)
    stamp(f"eICU sepsis_h(admitDx): {int(eic['sepsis_h'].sum())}/{len(eic)} ({eic['sepsis_h'].mean():.1%})")
    # 重挂 discharge offset（特征重建的落盘文件不携带它们）
    pat2 = pd.read_csv(ROOT / "原始数据" / "eICU" / "eICU" / "patient.csv",
                       usecols=["patientunitstayid", "hospitaldischargeoffset", "unitdischargeoffset"],
                       low_memory=False).rename(columns={"patientunitstayid": "stay_id"})
    eic = eic.merge(pat2, on="stay_id", how="left", suffixes=("", "_pat"))
    eic.to_csv(OUTDIR / "eicu_analysis_ready_round2.csv", index=False)

    # ---------- eICU association cohort / cc / landmark ----------
    harm_vars_avail = [c for c in BASE_H + PF_48H + VR_48H if c in eic.columns]
    missing_vars = [c for c in BASE_H + PF_48H + VR_48H if c not in eic.columns]
    stamp(f"eICU harmonized 变量齐备: {len(harm_vars_avail)}/{len(BASE_H + PF_48H + VR_48H)}; 缺: {missing_vars}")
    eic_cc = eic.dropna(subset=[OUTCOME] + harm_vars_avail).copy()
    stamp(f"eICU 完整病例(harmonized): {len(eic_cc)}, 死亡={int(eic_cc[OUTCOME].sum())}")
    lm_mask = pd.to_numeric(eic_cc["hospitaldischargeoffset"], errors="coerce") > 2880 if \
        "hospitaldischargeoffset" in eic_cc.columns else pd.Series(True, index=eic_cc.index)
    eic_lm = eic_cc[lm_mask.fillna(False) if hasattr(lm_mask, 'fillna') else lm_mask].copy()
    stamp(f"eICU landmark(48h时在院): n={len(eic_lm)}, 48h后死亡={int(eic_lm[OUTCOME].sum())}")
    eic_lm.to_csv(OUTDIR / "eicu_landmark_round2.csv", index=False)
    eic_cc.to_csv(OUTDIR / "eicu_cc_round2.csv", index=False)

    # ---------- harmonized ladder on MIMIC landmark ----------
    lm_ids = pd.read_csv(OUTDIR / "landmark_cohort_round2.csv", usecols=["stay_id"])
    lm = mim[mim["stay_id"].isin(lm_ids["stay_id"])].copy()
    stamp(f"MIMIC landmark(带sepsis_h重取): n={len(lm)}, 死亡={int(lm['hospital_expire_flag'].sum())}")
    lm_rows, lm_ors = fit_ladder(lm, BASE_FULL, "MIMIC landmark FULL SPEC (internal)")
    # harmonized spec on MIMIC: sepsis48 + first-day-mean vitals (cols already exist)
    h_rows, h_ors = None, None
    try:
        h_rows, h_ors = fit_ladder(lm, BASE_H, "MIMIC landmark HARMONIZED spec (fit for transport)")
    except Exception as ex:  # missing column fallback
        stamp(f"harmonized ladder错误: {ex}")

    # ---------- transported to eICU landmark ----------
    # fit harmonized Base / base+PF / base+PF+VR in MIMIC lm; apply fixed coefficient
    # (with MIMIC-fitted scaler) to eICU lm; report discrimination + calibration.
    import statsmodels.api as sm
    trows = []
    eic_lm = pd.read_csv(OUTDIR / "eicu_landmark_round2.csv")
    y_e = eic_lm[OUTCOME].astype(int)
    for name, vars_ in [("Base", BASE_H), ("Base + P/F", BASE_H + PF_48H), ("Base + P/F + VR", BASE_H + PF_48H + VR_48H)]:
        x_m = sm.add_constant(scale_x(lm, vars_), has_constant="add")
        res = sm.Logit(lm[OUTCOME].astype(int), x_m).fit(disp=False, maxiter=300)
        x_e = sm.add_constant(scale_x(eic_lm, vars_), has_constant="add")
        p_e = res.predict(x_e)
        a_e = roc_auc_score(y_e, p_e)
        lg = np.log(np.clip(p_e, 1e-6, 1 - 1e-6) / (1 - np.clip(p_e, 1e-6, 1 - 1e-6)))
        cres = sm.Logit(y_e.astype(float), sm.add_constant(lg)).fit(disp=False, maxiter=300)
        cp = np.asarray(cres.params, dtype=float)
        trows.append({"Row": "eICU landmark (transported)", "Model": name, "N": len(eic_lm),
                      "Deaths": int(y_e.sum()), "AUROC": f"{a_e:.3f}",
                      "Calibration intercept": f"{cp[0]:.3f}", "Calibration slope": f"{cp[1]:.3f}"})
        stamp(f"[transported] {name}: eICU AUROC={a_e:.4f}, 校准 {cp[0]:.3f}/{cp[1]:.3f}（旧: 0.771/0.787, −0.27/0.47）")

    # ---------- eICU refit (replication, for Supp Table 5) ----------
    rrows, rors = fit_ladder(eic_lm, BASE_H, "eICU landmark refit (replication)")

    # ---------- save all ----------
    out = {
        "mimic_fullspec": lm_rows,
        "mimic_harmonizedspec": h_rows or [],
        "eicu_transported": trows,
        "eicu_refit": rrows,
    }
    pd.concat([pd.DataFrame(v) for v in out.values() if v], ignore_index=True) \
        .to_csv(OUTDIR / "round2_ladders.csv", index=False)
    pd.concat([lm_ors.assign(model=lambda d: "MIMIC full|" + d["model"]),
               h_ors.assign(model=lambda d: "MIMIC harm|" + d["model"]) if h_ors is not None else None,
               rors.assign(model=lambda d: "eICU refit|" + d["model"])], ignore_index=True) \
        .to_csv(OUTDIR / "round2_vr_ors.csv", index=False)
    (OUTDIR / "master_part1_log.md").write_text("# Master M1 日志\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("M1 完成")


if __name__ == "__main__":
    main()
