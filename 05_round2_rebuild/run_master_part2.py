#!/usr/bin/env python3
"""Round-2 master, part 2 (A): harmonized family on ITS OWN cohort + transported +
association primary (Table 2) on the rebuilt complete-case cohort.

Mirrors the ORIGINAL Table-3 design faithfully (harmonized row on its own complete-case
subset, transported coefficients fitted THERE, eICU row = fixed-coefficient application),
but everything now lives in the Round-2 rebuilt world:
  * MIMIC association/complete-case/landmark cohorts = Rule-B truncation rebuild;
  * harmonized spec includes COPD/CHF (14+1 covariates) with sepsis48 (sepsis
    determinable by the 48-h landmark: 48-h Sepsis-3 in MIMIC-IV, admitDx text in eICU);
  * eICU vitals = native first-day means (R2-2 fix).

Outputs: harm ladder (MIMIC), transported (eICU landmark), refit ladder (supp),
Table-2 OR table (8 features x 3 models), comparison printout.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTCOME = "hospital_expire_flag"
RNG = 20260721
ICU = ROOT / "原始数据" / "core" / "icustays.csv.gz"
ADM = ROOT / "原始数据" / "core" / "admissions.csv.gz"

BASE_FULL = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
             "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
             "bun", "creatinine", "wbc", "bicarbonate"]
BASE_H = ["age", "sex", "charlson", "sepsis48", "copd", "chf", "peep",
          "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF_48H = ["pf_initial", "pf_min"]
VR_48H = ["vr_initial", "vr_delta"]
FEATURES8 = ["vr_initial", "vr_last", "vr_mean", "vr_max", "vr_min", "vr_auc", "vr_delta", "vr_slope"]
DEMO = ["age", "sex", "charlson", "sepsis3", "copd", "chf"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def scale_x(df: pd.DataFrame, variables: list[str], scaler: StandardScaler | None = None):
    if scaler is None:
        scaler = StandardScaler().fit(df[variables])
    return pd.DataFrame(scaler.transform(df[variables]), columns=variables, index=df.index), scaler


def fit_logit_vars(df: pd.DataFrame, vars_: list[str]) -> tuple[object, float]:
    import statsmodels.api as sm
    x, _ = scale_x(df, vars_)
    xl = sm.add_constant(x, has_constant="add")
    y = df[OUTCOME].astype(int)
    res = sm.Logit(y, xl).fit(disp=False, maxiter=300)
    auc = roc_auc_score(y, res.predict(xl))
    return res, auc


def or_of(res, v: str):
    b, se = float(res.params[v]), float(res.bse[v])
    z = b / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se), p


def main() -> None:
    mim = pd.read_csv(OUTDIR / "mimic_analysis_ready_round2.csv")
    eic = pd.read_csv(OUTDIR / "eicu_analysis_ready_round2.csv")
    stamp(f"读入: MIMIC {len(mim)}, eICU {len(eic)}")

    # ---------- Part 1: harmonized OWN cohort (MIMIC) + ladder ----------
    # Harmonized sepsis slot: the 48-h-determinable Sepsis-3 flag derived by the rebuild
    # (present when master_part1 has run; fall back to the rebuild column otherwise).
    if "sepsis48" not in mim.columns:
        mim["sepsis48"] = mim["sepsis3_48h"].astype(int)
    mim2 = mim.merge(pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime"]), on="stay_id", how="left")
    mim2 = mim2.merge(pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime"]), on="hadm_id", how="left")
    for col in ["intime", "outtime", "dischtime", "deathtime"]:
        mim2[col] = pd.to_datetime(mim2[col], errors="coerce")
    harm_cc = mim.dropna(subset=[OUTCOME] + BASE_H + PF_48H + VR_48H).copy()
    stamp(f"[harm-cc MIMIC] n={len(harm_cc)}, 死亡={int(harm_cc[OUTCOME].sum())}（旧全队列harmonized: 11,647）")
    hh = mim2.loc[harm_cc.index]
    hour48 = hh["intime"] + pd.Timedelta(hours=48)
    died = harm_cc[OUTCOME] == 1
    wd = died & ((hh["deathtime"].notna() & (hh["deathtime"] <= hour48)) |
                 (hh["deathtime"].isna() & hh["dischtime"].notna() & (hh["dischtime"] <= hour48)))
    ed = (~died) & hh["dischtime"].notna() & (hh["dischtime"] <= hour48)
    harm_lm = harm_cc.loc[~wd & ~ed].copy()
    stamp(f"[harm-landmark MIMIC] n={len(harm_lm)}, 48h后死亡={int(harm_lm[OUTCOME].sum())}")
    harm_lm.to_csv(OUTDIR / "harm_landmark_mimic.csv", index=False)

    hrows = []
    for name, vars_ in [("Base", BASE_H), ("Base + P/F", BASE_H + PF_48H), ("Base + P/F + VR", BASE_H + PF_48H + VR_48H)]:
        res, auc = fit_logit_vars(harm_lm, vars_)
        import statsmodels.api as sm
        x_h, _ = scale_x(harm_lm, vars_)
        pred = res.predict(sm.add_constant(x_h, has_constant="add"))
        hrows.append({"Row": "MIMIC harm (own cohort)", "Model": name, "N": len(harm_lm),
                      "Deaths": int(harm_lm[OUTCOME].sum()), "AUROC": f"{auc:.3f}",
                      "Brier": f"{brier_score_loss(harm_lm[OUTCOME].astype(int), pred):.3f}"})
        stamp(f"[MIMIC harm·own] {name}: N={len(harm_lm)}, AUROC={auc:.4f}, Brier={hrows[-1]['Brier']}")

    # ---------- Part 2: transported to eICU landmark (coeffs from harm-own fit) ----------
    eic["sepsis_h"] = eic["sepsis3"].astype(int)
    # harmonized/transport slot: admissionDx-recorded sepsis (admission-form entries; in the current eICU landmark cohort every sepsis-positive entry carries an offset <=48 h,
    # hence determinable before the landmark time)
    eic["sepsis48"] = eic["sepsis3"].astype(int)
    eic_lm = pd.read_csv(OUTDIR / "eicu_landmark_round2.csv")
    eic_lm = eic_lm[eic_lm["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy() if \
        "hospitaldischargestatus" in eic_lm.columns else eic_lm
    # 以原生首日均值列为准（R2-2 修复）：删除 M1 从旧 builder 回填的 aps 版本，再映射
    aps_poll = [c for c in ["hr", "map", "temp", "rr_vitalsign", "spo2"] if c in eic_lm.columns]
    if aps_poll:
        eic_lm = eic_lm.drop(columns=aps_poll)
    nat_map = {"hr_native": "hr", "map_native": "map", "temp_native": "temp",
               "rr_native": "rr_vitalsign", "spo2_native": "spo2"}
    for a, b in nat_map.items():
        if a in eic_lm.columns:
            eic_lm = eic_lm.rename(columns={a: b})
    eic_lm = eic_lm.dropna(subset=[OUTCOME] + BASE_H + PF_48H + VR_48H).copy()
    stamp(f"[诊断] eICU-lm 协变量均值: " + ", ".join(
        f"{c}={pd.to_numeric(eic_lm[c], errors='coerce').mean():.2f}"
        for c in ["hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"] if c in eic_lm.columns))
    stamp(f"[诊断] MIMIC-harm-lm 协变量均值: " + ", ".join(
        f"{c}={pd.to_numeric(harm_lm[c], errors='coerce').mean():.2f}"
        for c in ["hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]))
    stamp(f"[eICU landmark] n={len(eic_lm)}, 48h后死亡={int(eic_lm[OUTCOME].sum())}")
    trows = []
    import statsmodels.api as sm
    y_e = eic_lm[OUTCOME].astype(int)
    for name, vars_ in [("Base", BASE_H), ("Base + P/F", BASE_H + PF_48H), ("Base + P/F + VR", BASE_H + PF_48H + VR_48H)]:
        x_h, scaler = scale_x(harm_lm, vars_)
        res = sm.Logit(harm_lm[OUTCOME].astype(int), sm.add_constant(x_h, has_constant="add")).fit(disp=False, maxiter=300)
        x_e, _ = scale_x(eic_lm, vars_, scaler=scaler)
        p_e = res.predict(sm.add_constant(x_e, has_constant="add"))
        a_e = roc_auc_score(y_e, p_e)
        lg = np.log(np.clip(p_e, 1e-6, 1 - 1e-6) / (1 - np.clip(p_e, 1e-6, 1 - 1e-6)))
        cres = sm.Logit(y_e.astype(float), sm.add_constant(lg)).fit(disp=False, maxiter=300)
        cp = np.asarray(cres.params, dtype=float)
        trows.append({"Row": "eICU landmark (transported)", "Model": name, "N": len(eic_lm),
                      "Deaths": int(y_e.sum()), "AUROC": f"{a_e:.3f}",
                      "Calibration intercept": f"{cp[0]:.3f}", "Calibration slope": f"{cp[1]:.3f}"})
        stamp(f"[transported·own-cohort-coeffs] {name}: AUROC={a_e:.4f}, 校准 {cp[0]:.3f}/{cp[1]:.3f}")

    # ---------- Part 3: eICU refit ladder (replication, supp) ----------
    rrows = []
    for name, vars_ in [("Base", BASE_H), ("Base + P/F", BASE_H + PF_48H), ("Base + P/F + VR", BASE_H + PF_48H + VR_48H)]:
        res, auc = fit_logit_vars(eic_lm, vars_)
        if "vr_initial" in vars_ and "vr_delta" in vars_:
            o, lo, hi, p = or_of(res, "vr_initial")
            o2, lo2, hi2, p2 = or_of(res, "vr_delta")
            ors_str = (f"VR initial {o:.3f} ({lo:.3f}-{hi:.3f}) P={p:.3g}; "
                       f"VR delta {o2:.3f} ({lo2:.3f}-{hi2:.3f}) P={p2:.3g}")
        else:
            ors_str = "—"
        rrows.append({"Row": "eICU refit (replication)", "Model": name, "N": len(eic_lm),
                      "Deaths": int(y_e.sum()), "AUROC": f"{auc:.3f}",
                      "vr_initial OR": f"{o:.3f} ({lo:.3f}-{hi:.3f})" if "vr_initial" in vars_ else "",
                      "vr_delta OR": f"{o2:.3f} ({lo2:.3f}-{hi2:.3f})" if "vr_delta" in vars_ else ""})
        stamp(f"[eICU refit] {name}: AUROC={auc:.4f}, {ors_str}")

    # ---------- Part 4: Table 2 (association primary) on rebuilt cc ----------
    cc = mim.dropna(subset=[OUTCOME] + BASE_FULL + PF_48H + VR_48H).copy()
    stamp(f"[Table2 cc] n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")
    or_rows = []
    for feat in FEATURES8:
        for m_idx, extra in [(1, []), (2, DEMO), (3, BASE_FULL + PF_48H)]:
            vars_ = [feat] + extra
            df_m = cc
            if feat == "vr_delta" and m_idx >= 2:
                df_m = cc  # delta 模型条件于 vr_initial: 变量集 = [delta, initial] + 协变量
                vars_ = ["vr_delta", "vr_initial"] + extra
            res, _ = fit_logit_vars(df_m, vars_)
            target = "vr_delta" if feat == "vr_delta" else feat
            o, lo, hi, p = or_of(res, target)
            or_rows.append({"Feature": feat, "Model": m_idx, "N": len(df_m),
                            "OR": round(o, 3), "CI_lower": round(lo, 3), "CI_upper": round(hi, 3), "P": f"{p:.3g}"})
            stamp(f"[Table2] {feat} M{m_idx}: OR {o:.3f} ({lo:.3f}-{hi:.3f}) P={p:.3g}")

    pd.DataFrame(hrows + trows + rrows).to_csv(OUTDIR / "round2_harm_ladders.csv", index=False)
    pd.DataFrame(or_rows).to_csv(OUTDIR / "table2_round2.csv", index=False)
    (OUTDIR / "master_part2_log.md").write_text("# Master Part A 日志\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Part A 完成")


if __name__ == "__main__":
    main()