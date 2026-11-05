#!/usr/bin/env python3
"""Round-2 master, part 3: Table 4 (sensitivities, rebuilt world) assembly + docx.

Rows (all on the Round-2 rebuilt data):
  1. primary_cc          - complete-case joint model on the rebuilt association cohort
  2. mice                - from run_mice_round2.py pooled output (outcome in the model)
  3. landmark_48h        - the TRUE 48-h landmark cohort (offset-based; 31号 construction)
  4. narrow_window_±1h   - |pair lag| <= 1 h features (from run_narrow_round2.py output)
  5. first_24h_features  - features recalculated within 0-24 h (rule-B-trimmed points)
  6. measurement_adjusted- + n_points + vr_time_span_h into the fully adjusted model
Strata table (Supplementary Table 4) rebuilt from the rebuilt n_points.

Reuses make_table4's docx builders via module import (paths overridden).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTCOME = "hospital_expire_flag"
RNG = 20260721
ICU = ROOT / "原始数据" / "core" / "icustays.csv.gz"
ADM = ROOT / "原始数据" / "core" / "admissions.csv.gz"
BASE_A = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
FULL = BASE_A + PF + VR
# Landmark row must be covariate-time-consistent with Table 3 Row 1: sepsis is the
# 48-h-determinable Sepsis-3 flag derived by the rebuild (replaces the any-time flag).
BASE_LM = ["sepsis3_48h" if v == "sepsis3" else v for v in BASE_A]
FULL_LM = BASE_LM + PF + VR
MEAS = ["n_points", "vr_time_span_h"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def scale_x(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(df[variables]), columns=variables, index=df.index)


def fmt_or(o, lo, hi):
    return f"{o:.3f} ({lo:.3f}-{hi:.3f})"


def fit_joint(cc: pd.DataFrame, variables: list[str], label: str,
              vr_pair: tuple[str, str] = ("vr_initial", "vr_delta")) -> tuple[dict, pd.DataFrame]:
    import statsmodels.api as sm
    x = sm.add_constant(scale_x(cc, variables), has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)
    rows = []
    for var in vr_pair:
        beta, se = float(res.params[var]), float(res.bse[var])
        z = beta / se
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        rows.append({"variable": var, "OR": math.exp(beta),
                     "CI_lower": math.exp(beta - 1.96 * se), "CI_upper": math.exp(beta + 1.96 * se), "p_value": p})
    or_df = pd.DataFrame(rows)
    initial = or_df.loc[or_df.variable.eq(vr_pair[0])].iloc[0]
    delta = or_df.loc[or_df.variable.eq(vr_pair[1])].iloc[0]
    out = {"Analysis": label, "N": len(cc), "Deaths": int(y.sum()),
           "VR initial OR (95% CI)": fmt_or(initial.OR, initial.CI_lower, initial.CI_upper),
           "VR initial P value": "<0.001" if initial.p_value < 0.001 else f"{initial.p_value:.3f}",
           "VR delta OR (95% CI)": fmt_or(delta.OR, delta.CI_lower, delta.CI_upper),
           "VR delta P value": "<0.001" if delta.p_value < 0.001 else f"{delta.p_value:.3f}",
           "AUROC": f"{roc_auc_score(y, pred):.3f}",
           "Brier": f"{brier_score_loss(y, pred):.3f}"}
    return out, or_df




def build_docx(df, title_text, note_text, out_path):
    """Self-contained landscape TNR docx writer (replaces the legacy external helper)."""
    import pandas as pd
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.top_margin = Cm(1.5); sec.bottom_margin = Cm(1.5)
    sec.left_margin = Cm(1.6); sec.right_margin = Cm(1.6)

    title = doc.add_paragraph(); title.paragraph_format.space_after = Pt(8)
    r0 = title.add_run(title_text); r0.font.name = "Times New Roman"; r0.font.size = Pt(9); r0.font.bold = True

    headers = list(df.columns)
    table = doc.add_table(rows=1 + len(df), cols=len(headers))
    table.style = "Table Grid"; table.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Cm(6.4), Cm(1.6), Cm(1.7), Cm(2.6), Cm(2.6), Cm(3.4), Cm(2.4), Cm(3.0), Cm(3.0)]
    for j, w in enumerate(widths[:len(headers)]):
        for row_cells in table.rows:
            if j < len(row_cells.cells):
                row_cells.cells[j].width = w
    for j, h in enumerate(headers):
        c = table.rows[0].cells[j]; c.text = ""
        rn = c.paragraphs[0].add_run(str(h)); rn.font.name = "Times New Roman"; rn.font.size = Pt(7); rn.font.bold = True
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, h in enumerate(headers):
            c = table.rows[i].cells[j]; c.text = ""
            tval = row[h]
            if pd.isna(tval):
                tval = "—"
            rn = c.paragraphs[0].add_run(str(tval))
            rn.font.name = "Times New Roman"; rn.font.size = Pt(7)
            if j > 0:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = doc.add_paragraph(); note.paragraph_format.space_before = Pt(6)
    rn = note.add_run(note_text); rn.font.name = "Times New Roman"; rn.font.size = Pt(6.5)
    doc.save(out_path)


def main() -> None:
    # 装配输入
    rd = pd.read_csv(OUTDIR / "mimic_analysis_ready_round2.csv")
    rows = []

    row_a, or_a = fit_joint(rd.dropna(subset=[OUTCOME] + FULL), FULL, "primary_cc")
    rows.append(row_a)
    stamp(f"primary_cc: {row_a['N']}/{row_a['Deaths']} {row_a['VR initial OR (95% CI)']} / {row_a['VR delta OR (95% CI)']} AUROC {row_a['AUROC']}")

    mice_path = OUTDIR / "mice_outcome_pooled_OR.csv"
    if mice_path.exists():
        mp = pd.read_csv(mice_path)
        mi = mp.loc[mp.variable.eq("vr_initial")].iloc[0]
        dl = mp.loc[mp.variable.eq("vr_delta")].iloc[0]
        rows.append({"Analysis": "mice (m=10, outcome in model)", "N": len(rd),
                     "Deaths": int(rd[OUTCOME].sum()),
                     "VR initial OR (95% CI)": fmt_or(mi.OR, mi.CI_lower, mi.CI_upper),
                     "VR initial P value": "<0.001" if mi.p_value < 0.001 else f"{mi.p_value:.3f}",
                     "VR delta OR (95% CI)": fmt_or(dl.OR, dl.CI_lower, dl.CI_upper),
                     "VR delta P value": "<0.001" if dl.p_value < 0.001 else f"{dl.p_value:.3f}",
                     "AUROC": str(round(float(mp.get("AUROC", [np.nan])[0]), 3)) if "AUROC" in mp.columns else "",
                     "Brier": ""})
        stamp(f"mice行: {rows[-1]['VR initial OR (95% CI)']} / {rows[-1]['VR delta OR (95% CI)']}")
    else:
        stamp("!! mice输出尚未生成——待 run_mice_round2 完成后重跑此装配")

    # landmark (TRUE, offset-based) - reuse 31号 construction
    nr = rd.copy()
    icu = pd.read_csv(ICU, usecols=["stay_id", "intime", "outtime"])
    adm = pd.read_csv(ADM, usecols=["hadm_id", "dischtime", "deathtime"])
    for c in ["intime", "outtime"]:
        icu[c] = pd.to_datetime(icu[c], errors="coerce")
    for c in ["dischtime", "deathtime"]:
        adm[c] = pd.to_datetime(adm[c], errors="coerce")
    nr = nr.merge(icu, on="stay_id", how="left").merge(adm, on="hadm_id", how="left")
    cc0 = nr.dropna(subset=[OUTCOME] + FULL)
    hour48 = cc0["intime"] + pd.Timedelta(hours=48)
    died = cc0[OUTCOME].astype(int) == 1
    wd = died & ((cc0["deathtime"].notna() & (cc0["deathtime"] <= hour48)) |
                 (cc0["deathtime"].isna() & cc0["dischtime"].notna() & (cc0["dischtime"] <= hour48)))
    ed = (~died) & cc0["dischtime"].notna() & (cc0["dischtime"] <= hour48)
    cc_lm = cc0.loc[~wd & ~ed]
    row_l, or_l = fit_joint(cc_lm, FULL_LM, "48-h landmark (alive and hospitalized at 48 h)")
    rows.append(row_l)
    stamp(f"landmark行: {row_l['N']}/{row_l['Deaths']} {row_l['VR initial OR (95% CI)']} / {row_l['VR delta OR (95% CI)']} AUROC {row_l['AUROC']}")

    # ±1h narrow features (if rebuilt) else fallback note
    nw_path = OUTDIR / "narrow_window_OR_round2.csv"
    if nw_path.exists():
        nw = pd.read_csv(nw_path)
        o_r = nw.loc[nw.variable.eq("vr_initial"), "OR_CI"].iloc[0] if "variable" in nw.columns else str(nw.iloc[0]["OR_CI"])
        d_r = nw.loc[nw.variable.eq("vr_delta"), "OR_CI"].iloc[0] if "variable" in nw.columns else ""
        o_p = nw.loc[nw.variable.eq("vr_initial"), "p"].iloc[0]
        d_p = nw.loc[nw.variable.eq("vr_delta"), "p"].iloc[0]
        narrow = nw.loc[nw.variable.eq("vr_initial")].iloc[0]
        rows.append({"Analysis": "ABG–ventilator pairs matched within ±1 h", "N": int(nw.get("N", [None])[0] if "N" in nw.columns else None),
                     "Deaths": int(narrow["Deaths"]), "VR initial OR (95% CI)": o_r, "VR initial P value": o_p,
                     "VR delta OR (95% CI)": d_r, "VR delta P value": d_p,
                     "AUROC": f"{narrow['AUROC']:.3f}", "Brier": f"{narrow['Brier']:.3f}"})
        stamp(f"±1h行: {o_r} / {d_r}")

    # first-24h features (rule-B-trimmed points in [0,24])
    pts = pd.read_csv(OUTDIR / "trajectory_mimic_48h_round2.csv")
    pts24 = pts.loc[(pts["hour"] >= 0) & (pts["hour"] <= 24)].sort_values(["stay_id", "hour"])
    f24_rows = []
    for sid, g in pts24.groupby("stay_id", sort=False):
        f24_rows.append({"stay_id": sid, "n_points_24h": len(g),
                         "vr_initial_24h": float(g["vr"].iloc[0]),
                         "vr_delta_24h": float(g["vr"].iloc[-1] - g["vr"].iloc[0]),
                         "pf_initial_24h": float(g["pf"].iloc[0]),
                         "pf_min_24h": float(g["pf"].min()),
                         "vr_time_span_24h": float(g["hour"].iloc[-1] - g["hour"].iloc[0])})
    f24 = pd.DataFrame(f24_rows)
    h24 = cc0.drop(columns=["vr_initial", "vr_delta", "pf_initial", "pf_min"], errors="ignore") \
             .merge(f24, on="stay_id", how="left")
    h24 = h24.loc[h24["n_points_24h"] >= 2]
    vars24 = BASE_A + ["pf_initial_24h", "pf_min_24h", "vr_initial_24h", "vr_delta_24h"]
    row_h, or_h = fit_joint(h24, vars24, "features restricted to the first 24 h", ("vr_initial_24h", "vr_delta_24h"))
    rows.append(row_h)
    stamp(f"24h行: {row_h['N']}/{row_h['Deaths']} {row_h['VR initial OR (95% CI)']} / {row_h['VR delta OR (95% CI)']} AUROC {row_h['AUROC']}")

    # measurement-adjusted
    row_m, or_m = fit_joint(rd.dropna(subset=[OUTCOME] + FULL + MEAS), FULL + MEAS, "adjusted for measurement count and observation span")
    rows.append(row_m)
    stamp(f"measurement行: {row_m['N']}/{row_m['Deaths']} {row_m['VR initial OR (95% CI)']} / {row_m['VR delta OR (95% CI)']} AUROC {row_m['AUROC']}")

    t4 = pd.DataFrame(rows)[["Analysis", "N", "Deaths", "VR initial OR (95% CI)", "VR initial P value",
                             "VR delta OR (95% CI)", "VR delta P value", "AUROC", "Brier"]]
    t4.to_csv(OUTDIR / "Table4_round2.csv", index=False)

    # strata (Supplementary Table 4, rebuilt)
    srows = []
    for label, sub in [("3-4 paired measurements", rd.loc[rd["n_points"].between(3, 4)]),
                       ("5-7 paired measurements", rd.loc[rd["n_points"].between(5, 7)]),
                       (">=8 paired measurements", rd.loc[rd["n_points"] >= 8])]:
        r, o_df = fit_joint(sub.dropna(subset=[OUTCOME] + FULL), FULL, label)
        r["Stratum"] = label
        srows.append(r)
        stamp(f"strata {label}: {r['N']}/{r['Deaths']} {r['VR initial OR (95% CI)']} / {r['VR delta OR (95% CI)']}")
    pd.DataFrame(srows).to_csv(OUTDIR / "TableS_strata_round2.csv", index=False)

    # docx (reuse make_table4 style builders)
    table4_note = (
        "Sensitivity analyses used the MIMIC-IV internal fully adjusted model on the Round-2 rebuilt cohort "
        "(features re-derived with explicit truncation of the observation window at the earliest of death, "
        "terminal extubation, ICU discharge, or hospital discharge; vital outcomes re-jointed from official "
        "source tables). The primary complete-case model included age, sex, Charlson comorbidity index, sepsis, "
        "COPD, congestive heart failure, SOFA score, PEEP, driving pressure, tidal volume per predicted body "
        "weight, heart rate, mean arterial pressure, temperature, lactate, blood urea nitrogen, creatinine, "
        "white blood cell count, bicarbonate, initial P/F ratio, minimum P/F ratio, VR initial, and VR delta. "
        "Odds ratios are reported per 1-SD increase. Multiple imputation used chained equations with m=10 "
        "imputations for the same predictor set, with the outcome included in the imputation model and estimates "
        "pooled by Rubin's rules. The first 24 h analysis recalculated VR initial, VR delta, initial P/F ratio, "
        "and minimum P/F ratio using paired observations within 0-24 h after ICU admission and required at least "
        "two paired measurements in that window. The measurement-adjusted model additionally included the number "
        "of paired measurements and the VR observation time span. In the 48-h landmark row, sepsis denotes the "
        "48-hour-determinable Sepsis-3 definition (suspected-infection onset and qualifying SOFA assessment both "
        "within the first 48 h), consistent with Table 3 Row 1, which is fitted on the same landmark cohort; the "
        "association-framework rows retain the any-time Sepsis-3 flag and the covariate-adjustment interpretation "
        "stated in the Methods. Brier score is shown as a secondary overall prediction error metric.")
    strata_note = (
        "Patients were stratified according to the number of paired P/F-VR measurements obtained within the first "
        "48 h after ICU admission (rebuilt cohort). Each stratum used the same fully adjusted complete-case model "
        "as the primary analysis, including COPD, congestive heart failure, initial and minimum P/F ratio, VR "
        "initial, and VR delta. Odds ratios are reported per 1-SD increase within each stratum. This analysis "
        "evaluates whether the VR-mortality association is consistent across different measurement frequencies.")
    build_docx(t4, "Table 4. Sensitivity analyses for the association between early VR features and hospital mortality.",
                  table4_note, OUTDIR / "Table4_round2_TNR_xiaoliu.docx")
    build_docx(pd.DataFrame(srows)[["Stratum", "N", "Deaths", "VR initial OR (95% CI)", "VR initial P value",
                                       "VR delta OR (95% CI)", "VR delta P value", "AUROC", "Brier"]],
                  "Supplementary Table 4. Stratified analyses by the number of paired P/F-VR measurements.",
                  strata_note, OUTDIR / "TableS4_round2_TNR_xiaoliu.docx")
    (OUTDIR / "master_part3_log.md").write_text("# Master Part 3 (Table 4 assembly)\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Part 3 完成")


if __name__ == "__main__":
    main()
