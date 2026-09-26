#!/usr/bin/env python3
"""Create Table 4 and supplementary measurement-count stratified table."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.imputation.mice import MICEData


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "13_Table4正式版"
OUTDIR.mkdir(exist_ok=True)
DATA = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
POINTS_LONG = Path("./raw_data/mimmc4/mimmc/项目2/outputs/pf_vr_dynamic_analysis/mimic_points_long.csv")
OUTCOME = "hospital_expire_flag"
RNG = 20260721
MICE_M = 10

BASE_A = [
    "age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
PF_48H = ["pf_initial", "pf_min"]
VR_48H = ["vr_initial", "vr_delta"]
MEASUREMENT_ADJUSTERS = ["n_points", "vr_time_span_h"]

DISPLAY_ANALYSIS = {
    "primary_cc": "Complete-case primary analysis",
    "mice": "Multiple imputation analysis (m=10)",
    "exclude_48h_deaths": "Excluding deaths within 48 h",
    "first_24h": "First 24 h features only",
    "adjust_measurement": "Additional adjustment for measurement count and VR time span",
}


def p_from_z(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.3f} ({lo:.3f}-{hi:.3f})"


def scale_x(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    scaler = StandardScaler().fit(df[variables])
    return pd.DataFrame(scaler.transform(df[variables]), columns=variables, index=df.index)


def fit_complete_case(df: pd.DataFrame, label: str, variables: list[str], vr_vars: tuple[str, str]) -> tuple[dict, pd.DataFrame]:
    cc = df.dropna(subset=[OUTCOME] + variables).copy()
    x = sm.add_constant(scale_x(cc, variables), has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)

    rows = []
    for var in variables:
        beta = float(res.params[var])
        se = float(res.bse[var])
        rows.append({
            "analysis": label,
            "variable": var,
            "OR": math.exp(beta),
            "CI_lower": math.exp(beta - 1.96 * se),
            "CI_upper": math.exp(beta + 1.96 * se),
            "p_value": p_from_z(beta / se),
        })
    or_df = pd.DataFrame(rows)

    initial_var, delta_var = vr_vars
    initial = or_df.loc[or_df["variable"] == initial_var].iloc[0]
    delta = or_df.loc[or_df["variable"] == delta_var].iloc[0]
    out = {
        "analysis_key": label,
        "Analysis": DISPLAY_ANALYSIS.get(label, label),
        "N": int(len(cc)),
        "Deaths": int(y.sum()),
        "VR initial OR (95% CI)": fmt_or(initial.OR, initial.CI_lower, initial.CI_upper),
        "VR initial P value": fmt_p(initial.p_value),
        "VR delta OR (95% CI)": fmt_or(delta.OR, delta.CI_lower, delta.CI_upper),
        "VR delta P value": fmt_p(delta.p_value),
        "AUROC": f"{roc_auc_score(y, pred):.3f}",
        "Brier": f"{brier_score_loss(y, pred):.3f}",
    }
    return out, or_df


def mice_datasets(df: pd.DataFrame, variables: list[str], m: int, burnin: int = 5) -> list[pd.DataFrame]:
    data = MICEData(df[variables].apply(pd.to_numeric, errors="coerce"))
    np.random.seed(RNG)
    for _ in range(burnin):
        data.update_all(1)
    out = []
    for _ in range(m):
        data.update_all(1)
        imp = df.copy()
        imp[variables] = data.data[variables].copy()
        out.append(imp)
    return out


def fit_mice(df: pd.DataFrame, variables: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    parts = []
    perf_rows = []
    for idx, imp in enumerate(mice_datasets(df, variables, MICE_M), start=1):
        x = sm.add_constant(scale_x(imp, variables), has_constant="add")
        y = imp[OUTCOME].astype(int)
        res = sm.Logit(y, x).fit(disp=False, maxiter=300)
        pred = res.predict(x)
        perf_rows.append({
            "imputation": idx,
            "n": int(len(imp)),
            "events": int(y.sum()),
            "AUROC": float(roc_auc_score(y, pred)),
            "Brier": float(brier_score_loss(y, pred)),
        })
        rows = []
        for var in variables:
            rows.append({
                "imputation": idx,
                "variable": var,
                "coef": float(res.params[var]),
                "se": float(res.bse[var]),
            })
        parts.append(pd.DataFrame(rows))

    pooled = []
    for var in variables:
        coefs = np.array([p.loc[p.variable.eq(var), "coef"].iloc[0] for p in parts])
        ses = np.array([p.loc[p.variable.eq(var), "se"].iloc[0] for p in parts])
        qbar = float(coefs.mean())
        ubar = float(np.mean(ses ** 2))
        b = float(np.var(coefs, ddof=1))
        total = ubar + (1 + 1 / MICE_M) * b
        se = math.sqrt(total)
        pooled.append({
            "analysis": "mice",
            "variable": var,
            "OR": math.exp(qbar),
            "CI_lower": math.exp(qbar - 1.96 * se),
            "CI_upper": math.exp(qbar + 1.96 * se),
            "p_value": p_from_z(qbar / se),
            "within_var": ubar,
            "between_var": b,
        })
    pooled_df = pd.DataFrame(pooled)
    perf = pd.DataFrame(perf_rows)
    initial = pooled_df.loc[pooled_df["variable"] == "vr_initial"].iloc[0]
    delta = pooled_df.loc[pooled_df["variable"] == "vr_delta"].iloc[0]
    out = {
        "analysis_key": "mice",
        "Analysis": DISPLAY_ANALYSIS["mice"],
        "N": int(len(df)),
        "Deaths": int(df[OUTCOME].sum()),
        "VR initial OR (95% CI)": fmt_or(initial.OR, initial.CI_lower, initial.CI_upper),
        "VR initial P value": fmt_p(initial.p_value),
        "VR delta OR (95% CI)": fmt_or(delta.OR, delta.CI_lower, delta.CI_upper),
        "VR delta P value": fmt_p(delta.p_value),
        "AUROC": f"{perf.AUROC.mean():.3f}",
        "Brier": f"{perf.Brier.mean():.3f}",
    }
    return out, pooled_df, perf


def build_24h_dataset(main: pd.DataFrame) -> pd.DataFrame:
    points = pd.read_csv(POINTS_LONG)
    pts = points.loc[(points["hour"] >= 0) & (points["hour"] <= 24)].copy()
    pts = pts.sort_values(["stay_id", "hour"])
    rows = []
    for stay_id, g in pts.groupby("stay_id", sort=False):
        rows.append({
            "stay_id": stay_id,
            "n_points_24h": int(len(g)),
            "vr_initial_24h": float(g["vr"].iloc[0]),
            "vr_delta_24h": float(g["vr"].iloc[-1] - g["vr"].iloc[0]),
            "pf_initial_24h": float(g["pf"].iloc[0]),
            "pf_min_24h": float(g["pf"].min()),
            "vr_time_span_24h": float(g["hour"].iloc[-1] - g["hour"].iloc[0]),
        })
    features = pd.DataFrame(rows)
    merged = main.merge(features, on="stay_id", how="left")
    return merged.loc[merged["n_points_24h"] >= 2].copy()


def set_cell_border(cell, **kwargs) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            tag = "w:" + edge
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            for key, val in kwargs[edge].items():
                element.set(qn("w:" + key), str(val))


def set_cell_shading(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=60, start=80, bottom=60, end=80) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def style_run(run, size=6.5, bold=False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def style_cell(cell, bold=False, align="center", fill=None) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if fill:
        set_cell_shading(cell, fill)
    set_cell_margins(cell)
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if align == "left" else WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            style_run(run, bold=bold)


def add_text(cell, text: str, bold=False, align="center", fill=None) -> None:
    cell.text = str(text)
    style_cell(cell, bold=bold, align=align, fill=fill)


def set_table_borders(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(
                cell,
                top={"val": "nil"},
                left={"val": "nil"},
                right={"val": "nil"},
                bottom={"val": "single", "sz": "4", "color": "D9D9D9"},
            )
    for cell in table.rows[0].cells:
        set_cell_border(
            cell,
            top={"val": "single", "sz": "12", "color": "000000"},
            bottom={"val": "single", "sz": "8", "color": "000000"},
            left={"val": "nil"},
            right={"val": "nil"},
        )
    for cell in table.rows[-1].cells:
        set_cell_border(
            cell,
            bottom={"val": "single", "sz": "12", "color": "000000"},
            left={"val": "nil"},
            right={"val": "nil"},
        )


def set_col_widths(table, widths_cm: list[float]) -> None:
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            row.cells[idx].width = Cm(width)


def build_docx(df: pd.DataFrame, title_text: str, note_text: str, out_path: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.6)
    section.right_margin = Cm(1.6)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run(title_text)
    style_run(run, size=9, bold=True)

    headers = list(df.columns)
    table = doc.add_table(rows=1 + len(df), cols=len(headers))
    table.autofit = False
    widths = [5.2, 1.3, 1.4, 3.0, 1.4, 3.0, 1.4, 1.4, 1.4]
    if len(headers) == 8:
        widths = [4.8, 1.4, 1.5, 3.2, 1.4, 3.2, 1.4, 1.5]
    set_col_widths(table, widths)

    for idx, header in enumerate(headers):
        add_text(table.rows[0].cells[idx], header, bold=True, fill="F2F2F2", align="left" if idx == 0 else "center")
    for ridx, (_, row) in enumerate(df.iterrows(), start=1):
        for cidx, header in enumerate(headers):
            add_text(table.rows[ridx].cells[cidx], row[header], align="left" if cidx == 0 else "center")
    set_table_borders(table)

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(6)
    note.paragraph_format.space_after = Pt(0)
    run = note.add_run(note_text)
    style_run(run, size=6.5)
    doc.save(out_path)


def main() -> None:
    df = pd.read_csv(DATA)
    df[OUTCOME] = df[OUTCOME].astype(int)

    full_vars = BASE_A + PF_48H + VR_48H
    primary_row, primary_or = fit_complete_case(df, "primary_cc", full_vars, ("vr_initial", "vr_delta"))

    mice_row, mice_or, mice_perf = fit_mice(df, full_vars)

    no_early = df.loc[~((df[OUTCOME] == 1) & (df["los_icu_days"] <= 2))].copy()
    early_row, early_or = fit_complete_case(no_early, "exclude_48h_deaths", full_vars, ("vr_initial", "vr_delta"))

    h24 = build_24h_dataset(df)
    vars_24h = BASE_A + ["pf_initial_24h", "pf_min_24h", "vr_initial_24h", "vr_delta_24h"]
    h24_row, h24_or = fit_complete_case(h24, "first_24h", vars_24h, ("vr_initial_24h", "vr_delta_24h"))

    measurement_vars = full_vars + MEASUREMENT_ADJUSTERS
    meas_row, meas_or = fit_complete_case(df, "adjust_measurement", measurement_vars, ("vr_initial", "vr_delta"))

    table4 = pd.DataFrame([primary_row, mice_row, early_row, h24_row, meas_row])
    table4 = table4[[
        "Analysis", "N", "Deaths", "VR initial OR (95% CI)", "VR initial P value",
        "VR delta OR (95% CI)", "VR delta P value", "AUROC", "Brier",
    ]]

    strata_specs = [
        ("3-4 paired measurements", df.loc[df["n_points"].between(3, 4)].copy()),
        ("5-7 paired measurements", df.loc[df["n_points"].between(5, 7)].copy()),
        (">=8 paired measurements", df.loc[df["n_points"] >= 8].copy()),
    ]
    strata_rows = []
    strata_or_parts = []
    for label, sub in strata_specs:
        row, or_df = fit_complete_case(sub, label, full_vars, ("vr_initial", "vr_delta"))
        row["Measurement-count stratum"] = label
        strata_rows.append(row)
        strata_or_parts.append(or_df)
    strata = pd.DataFrame(strata_rows)[[
        "Measurement-count stratum", "N", "Deaths", "VR initial OR (95% CI)", "VR initial P value",
        "VR delta OR (95% CI)", "VR delta P value", "AUROC", "Brier",
    ]]

    table4.to_csv(OUTDIR / "Table4_sensitivity_analyses.csv", index=False)
    strata.to_csv(OUTDIR / "TableS_measurement_count_strata.csv", index=False)
    pd.concat([primary_or, early_or, h24_or, meas_or], ignore_index=True).to_csv(
        OUTDIR / "Table4_complete_case_OR_details.csv", index=False
    )
    mice_or.to_csv(OUTDIR / "Table4_mice_pooled_OR_details.csv", index=False)
    mice_perf.to_csv(OUTDIR / "Table4_mice_iteration_performance.csv", index=False)
    pd.concat(strata_or_parts, ignore_index=True).to_csv(OUTDIR / "TableS_measurement_count_OR_details.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "Table4_and_measurement_strata.xlsx") as writer:
        table4.to_excel(writer, sheet_name="Table4", index=False)
        strata.to_excel(writer, sheet_name="Measurement_strata", index=False)
        mice_or.to_excel(writer, sheet_name="MICE_OR_details", index=False)

    table4_note = (
        "Sensitivity analyses used the MIMIC-IV internal fully adjusted model. The primary complete-case model included age, sex, "
        "Charlson comorbidity index, sepsis, COPD, congestive heart failure, SOFA score, PEEP, driving pressure, tidal volume per predicted body weight, heart rate, "
        "mean arterial pressure, temperature, lactate, blood urea nitrogen, creatinine, white blood cell count, bicarbonate, initial P/F ratio, "
        "minimum P/F ratio, VR initial, and VR delta. Odds ratios are reported per 1-SD increase. Multiple imputation used chained equations "
        "with m=10 imputations for the same predictor set. The first 24 h analysis recalculated VR initial, VR delta, initial P/F ratio, and minimum P/F ratio "
        "using paired observations within 0-24 h after ICU admission and required at least two paired measurements in that window. The measurement-adjusted model "
        "additionally included the number of paired measurements and the VR observation time span. Brier score is shown as a secondary overall prediction error metric."
    )
    strata_note = (
        "Patients were stratified according to the number of paired P/F-VR measurements obtained within the first 48 h after ICU admission. "
        "Each stratum used the same fully adjusted complete-case model as the primary analysis, including COPD, congestive heart failure, initial and minimum P/F ratio, VR initial, and VR delta. "
        "Odds ratios are reported per 1-SD increase within each stratum. This analysis evaluates whether the VR-mortality association is consistent across different measurement frequencies."
    )

    build_docx(
        table4,
        "Table 4. Sensitivity analyses for the association between early VR features and hospital mortality.",
        table4_note,
        OUTDIR / "Table4_sensitivity_analyses_TNR_xiaoliu.docx",
    )
    build_docx(
        strata,
        "Supplementary Table. Stratified analyses by the number of paired P/F-VR measurements.",
        strata_note,
        OUTDIR / "TableS_measurement_count_strata_TNR_xiaoliu.docx",
    )

    lines = [
        "# Table 4 and measurement-count stratified table generation log",
        "",
        f"Input cohort: {DATA}",
        f"Point-level data for 24h analysis: {POINTS_LONG}",
        "",
        "## Table 4",
        "",
        table4.to_markdown(index=False),
        "",
        "## Measurement-count strata",
        "",
        strata.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Table 4 was recalculated under the revised low-missingness, fully adjusted MIMIC-IV internal model with COPD, CHF, P/F initial, and P/F minimum.",
        "- The measurement strata are 3-4, 5-7, and >=8 paired P/F-VR measurements within 48h.",
        "- Original source files were not overwritten.",
    ]
    (OUTDIR / "Table4_generation_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Table 4 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
