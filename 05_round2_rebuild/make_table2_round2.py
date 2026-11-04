#!/usr/bin/env python3
"""Create formatted Table 2: early VR features and hospital mortality."""

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
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(exist_ok=True)
MIMIC_PATH = Path(__file__).resolve().parent / "mimic_analysis_ready_round2.csv"
OUTCOME = "hospital_expire_flag"

VR_FEATURES = [
    "vr_initial",
    "vr_mean",
    "vr_max",
    "vr_min",
    "vr_last",
    "vr_auc",
    "vr_delta_adj_initial",
    "vr_slope",
]

CORE_ADJUSTERS = [
    "age", "sex", "charlson", "sepsis3", "copd", "chf",
]

FULL_ADJUSTERS = CORE_ADJUSTERS + [
    "sofa", "peep", "driving_pressure", "vt_pbw", "hr", "map",
    "temp", "lactate", "bun", "creatinine", "wbc", "bicarbonate",
    "pf_initial", "pf_min",
]

MODEL_SPECS = {
    "Model 1": {
        "description": "Unadjusted model",
        "adjusters": [],
    },
    "Model 2": {
        "description": "Adjusted for demographics and comorbidities",
        "adjusters": CORE_ADJUSTERS,
    },
    "Model 3": {
        "description": (
            "Adjusted for demographics, comorbidities, severity score, ventilator "
            "parameters, vital signs, laboratory tests, and P/F indices"
        ),
        "adjusters": FULL_ADJUSTERS,
    },
}

DISPLAY = {
    "vr_initial": "VR initial, per 1-SD increase",
    "vr_mean": "VR mean, per 1-SD increase",
    "vr_max": "VR maximum, per 1-SD increase",
    "vr_min": "VR minimum, per 1-SD increase",
    "vr_last": "VR last, per 1-SD increase",
    "vr_auc": "VR AUC, per 1-SD increase",
    "vr_delta": "VR delta, per 1-SD increase",
    "vr_delta_adj_initial": "VR delta adjusted for VR initial, per 1-SD increase",
    "vr_slope": "VR slope, per 1-SD increase",
    "age": "Age, years",
    "sex": "Male sex",
    "charlson": "Charlson comorbidity index",
    "sepsis3": "Sepsis",
    "copd": "COPD",
    "chf": "Congestive heart failure",
    "sofa": "SOFA score",
    "peep": "PEEP, cmH2O",
    "driving_pressure": "Driving pressure, cmH2O",
    "vt_pbw": "Tidal volume/PBW, mL/kg",
    "hr": "Heart rate, beats/min",
    "map": "MAP, mmHg",
    "temp": "Temperature, C",
    "lactate": "Lactate, mmol/L",
    "bun": "BUN, mg/dL",
    "creatinine": "Creatinine, mg/dL",
    "wbc": "WBC, 10^9/L",
    "bicarbonate": "Bicarbonate, mEq/L",
    "pf_initial": "P/F initial, mmHg",
    "pf_min": "P/F minimum, mmHg",
}


def p_from_z(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_or(row: pd.Series) -> str:
    return f"{row['OR']:.3f} ({row['CI_lower']:.3f}-{row['CI_upper']:.3f})"


def fit_model(df: pd.DataFrame, model_name: str, vr_feature: str, adjusters: list[str]) -> tuple[pd.DataFrame, dict]:
    display_feature = vr_feature
    if vr_feature == "vr_delta_adj_initial":
        variables = ["vr_delta", "vr_initial"] + adjusters
        coefficient_feature = "vr_delta"
    else:
        variables = [vr_feature] + adjusters
        coefficient_feature = vr_feature
    cc = df.dropna(subset=[OUTCOME] + variables).copy()
    y = cc[OUTCOME].astype(int)

    scaler = StandardScaler().fit(cc[variables])
    x = pd.DataFrame(scaler.transform(cc[variables]), columns=variables, index=cc.index)
    x = sm.add_constant(x, has_constant="add")

    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    rows = []
    for var in variables:
        beta = float(res.params[var])
        se = float(res.bse[var])
        rows.append({
            "model": model_name,
            "target_feature": display_feature,
            "variable": var,
            "variable_label": DISPLAY.get(var, var),
            "OR": math.exp(beta),
            "CI_lower": math.exp(beta - 1.96 * se),
            "CI_upper": math.exp(beta + 1.96 * se),
            "p_value": p_from_z(beta / se),
        })

    meta = {
        "model": model_name,
        "vr_feature": display_feature,
        "vr_feature_label": DISPLAY.get(display_feature, display_feature),
        "coefficient_feature": coefficient_feature,
        "description": MODEL_SPECS[model_name]["description"],
        "n": int(len(cc)),
        "events": int(y.sum()),
        "non_events": int((1 - y).sum()),
        "variables": ", ".join(variables),
    }
    return pd.DataFrame(rows), meta


def set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
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
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


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


def add_text(cell, text: str, bold=False, align="center", fill=None) -> None:
    cell.text = str(text)
    style_cell(cell, bold=bold, align=align, fill=fill)


def make_display_table(vr_results: pd.DataFrame, metas: list[dict]) -> pd.DataFrame:
    lookup = {}
    for _, row in vr_results.iterrows():
        lookup[(row["target_feature"], row["model"], "or")] = fmt_or(row)
        lookup[(row["target_feature"], row["model"], "p")] = fmt_p(float(row["p_value"]))

    meta_lookup = {(m["vr_feature"], m["model"]): m for m in metas}

    n_row = {"Early VR feature": "N"}
    for meta in metas:
        pass

    rows = []
    for var in VR_FEATURES:
        row = {"Early VR feature": DISPLAY[var]}
        for model_name in MODEL_SPECS:
            row[f"{model_name} OR (95% CI)"] = lookup[(var, model_name, "or")]
            row[f"{model_name} P value"] = lookup[(var, model_name, "p")]
            row[f"{model_name} N"] = meta_lookup[(var, model_name)]["n"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_docx(table_df: pd.DataFrame, metas: list[dict]) -> None:
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
    run = title.add_run("Table 2. Association of early ventilatory ratio features with hospital mortality.")
    style_run(run, size=9, bold=True)

    table = doc.add_table(rows=2 + len(table_df), cols=10)
    table.autofit = False
    widths = [4.1, 1.4, 3.0, 1.4, 1.4, 3.0, 1.4, 1.4, 3.0, 1.4]
    set_col_widths(table, widths)

    headers = ["", "Model 1", "", "", "Model 2", "", "", "Model 3", "", ""]
    subheaders = [
        "Early VR feature",
        "N", "OR (95% CI)", "P value",
        "N", "OR (95% CI)", "P value",
        "N", "OR (95% CI)", "P value",
    ]
    for idx, text in enumerate(headers):
        add_text(table.rows[0].cells[idx], text, bold=True, fill="F2F2F2")
    for start in [1, 4, 7]:
        merged = table.rows[0].cells[start].merge(table.rows[0].cells[start + 2])
        add_text(merged, headers[start], bold=True, fill="F2F2F2")
    for idx, text in enumerate(subheaders):
        add_text(table.rows[1].cells[idx], text, bold=True, fill="F2F2F2", align="left" if idx == 0 else "center")

    for ridx, (_, row) in enumerate(table_df.iterrows(), start=2):
        values = [
            row["Early VR feature"],
            row["Model 1 N"],
            row["Model 1 OR (95% CI)"],
            row["Model 1 P value"],
            row["Model 2 N"],
            row["Model 2 OR (95% CI)"],
            row["Model 2 P value"],
            row["Model 3 N"],
            row["Model 3 OR (95% CI)"],
            row["Model 3 P value"],
        ]
        for cidx, value in enumerate(values):
            add_text(table.rows[ridx].cells[cidx], value, bold=(ridx == 2), align="left" if cidx == 0 else "center")

    set_table_borders(table)

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(6)
    note.paragraph_format.space_after = Pt(0)
    note_text = (
        "Data are odds ratios per 1-SD increase with 95% confidence intervals. "
        "Each early VR feature was entered into a separate logistic regression model. "
        "Because VR delta is baseline-dependent, the VR delta model additionally included VR initial. "
        "Model 1 was unadjusted except for VR initial in the VR delta row. Model 2 adjusted for age, sex, "
        "Charlson comorbidity index, sepsis, COPD, and congestive heart failure. "
        "Model 3 additionally adjusted for SOFA score, PEEP, driving pressure, tidal volume per predicted body weight, "
        "heart rate, mean arterial pressure, temperature, lactate, blood urea nitrogen, creatinine, white blood cell count, "
        "bicarbonate, initial P/F ratio, and minimum P/F ratio. "
        "Complete-case samples were defined separately for each model according to the variables included."
    )
    run = note.add_run(note_text)
    style_run(run, size=6.5)

    doc.save(OUTDIR / "Table2_early_VR_mortality_TNR_xiaoliu.docx")


def main() -> None:
    mimic = pd.read_csv(MIMIC_PATH)
    all_results = []
    metas = []
    for vr_feature in VR_FEATURES:
        for model_name, spec in MODEL_SPECS.items():
            results, meta = fit_model(mimic, model_name, vr_feature, spec["adjusters"])
            coef_feature = meta["coefficient_feature"]
            vr_row = results[results["variable"] == coef_feature].copy()
            all_results.append(vr_row)
            metas.append(meta)

    all_or = pd.concat(all_results, ignore_index=True)
    all_or["OR_95CI"] = all_or.apply(fmt_or, axis=1)
    all_or["P_value"] = all_or["p_value"].map(fmt_p)
    vr_results = all_or[all_or["target_feature"].isin(VR_FEATURES)].copy()
    display = make_display_table(vr_results, metas)

    display.to_csv(OUTDIR / "Table2_early_VR_mortality.csv", index=False)
    all_or.to_csv(OUTDIR / "Table2_all_separate_VR_feature_OR.csv", index=False)
    pd.DataFrame(metas).to_csv(OUTDIR / "Table2_model_definitions_and_counts.csv", index=False)

    with pd.ExcelWriter(OUTDIR / "Table2_early_VR_mortality.xlsx") as writer:
        display.to_excel(writer, sheet_name="Table2_main", index=False)
        all_or.to_excel(writer, sheet_name="Separate_VR_OR", index=False)
        pd.DataFrame(metas).to_excel(writer, sheet_name="Model_definitions", index=False)

    build_docx(display, metas)

    lines = [
        "# Table 2 generation log",
        "",
        f"Input: {MIMIC_PATH}",
        f"Outcome: {OUTCOME}",
        "Each VR feature was fitted separately to avoid multicollinearity among VR-derived dynamic indices.",
        "Complete-case samples were defined separately for each model according to the variables included.",
        "",
        "## Model definitions and complete-case counts",
        "",
        pd.DataFrame(metas).to_markdown(index=False),
        "",
        "## Main Table 2 values",
        "",
        display.to_markdown(index=False),
        "",
        "## Output files",
        "",
        "- Table2_early_VR_mortality_TNR_xiaoliu.docx",
        "- Table2_early_VR_mortality.xlsx",
        "- Table2_early_VR_mortality.csv",
        "- Table2_all_separate_VR_feature_OR.csv",
        "- Table2_model_definitions_and_counts.csv",
    ]
    (OUTDIR / "Table2_generation_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Table 2 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
