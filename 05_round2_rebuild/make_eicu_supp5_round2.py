#!/usr/bin/env python3
"""Create supplementary eICU association table for VR initial/delta."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTDIR.mkdir(exist_ok=True)
DATA = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "eicu_analysis_ready_round2.csv"
OUTCOME = "hospital_expire_flag"

VR = ["vr_initial", "vr_delta"]
CORE = ["age", "sex", "charlson", "sepsis3", "copd", "chf"]  # round-2 harmonized spec; sepsis3 IS the admitDx def
FULL = CORE + [
    "peep", "vt_pbw", "hr", "map", "temp", "bun", "creatinine",
    "wbc", "bicarbonate", "pf_initial", "pf_min",
]
MODELS = {
    "Model 1": VR,
    "Model 2": VR + CORE,
    "Model 3": VR + FULL,
}


def p_from_z(z: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_or(or_: float, lo: float, hi: float) -> str:
    return f"{or_:.3f} ({lo:.3f}-{hi:.3f})"


def fit(df: pd.DataFrame, model_name: str, variables: list[str]) -> tuple[dict, pd.DataFrame]:
    cc = df.dropna(subset=[OUTCOME] + variables).copy()
    scaler = StandardScaler().fit(cc[variables])
    x = pd.DataFrame(scaler.transform(cc[variables]), columns=variables, index=cc.index)
    x = sm.add_constant(x, has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)

    rows = []
    for var in variables:
        beta = float(res.params[var])
        se = float(res.bse[var])
        rows.append({
            "model": model_name,
            "variable": var,
            "OR": math.exp(beta),
            "CI_lower": math.exp(beta - 1.96 * se),
            "CI_upper": math.exp(beta + 1.96 * se),
            "p_value": p_from_z(beta / se),
        })
    ors = pd.DataFrame(rows)
    out = {
        "Model": model_name,
        "N": int(len(cc)),
        "Deaths": int(y.sum()),
        "VR initial OR (95% CI)": fmt_or(
            ors.loc[ors.variable.eq("vr_initial"), "OR"].iloc[0],
            ors.loc[ors.variable.eq("vr_initial"), "CI_lower"].iloc[0],
            ors.loc[ors.variable.eq("vr_initial"), "CI_upper"].iloc[0],
        ),
        "VR initial P value": fmt_p(ors.loc[ors.variable.eq("vr_initial"), "p_value"].iloc[0]),
        "VR delta OR (95% CI)": fmt_or(
            ors.loc[ors.variable.eq("vr_delta"), "OR"].iloc[0],
            ors.loc[ors.variable.eq("vr_delta"), "CI_lower"].iloc[0],
            ors.loc[ors.variable.eq("vr_delta"), "CI_upper"].iloc[0],
        ),
        "VR delta P value": fmt_p(ors.loc[ors.variable.eq("vr_delta"), "p_value"].iloc[0]),
        "AUROC": f"{roc_auc_score(y, pred):.3f}",
        "Brier": f"{brier_score_loss(y, pred):.3f}",
        "Variables": ", ".join(variables),
    }
    return out, ors


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


def build_docx(table_df: pd.DataFrame) -> None:
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
    run = title.add_run("Supplementary Table 5. Association between early VR features and hospital mortality in eICU.")
    style_run(run, size=9, bold=True)

    headers = [c for c in table_df.columns if c != "Variables"]
    table = doc.add_table(rows=1 + len(table_df), cols=len(headers))
    table.autofit = False
    set_col_widths(table, [2.0, 1.3, 1.4, 3.2, 1.4, 3.2, 1.4, 1.4, 1.4])
    for idx, header in enumerate(headers):
        add_text(table.rows[0].cells[idx], header, bold=True, fill="F2F2F2", align="left" if idx == 0 else "center")
    for ridx, (_, row) in enumerate(table_df.iterrows(), start=1):
        for cidx, header in enumerate(headers):
            add_text(table.rows[ridx].cells[cidx], row[header], align="left" if cidx == 0 else "center")
    set_table_borders(table)

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(6)
    note.paragraph_format.space_after = Pt(0)
    note_text = (
        "Analyses were conducted in the eICU external validation cohort using complete cases. "
        "Model 1 included VR initial and VR delta. Model 2 additionally adjusted for age, sex, Charlson comorbidity index, sepsis (recorded admission diagnosis), COPD, and congestive heart failure. "
        "Model 3 additionally adjusted for PEEP, tidal volume per predicted body weight, heart rate, mean arterial pressure, temperature, blood urea nitrogen, creatinine, "
        "white blood cell count, bicarbonate, initial P/F ratio, and minimum P/F ratio. Odds ratios are reported per 1-SD increase. "
        "This supplementary analysis was used to assess directional consistency of the VR-mortality association in the external validation cohort."
    )
    run = note.add_run(note_text)
    style_run(run, size=6.5)
    doc.save(OUTDIR / "TableS_eICU_VR_association_TNR_xiaoliu.docx")


def main() -> None:
    df = pd.read_csv(DATA)
    # R2-2 harmonization (idempotent): harmonized-model vitals = eICU NATIVE first-day means.
    # If the ready file already carries native values under the standard names this is a no-op;
    # if it still has the legacy APS/extreme columns, they are replaced here.
    if all(c in df.columns for c in ("hr_native", "map_native", "temp_native")):
        _poll = [c for c in ["hr", "map", "temp", "rr_vitalsign", "spo2"] if c in df.columns]
        if _poll:
            df = df.drop(columns=_poll)
        for _a, _b in {"hr_native": "hr", "map_native": "map", "temp_native": "temp"}.items():
            df = df.rename(columns={_a: _b})
    rows = []
    ors = []
    for model_name, variables in MODELS.items():
        row, or_df = fit(df, model_name, variables)
        rows.append(row)
        ors.append(or_df)
    table = pd.DataFrame(rows)
    or_details = pd.concat(ors, ignore_index=True)

    table.to_csv(OUTDIR / "TableS_eICU_VR_association.csv", index=False)
    or_details.to_csv(OUTDIR / "TableS_eICU_VR_association_OR_details.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "TableS_eICU_VR_association.xlsx") as writer:
        table.to_excel(writer, sheet_name="eICU_VR_association", index=False)
        or_details.to_excel(writer, sheet_name="OR_details", index=False)
    build_docx(table)

    lines = [
        "# eICU VR association supplementary table log",
        "",
        f"Input: {DATA}",
        "",
        table.to_markdown(index=False),
        "",
        "## Model definitions",
        "",
        "- Model 1: VR initial + VR delta.",
        "- Model 2: Model 1 + age, sex, Charlson comorbidity index, sepsis.",
        "- Model 3: Model 2 + PEEP, VT/PBW, heart rate, MAP, temperature, BUN, creatinine, WBC, bicarbonate, P/F initial, P/F minimum.",
    ]
    (OUTDIR / "TableS_eICU_VR_association_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote eICU supplementary table to {OUTDIR}")


if __name__ == "__main__":
    main()
