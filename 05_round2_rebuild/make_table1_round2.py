#!/usr/bin/env python3
"""Create formatted Table 1 for the revised manuscript.

Layout: one Table 1 with two vertical panels:
Panel A, MIMIC-IV internal complete-case cohort; Panel B, eICU external
validation complete-case cohort.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTDIR.mkdir(exist_ok=True)
MIMIC_PATH = OUTDIR / "mimic_analysis_ready_round2.csv"
EICU_PATH = OUTDIR / "eicu_analysis_ready_round2.csv"
OUTCOME = "hospital_expire_flag"

MODEL_A_VARS = [
    "age", "sex", "charlson", "sepsis3", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate", "vr_initial", "vr_delta",
]
MODEL_B_VARS = [
    "age", "sex", "charlson", "sepsis3", "peep", "vt_pbw", "hr",
    "map", "temp", "bun", "creatinine", "wbc", "bicarbonate",
    "vr_initial", "vr_delta",
]


VARIABLES = [
    {"label": "Age, years", "var": "age", "type": "continuous", "force": "mean"},
    {"label": "Male, n (%)", "var": "sex", "type": "binary"},
    {"label": "BMI, kg/m2", "var": "bmi", "type": "continuous", "force": "mean"},
    {"label": "White race, n (%)", "var": "race_grouped", "type": "category", "level": "White"},
    {"label": "SOFA score", "var": "sofa", "type": "continuous", "panel": "MIMIC", "force": "median"},
    {"label": "APACHE-IV score", "var": "apache_iv", "type": "continuous", "panel": "eICU", "force": "mean"},
    {"label": "Charlson comorbidity index", "var": "charlson", "type": "continuous", "force": "median"},
    {"label": "Sepsis, n (%)", "var": "sepsis3", "type": "binary"},
    {"label": "COPD, n (%)", "var": "copd", "type": "binary"},
    {"label": "Congestive heart failure, n (%)", "var": "chf", "type": "binary"},
    {"label": "PEEP, cmH2O", "var": "peep", "type": "continuous", "force": "median"},
    {"label": "Driving pressure, cmH2O", "var": "driving_pressure", "type": "continuous", "panel": "MIMIC", "force": "median"},
    {"label": "Tidal volume/PBW, mL/kg", "var": "vt_pbw", "type": "continuous", "force": "mean"},
    {"label": "Heart rate, beats/min", "var": "hr", "type": "continuous", "force": "mean"},
    {"label": "MAP, mmHg", "var": "map", "type": "continuous", "force": "mean"},
    {"label": "Temperature, °C", "var": "temp", "type": "continuous", "force": "mean"},
    {"label": "Lactate, mmol/L", "var": "lactate", "type": "continuous", "panel": "MIMIC", "force": "median"},
    {"label": "BUN, mg/dL", "var": "bun", "type": "continuous", "force": "median"},
    {"label": "Creatinine, mg/dL", "var": "creatinine", "type": "continuous", "force": "median"},
    {"label": "WBC, 10^9/L", "var": "wbc", "type": "continuous", "force": "median"},
    {"label": "Bicarbonate, mEq/L", "var": "bicarbonate", "type": "continuous", "force": "mean"},
    {"label": "VR initial", "var": "vr_initial", "type": "continuous", "force": "median"},
    {"label": "VR delta", "var": "vr_delta", "type": "continuous", "force": "median"},
    {"label": "P/F initial, mmHg", "var": "pf_initial", "type": "continuous", "force": "mean"},
    {"label": "P/F minimum, mmHg", "var": "pf_min", "type": "continuous", "force": "mean"},
    {"label": "PaCO2 initial, mmHg", "var": "paco2_initial", "type": "continuous", "force": "median"},
    {"label": "pH initial", "var": "ph_initial", "type": "continuous", "force": "mean"},
    {"label": "Paired P/F-VR measurements, n", "var": "n_points", "type": "continuous", "force": "median"},
]


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def clean_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()


def choose_summary(x: pd.Series, force: str | None = None) -> str:
    vals = clean_series(x)
    if len(vals) == 0:
        return "-"
    if force == "mean":
        return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
    if force == "median":
        return f"{vals.median():.2f} ({vals.quantile(0.25):.2f}-{vals.quantile(0.75):.2f})"
    skew = vals.skew()
    if np.isfinite(skew) and abs(skew) <= 1:
        return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}"
    return f"{vals.median():.2f} ({vals.quantile(0.25):.2f}-{vals.quantile(0.75):.2f})"


def p_cont(df: pd.DataFrame, var: str, force: str | None = None) -> str:
    a = clean_series(df.loc[df[OUTCOME] == 0, var])
    d = clean_series(df.loc[df[OUTCOME] == 1, var])
    if len(a) < 2 or len(d) < 2:
        return ""
    use_median = force == "median" or (force is None and abs(pd.concat([a, d]).skew()) > 1)
    if use_median:
        return fmt_p(stats.mannwhitneyu(a, d, alternative="two-sided").pvalue)
    return fmt_p(stats.ttest_ind(a, d, equal_var=False, nan_policy="omit").pvalue)


def fmt_cat(n: int, denom: int) -> str:
    return f"{n} ({n / denom * 100:.1f})" if denom else "-"


def p_cat(df: pd.DataFrame, var: str, level=None) -> str:
    if level is None:
        x = df[var].fillna(0)
    else:
        x = (df[var].fillna("Other/Unknown") == level).astype(int)
    ct = pd.crosstab(x, df[OUTCOME])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ""
    return fmt_p(stats.chi2_contingency(ct)[1])


def table_rows(df: pd.DataFrame, panel: str) -> list[list[str]]:
    alive = df[df[OUTCOME] == 0]
    dead = df[df[OUTCOME] == 1]
    rows = []
    for item in VARIABLES:
        item_panel = item.get("panel")
        if item_panel and item_panel != panel:
            continue
        var = item["var"]
        label = item["label"]
        if var not in df.columns:
            continue
        if item["type"] == "continuous":
            force = item.get("force")
            rows.append([
                label,
                choose_summary(df[var], force),
                choose_summary(alive[var], force),
                choose_summary(dead[var], force),
                p_cont(df, var, force),
            ])
        elif item["type"] == "binary":
            rows.append([
                label,
                fmt_cat(int((df[var] == 1).sum()), len(df)),
                fmt_cat(int((alive[var] == 1).sum()), len(alive)),
                fmt_cat(int((dead[var] == 1).sum()), len(dead)),
                p_cat(df, var),
            ])
        elif item["type"] == "category":
            level = item["level"]
            rows.append([
                label,
                fmt_cat(int((df[var].fillna("Other/Unknown") == level).sum()), len(df)),
                fmt_cat(int((alive[var].fillna("Other/Unknown") == level).sum()), len(alive)),
                fmt_cat(int((dead[var].fillna("Other/Unknown") == level).sum()), len(dead)),
                p_cat(df, var, level),
            ])
    return rows


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


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


def set_cell_margins(cell, top=60, start=80, bottom=60, end=80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
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
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if align == "left" else WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
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


def set_col_widths(table, widths_cm):
    for row in table.rows:
        for idx, width in enumerate(widths_cm):
            cell = row.cells[idx]
            cell.width = Cm(width)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 567)))
            tc_w.set(qn("w:type"), "dxa")


def add_panel(doc: Document, title: str, rows: list[list[str]], n_total: int, n_alive: int, n_dead: int) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title)
    style_run(r, size=6.5, bold=True)

    table = doc.add_table(rows=1, cols=5)
    table.autofit = False
    headers = ["Variable", f"Overall\n(N={n_total})", f"Survivors\n(n={n_alive})", f"Non-survivors\n(n={n_dead})", "P value"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        style_cell(table.rows[0].cells[i], bold=True, align="left" if i == 0 else "center", fill="F2F2F2")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
            style_cell(cells[i], align="left" if i == 0 else "center")
    set_col_widths(table, [5.0, 3.2, 3.2, 3.2, 1.6])
    set_table_borders(table)


def make_docx(mimic_rows, eicu_rows, mimic, eicu) -> None:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)

    title = doc.add_paragraph()
    title.paragraph_format.space_after = Pt(6)
    run = title.add_run("Table 1. Baseline characteristics of the study cohorts.")
    style_run(run, size=9, bold=True)

    add_panel(
        doc,
        "Panel A. MIMIC-IV internal complete-case cohort",
        mimic_rows,
        len(mimic),
        int((mimic[OUTCOME] == 0).sum()),
        int((mimic[OUTCOME] == 1).sum()),
    )
    doc.add_page_break()
    add_panel(
        doc,
        "Panel B. eICU external validation complete-case cohort",
        eicu_rows,
        len(eicu),
        int((eicu[OUTCOME] == 0).sum()),
        int((eicu[OUTCOME] == 1).sum()),
    )

    note = doc.add_paragraph()
    note.paragraph_format.space_before = Pt(4)
    note.paragraph_format.space_after = Pt(0)
    text = (
        "Data are expressed as the mean ± SD, median (interquartile range), or n (%). "
        "P values compare survivors and non-survivors within each cohort. "
        "SOFA was available in MIMIC-IV, whereas APACHE-IV was available in eICU; these severity scores were not treated as interchangeable. "
        "Albumin and bilirubin were excluded from the main baseline/model tables because of missingness >=20%."
    )
    r = note.add_run(text)
    style_run(r, size=6.5)

    doc.save(OUTDIR / "Table1_baseline_two_panel_TNR_xiaoliu.docx")


def make_excel(mimic_rows, eicu_rows, mimic, eicu) -> None:
    with pd.ExcelWriter(OUTDIR / "Table1_baseline_two_panel.xlsx", engine="openpyxl") as writer:
        for sheet, rows, df in [
            ("Panel A MIMIC", mimic_rows, mimic),
            ("Panel B eICU", eicu_rows, eicu),
        ]:
            alive = int((df[OUTCOME] == 0).sum())
            dead = int((df[OUTCOME] == 1).sum())
            headers = ["Variable", f"Overall (N={len(df)})", f"Survivors (n={alive})", f"Non-survivors (n={dead})", "P value"]
            pd.DataFrame(rows, columns=headers).to_excel(writer, index=False, sheet_name=sheet)


def main() -> None:
    mimic = pd.read_csv(MIMIC_PATH)
    eicu = pd.read_csv(EICU_PATH)
    mimic = mimic.dropna(subset=[OUTCOME] + MODEL_A_VARS).copy()
    eicu = eicu.dropna(subset=[OUTCOME] + MODEL_B_VARS).copy()

    mimic_rows = table_rows(mimic, "MIMIC")
    eicu_rows = table_rows(eicu, "eICU")
    make_docx(mimic_rows, eicu_rows, mimic, eicu)
    make_excel(mimic_rows, eicu_rows, mimic, eicu)

    pd.DataFrame(mimic_rows, columns=["Variable", "Overall", "Survivors", "Non-survivors", "P value"]).to_csv(
        OUTDIR / "Table1_panelA_MIMIC.csv", index=False
    )
    pd.DataFrame(eicu_rows, columns=["Variable", "Overall", "Survivors", "Non-survivors", "P value"]).to_csv(
        OUTDIR / "Table1_panelB_eICU.csv", index=False
    )
    (OUTDIR / "Table1_generation_log.md").write_text(
        "Table 1 generated as two stacked panels to avoid an overly wide six-to-nine-column table. "
        "Font in DOCX is Times New Roman 6.5 pt for table body.\n",
        encoding="utf-8",
    )
    print(f"Wrote Table 1 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
