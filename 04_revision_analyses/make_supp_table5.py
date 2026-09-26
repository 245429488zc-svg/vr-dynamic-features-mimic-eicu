#!/usr/bin/env python3
"""Supplementary Table 5: Temporal definitions of covariates (R1.4 / R2.9).

Rules transcribed from the cohort-construction and covariate-derivation scripts
(enhanced_covariates_analysis.py; MIMIC-IV v3.1 derived tables; eICU-CRD v2.0
respiratorycharting / labsfirstday / apacheapsvar / charlson / admissiondx).
"""
from pathlib import Path
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Cm, Pt

OUT = Path("./34_协变量时间定义")
OUT.mkdir(exist_ok=True)

MIMIC = [
    ("Age, sex", "ICU admission", "Fixed demographic characteristics"),
    ("Charlson comorbidity index", "Admission (pre-ICU history)", "Weighted comorbidity score"),
    ("Sepsis", "Admission / first stay", "Sepsis-3 criteria (suspected infection plus SOFA change)"),
    ("COPD, congestive heart failure", "Pre-ICU history", "Documented comorbidity diagnoses"),
    ("SOFA score", "First 24 h", "24-hour SOFA score computed from first-24 h data"),
    ("Heart rate, mean arterial pressure, temperature", "First 24 h (hospital day 1)", "Mean of documented vital-sign values"),
    ("Lactate, BUN, creatinine, white blood cell count", "First 24 h (hospital day 1)", "Maximum documented value"),
    ("Bicarbonate", "First 24 h (hospital day 1)", "Minimum documented value"),
    ("PEEP", "0-48 h", "Mean of documented ventilator settings"),
    ("Driving pressure", "0-48 h", "Mean plateau pressure minus mean PEEP"),
    ("Vt/PBW", "0-48 h", "Mean set tidal volume divided by predicted body weight"),
    ("P/F initial", "0-48 h", "First valid paired P/F value"),
    ("P/F minimum", "0-48 h", "Minimum valid paired P/F value"),
    ("VR features (eight)", "0-48 h", "Derived from all valid paired P/F-VR measurements (see main text)"),
]
EICU = [
    ("Age, sex", "ICU admission", "Admission age; recorded sex"),
    ("Charlson comorbidity index", "Admission (pre-ICU history)", "Weighted comorbidity score"),
    ("Sepsis", "ICU admission", "Admission diagnosis of sepsis"),
    ("Heart rate, mean arterial pressure, temperature", "ICU admission (first hour)", "APACHE admission vital-sign variables"),
    ("PEEP", "0-48 h", "Mean of documented ventilator charting values"),
    ("Vt/PBW", "0-48 h", "Mean set tidal volume divided by predicted body weight"),
    ("BUN, creatinine, white blood cell count", "First 24 h (hospital day 1)", "Maximum documented value"),
    ("Bicarbonate", "First 24 h (hospital day 1)", "Minimum documented value"),
    ("P/F initial, P/F minimum", "0-48 h", "First / minimum valid paired P/F value"),
    ("VR features (eight)", "0-48 h", "Derived from all valid paired P/F-VR measurements (see main text)"),
]

doc = Document()
sec = doc.sections[0]
w, h = sec.page_width, sec.page_height
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = h, w
t0 = doc.add_paragraph()
r0 = t0.add_run("Supplementary Table 5. Temporal definitions of covariates.")
r0.font.name = "Times New Roman"; r0.font.size = Pt(9); r0.font.bold = True
n1 = doc.add_paragraph()
rr = n1.add_run(
    "Covariates were determined as listed below. Fixed characteristics were determined at admission; early-period variables were "
    "determined within the stated windows of the first 48 hours after ICU admission. Because several covariates and the VR features "
    "were collected during the same early period, covariate adjustment represents adjustment for concurrent early-period variables "
    "rather than strictly temporally preceding baseline characteristics; all covariates were, however, determined by the 48-hour "
    "landmark time used in the sensitivity analyses. In the eICU harmonized model, SOFA, COPD, congestive heart failure, driving "
    "pressure, and lactate were not included.")
rr.font.name = "Times New Roman"; rr.font.size = Pt(6.5)

tbl_data = [["Variable", "Time window", "Selection rule", ""]]
cur = "MIMIC-IV (derivation cohort)"
tbl_data.append([cur, "", "", ""])
for var, win, rule in MIMIC:
    tbl_data.append([var, win, rule, ""])
cur = "eICU (external validation cohort; harmonized model)"
tbl_data.append([cur, "", "", ""])
for var, win, rule in EICU:
    tbl_data.append([var, win, rule, ""])
# 去掉占位第4列：改为3列
tbl_data = [[r[0], r[1], r[2]] for r in tbl_data]
t = doc.add_table(rows=len(tbl_data), cols=3)
t.style = doc.styles["Table Grid"]
widths = [Cm(14), Cm(7), Cm(15)]
for i, row in enumerate(tbl_data):
    for j, val in enumerate(row):
        cell = t.rows[i].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(val))
        run.font.name = "Times New Roman"; run.font.size = Pt(6.5)
        run.font.bold = (i == 0) or str(row[0]).startswith(("MIMIC", "eICU"))
        cell.width = widths[j]
doc.save(OUT / "Supplementary Table 5.docx")
print(f"已保存: {OUT / 'Supplementary Table 5.docx'}（{len(tbl_data)-2}行定义，两库面板）")