#!/usr/bin/env python3
"""Create supplementary table comparing included and excluded early IMV patients.

Included patients are those in the final main analysis cohorts with >=3 valid
paired P/F-VR measurements within 48 h and hospital mortality available.
Excluded patients are early invasive mechanical ventilation patients who were not
included in those final cohorts, mainly because of insufficient paired
measurements or missing hospital mortality.
"""

from __future__ import annotations

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


ROOT = Path("/Volumes/tongtong/数据库/public/mimmc4/mimmc/项目4")
REVIEW = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUTDIR = Path(__file__).resolve().parent
RAW = ROOT / "原始数据"
MIMIC = RAW / "mimiciv_derived" / "mimiciv_derived"
EICU = RAW / "eICU" / "eICU"
FINAL_MIMIC = Path(__file__).resolve().parent / "mimic_analysis_ready_round2.csv"
FINAL_EICU = Path(__file__).resolve().parent / "eicu_analysis_ready_round2.csv"

WINDOW_H = 48
PAIR_WINDOW_H = 2.0
PF_MIN, PF_MAX = 30, 700
VR_MIN, VR_MAX = 0.2, 10
MV_MIN, MV_MAX = 1, 30
PCO2_MIN, PCO2_MAX = 10, 150


def norm_fio2(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    return pd.Series(np.where(y > 1.0, y / 100.0, y), index=y.index)


def pbw_from_height_sex(height_cm: pd.Series, sex: pd.Series) -> pd.Series:
    h = pd.to_numeric(height_cm, errors="coerce")
    s = sex.astype(str).str.upper().str[0]
    male = 50.0 + 0.91 * (h - 152.4)
    female = 45.5 + 0.91 * (h - 152.4)
    out = np.where(s.eq("M"), male, np.where(s.eq("F"), female, np.nan))
    return pd.Series(out, index=height_cm.index).where((h >= 120) & (h <= 230))


def race_group_mimic(race: pd.Series) -> pd.Series:
    r = race.astype(str).str.upper()
    return pd.Series(
        np.select(
            [
                r.str.contains("WHITE", na=False),
                r.str.contains("BLACK|AFRICAN", na=False),
                r.str.contains("ASIAN", na=False),
                r.str.contains("HISPANIC|LATINO", na=False),
            ],
            ["White", "Black", "Asian", "Hispanic/Latino"],
            default="Other/Unknown",
        ),
        index=race.index,
    )


def race_group_eicu(race: pd.Series) -> pd.Series:
    r = race.astype(str).str.upper()
    return pd.Series(
        np.select(
            [
                r.str.contains("CAUCASIAN|WHITE", na=False),
                r.str.contains("AFRICAN|BLACK", na=False),
                r.str.contains("ASIAN", na=False),
                r.str.contains("HISPANIC|LATINO", na=False),
            ],
            ["White", "Black", "Asian", "Hispanic/Latino"],
            default="Other/Unknown",
        ),
        index=race.index,
    )


def nearest_counts(bg: pd.DataFrame, mv: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if bg.empty or mv.empty:
        return pd.DataFrame(columns=[id_col, "n_points"])
    bg = bg.copy()
    mv = mv.copy()
    out = []
    for sid, sub_bg in bg.groupby(id_col, sort=False):
        sub_mv = mv[mv[id_col] == sid]
        if sub_mv.empty:
            continue
        mv_t = sub_mv["hour"].to_numpy(dtype=float)
        n = 0
        for t in sub_bg["hour"].to_numpy(dtype=float):
            if not np.isfinite(t) or len(mv_t) == 0:
                continue
            abs_d = np.abs(mv_t - t)
            if np.nanmin(abs_d) <= PAIR_WINDOW_H:
                n += 1
        if n:
            out.append({id_col: sid, "n_points": n})
    return pd.DataFrame(out)


def build_mimic_counts(early: pd.DataFrame) -> pd.DataFrame:
    bg = pd.read_csv(MIMIC / "bg.csv",
                     usecols=["subject_id", "hadm_id", "charttime", "po2", "pco2", "fio2", "pao2fio2ratio"])
    bg["charttime"] = pd.to_datetime(bg["charttime"], dayfirst=True, errors="coerce")
    bg = bg.merge(early[["stay_id", "subject_id", "hadm_id", "icu_intime"]], on=["subject_id", "hadm_id"], how="inner")
    bg["hour"] = (bg["charttime"] - bg["icu_intime"]).dt.total_seconds() / 3600
    bg = bg[bg["hour"].between(0, WINDOW_H)].copy()
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2fio2ratio"], errors="coerce")
    bg.loc[bg["pf"].isna(), "pf"] = pd.to_numeric(bg["po2"], errors="coerce") / bg["fio2_norm"]
    bg = bg[bg["pf"].between(PF_MIN, PF_MAX) & pd.to_numeric(bg["pco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)]

    mv = pd.read_csv(MIMIC / "ventilator_setting.csv", usecols=["stay_id", "charttime", "minute_volume"])
    mv["charttime"] = pd.to_datetime(mv["charttime"], dayfirst=True, errors="coerce")
    mv = mv.merge(early[["stay_id", "icu_intime"]], on="stay_id", how="inner")
    mv["hour"] = (mv["charttime"] - mv["icu_intime"]).dt.total_seconds() / 3600
    mv["minute_volume"] = pd.to_numeric(mv["minute_volume"], errors="coerce")
    mv = mv[mv["hour"].between(0, WINDOW_H) & mv["minute_volume"].between(MV_MIN, MV_MAX)]
    return nearest_counts(bg, mv, "stay_id")


def build_eicu_counts(early: pd.DataFrame) -> pd.DataFrame:
    bg = pd.read_csv(EICU / "pivoted_bg.csv", usecols=["patientunitstayid", "chartoffset", "fio2", "pao2", "paco2"])
    bg = bg[bg["patientunitstayid"].isin(early["stay_id"])]
    bg["hour"] = pd.to_numeric(bg["chartoffset"], errors="coerce") / 60
    bg["fio2_norm"] = norm_fio2(bg["fio2"])
    bg["pf"] = pd.to_numeric(bg["pao2"], errors="coerce") / bg["fio2_norm"]
    bg = bg[bg["hour"].between(0, WINDOW_H) & bg["pf"].between(PF_MIN, PF_MAX) &
            pd.to_numeric(bg["paco2"], errors="coerce").between(PCO2_MIN, PCO2_MAX)]
    bg = bg.rename(columns={"patientunitstayid": "stay_id"})

    labels = {"Exhaled MV", "Adult Con Pt/Vent MinuteVentil", "Measured Ve", "MVe"}
    chunks = []
    ids = set(early["stay_id"])
    for chunk in pd.read_csv(EICU / "respiratorycharting.csv",
                             usecols=["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"],
                             dtype={"respchartvalue": "string"}, chunksize=1_000_000):
        chunk = chunk[chunk["patientunitstayid"].isin(ids) & chunk["respchartvaluelabel"].isin(labels)]
        if chunk.empty:
            continue
        chunk["hour"] = pd.to_numeric(chunk["respchartoffset"], errors="coerce") / 60
        chunk["minute_volume"] = pd.to_numeric(chunk["respchartvalue"], errors="coerce")
        chunk = chunk[chunk["hour"].between(0, WINDOW_H) & chunk["minute_volume"].between(MV_MIN, MV_MAX)]
        chunks.append(chunk.rename(columns={"patientunitstayid": "stay_id"})[["stay_id", "hour", "minute_volume"]])
    mv = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["stay_id", "hour", "minute_volume"])
    return nearest_counts(bg, mv, "stay_id")


def build_mimic_early() -> pd.DataFrame:
    detail = pd.read_csv(MIMIC / "icustay_detail.csv")
    detail["icu_intime"] = pd.to_datetime(detail["icu_intime"], dayfirst=True, errors="coerce")
    detail = detail[(detail["first_icu_stay"].astype(str).str.lower().eq("t")) &
                    (pd.to_numeric(detail["admission_age"], errors="coerce") >= 18)].copy()
    vent = pd.read_csv(MIMIC / "ventilation.csv")
    vent["starttime"] = pd.to_datetime(vent["starttime"], dayfirst=True, errors="coerce")
    vent["endtime"] = pd.to_datetime(vent["endtime"], dayfirst=True, errors="coerce")
    inv = vent[vent["ventilation_status"].eq("InvasiveVent")].merge(detail[["stay_id", "icu_intime"]], on="stay_id", how="inner")
    inv["start_h"] = (inv["starttime"] - inv["icu_intime"]).dt.total_seconds() / 3600
    inv["end_h"] = (inv["endtime"] - inv["icu_intime"]).dt.total_seconds() / 3600
    early_ids = inv[(inv["start_h"] <= 24) & (inv["end_h"] >= 0)]["stay_id"].drop_duplicates()
    early = detail[detail["stay_id"].isin(early_ids)].copy()

    early = early.rename(columns={"admission_age": "age"})
    early["sex"] = early["gender"].map({"M": 1, "F": 0})
    early["hospital_expire_flag"] = pd.to_numeric(early["hospital_expire_flag"], errors="coerce")
    early["race_grouped"] = race_group_mimic(early["race"])
    height = pd.read_csv(MIMIC / "first_day_height.csv", usecols=["stay_id", "height"])
    weight = pd.read_csv(MIMIC / "first_day_weight.csv", usecols=["stay_id", "weight"])
    early = early.merge(height, on="stay_id", how="left").merge(weight, on="stay_id", how="left")
    early["bmi"] = pd.to_numeric(early["weight"], errors="coerce") / (pd.to_numeric(early["height"], errors="coerce") / 100) ** 2

    sofa = pd.read_csv(MIMIC / "sofa.csv", usecols=["stay_id", "sofa_24hours"])
    sofa = sofa.rename(columns={"sofa_24hours": "sofa"})
    sofa["sofa"] = pd.to_numeric(sofa["sofa"], errors="coerce")
    sofa = sofa.groupby("stay_id", as_index=False)["sofa"].median()
    charl = pd.read_csv(MIMIC / "charlson.csv")
    charl = charl.rename(columns={
        "charlson_comorbidity_index": "charlson",
        "chronic_pulmonary_disease": "copd",
        "congestive_heart_failure": "chf",
    })[["subject_id", "hadm_id", "charlson", "copd", "chf"]]
    sep = pd.read_csv(MIMIC / "sepsis3.csv", usecols=["stay_id", "sepsis3"])
    sep["sepsis3"] = sep["sepsis3"].astype(str).str.lower().isin(["t", "true", "1"]).astype(int)
    sep = sep.groupby("stay_id", as_index=False)["sepsis3"].max()
    early = early.merge(sofa, on="stay_id", how="left").merge(charl, on=["subject_id", "hadm_id"], how="left").merge(sep, on="stay_id", how="left")
    early["sepsis3"] = early["sepsis3"].fillna(0).astype(int)

    det = early[["stay_id", "icu_intime"]].copy()
    norepi = pd.read_csv(MIMIC / "norepinephrine_equivalent_dose.csv", usecols=["stay_id", "starttime", "norepinephrine_equivalent_dose"])
    norepi["starttime"] = pd.to_datetime(norepi["starttime"], dayfirst=True, errors="coerce")
    norepi = norepi.merge(det, on="stay_id", how="inner")
    norepi["hour"] = (norepi["starttime"] - norepi["icu_intime"]).dt.total_seconds() / 3600
    vaso_ids = norepi[norepi["hour"].between(0, WINDOW_H) & (pd.to_numeric(norepi["norepinephrine_equivalent_dose"], errors="coerce") > 0)]["stay_id"].unique()
    early["vasopressor"] = early["stay_id"].isin(vaso_ids).astype(int)
    rrt = pd.read_csv(MIMIC / "rrt.csv", usecols=["stay_id", "charttime", "dialysis_present"])
    rrt["charttime"] = pd.to_datetime(rrt["charttime"], dayfirst=True, errors="coerce")
    rrt = rrt.merge(det, on="stay_id", how="inner")
    rrt["hour"] = (rrt["charttime"] - rrt["icu_intime"]).dt.total_seconds() / 3600
    rrt_ids = rrt[rrt["hour"].between(0, WINDOW_H) & (rrt["dialysis_present"] == 1)]["stay_id"].unique()
    early["rrt"] = early["stay_id"].isin(rrt_ids).astype(int)

    counts = build_mimic_counts(early)
    early = early.merge(counts, on="stay_id", how="left")
    early["n_points"] = early["n_points"].fillna(0).astype(int)
    return early


def build_eicu_early() -> pd.DataFrame:
    det = pd.read_csv(EICU / "icustay_detail.csv")
    # Keep the same adult-cohort rule as the formal data-construction script:
    # nonnumeric age values such as "> 89" are treated as missing.
    det["age_num"] = pd.to_numeric(det["age"], errors="coerce")
    det = det[det["age_num"] >= 18].copy()
    events = pd.read_csv(EICU / "ventilation_events.csv")
    mv = events[events["event"].eq("mechvent start")]
    early_ids = mv[pd.to_numeric(mv["hrs"], errors="coerce").between(0, 24)]["patientunitstayid"].drop_duplicates()
    early = det[det["patientunitstayid"].isin(early_ids)].rename(columns={
        "patientunitstayid": "stay_id",
        "hosp_mort": "hospital_expire_flag",
        "apache_iv": "apache_iv",
        "ethnicity": "race",
        "icu_los_hours": "icu_los_hours",
    }).copy()
    early["age"] = early["age_num"]
    g = pd.to_numeric(early["gender"], errors="coerce")
    early["sex"] = np.where(g.eq(1), 1, np.where(g.eq(0), 0, np.nan))
    early["hospital_expire_flag"] = pd.to_numeric(early["hospital_expire_flag"], errors="coerce")
    early["race_grouped"] = race_group_eicu(early["race"])
    h = pd.to_numeric(early["admissionheight"], errors="coerce")
    w = pd.to_numeric(early["admissionweight"], errors="coerce")
    bmi = w / (h / 100) ** 2
    early["bmi"] = bmi.where(h.between(120, 230) & w.between(20, 300) & bmi.between(10, 80))
    early["los_icu"] = pd.to_numeric(early["icu_los_hours"], errors="coerce") / 24

    charl = pd.read_csv(EICU / "charlson.csv").rename(columns={
        "patientunitstayid": "stay_id",
        "final_charlson_score": "charlson",
        "copd1": "copd",
        "chf1": "chf",
    })[["stay_id", "charlson", "copd", "chf"]]
    adm = pd.read_csv(EICU / "admissiondx.csv", usecols=["patientunitstayid", "admitdxname"])
    sep_ids = adm[adm["admitdxname"].str.contains("sepsis", case=False, na=False)]["patientunitstayid"].unique()
    early["sepsis3"] = early["stay_id"].isin(sep_ids).astype(int)
    early = early.merge(charl, on="stay_id", how="left")

    aps = pd.read_csv(EICU / "apacheapsvar.csv", usecols=["patientunitstayid", "dialysis"]).rename(columns={"patientunitstayid": "stay_id"})
    aps = aps.drop_duplicates("stay_id")
    early = early.merge(aps, on="stay_id", how="left")
    early["rrt"] = pd.to_numeric(early["dialysis"], errors="coerce").fillna(0).astype(int)
    early = early.drop(columns=["dialysis"])
    try:
        vaso = pd.read_csv(EICU / "pivoted_treatment_vasopressor.csv", usecols=["patientunitstayid", "chartoffset", "vasopressor"])
        vaso = vaso[pd.to_numeric(vaso["chartoffset"], errors="coerce").between(0, WINDOW_H * 60) & (pd.to_numeric(vaso["vasopressor"], errors="coerce") == 1)]
        vaso_ids = vaso["patientunitstayid"].unique()
    except Exception:
        vaso_ids = []
    early["vasopressor"] = early["stay_id"].isin(vaso_ids).astype(int)

    counts = build_eicu_counts(early)
    early = early.merge(counts, on="stay_id", how="left")
    early["n_points"] = early["n_points"].fillna(0).astype(int)
    return early


VARIABLES = [
    {"label": "Age, years", "var": "age", "type": "continuous", "force": "mean"},
    {"label": "Male, n (%)", "var": "sex", "type": "binary"},
    {"label": "BMI, kg/m2", "var": "bmi", "type": "continuous", "force": "mean"},
    {"label": "White race, n (%)", "var": "race_grouped", "type": "category", "level": "White"},
    {"label": "SOFA score", "var": "sofa", "type": "continuous", "panel": "MIMIC-IV", "force": "median"},
    {"label": "APACHE-IV score", "var": "apache_iv", "type": "continuous", "panel": "eICU", "force": "mean"},
    {"label": "Charlson comorbidity index", "var": "charlson", "type": "continuous", "force": "median"},
    {"label": "Sepsis, n (%)", "var": "sepsis3", "type": "binary"},
    {"label": "COPD, n (%)", "var": "copd", "type": "binary"},
    {"label": "Congestive heart failure, n (%)", "var": "chf", "type": "binary"},
    {"label": "Vasopressor use within 48 h, n (%)", "var": "vasopressor", "type": "binary"},
    {"label": "Renal replacement therapy within 48 h, n (%)", "var": "rrt", "type": "binary"},
    {"label": "ICU length of stay, days", "var": "los_icu", "type": "continuous", "force": "median"},
    {"label": "Hospital mortality, n (%)", "var": "hospital_expire_flag", "type": "binary"},
    {"label": "Valid paired P/F-VR measurements within 48 h, n", "var": "n_points", "type": "continuous", "force": "median"},
]


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def clean_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def summary_cont(s: pd.Series, force: str | None = None) -> str:
    x = clean_num(s)
    if len(x) == 0:
        return "-"
    if force == "mean":
        return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
    if force == "median":
        return f"{x.median():.2f} ({x.quantile(0.25):.2f}-{x.quantile(0.75):.2f})"
    if abs(x.skew()) <= 1:
        return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
    return f"{x.median():.2f} ({x.quantile(0.25):.2f}-{x.quantile(0.75):.2f})"


def p_cont(df: pd.DataFrame, var: str, force: str | None = None) -> str:
    a = clean_num(df.loc[df["included"] == 1, var])
    b = clean_num(df.loc[df["included"] == 0, var])
    if len(a) < 2 or len(b) < 2:
        return ""
    use_rank = force == "median" or (force is None and abs(pd.concat([a, b]).skew()) > 1)
    if use_rank:
        return fmt_p(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return fmt_p(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def fmt_cat(s: pd.Series, denom: int) -> str:
    n = int(pd.to_numeric(s, errors="coerce").fillna(0).eq(1).sum())
    return f"{n} ({n / denom * 100:.1f})" if denom else "-"


def p_cat(df: pd.DataFrame, var: str, level=None) -> str:
    if level is None:
        x = pd.to_numeric(df[var], errors="coerce").fillna(0).astype(int)
    else:
        x = (df[var].fillna("Other/Unknown") == level).astype(int)
    ct = pd.crosstab(x, df["included"])
    if ct.shape != (2, 2):
        return ""
    if (ct.values < 5).any():
        return fmt_p(stats.fisher_exact(ct)[1])
    return fmt_p(stats.chi2_contingency(ct)[1])


def make_rows(df: pd.DataFrame, panel: str) -> list[dict[str, str]]:
    inc = df[df["included"] == 1]
    exc = df[df["included"] == 0]
    rows = []
    for item in VARIABLES:
        if item.get("panel") and item["panel"] != panel:
            continue
        var = item["var"]
        if var not in df.columns:
            continue
        if item["type"] == "continuous":
            rows.append({
                "Variable": item["label"],
                f"Included (n={len(inc):,})": summary_cont(inc[var], item.get("force")),
                f"Excluded (n={len(exc):,})": summary_cont(exc[var], item.get("force")),
                "P value": p_cont(df, var, item.get("force")),
            })
        elif item["type"] == "binary":
            rows.append({
                "Variable": item["label"],
                f"Included (n={len(inc):,})": fmt_cat(inc[var], len(inc)),
                f"Excluded (n={len(exc):,})": fmt_cat(exc[var], len(exc)),
                "P value": p_cat(df, var),
            })
        elif item["type"] == "category":
            level = item["level"]
            rows.append({
                "Variable": item["label"],
                f"Included (n={len(inc):,})": f"{int((inc[var].fillna('Other/Unknown') == level).sum())} ({(inc[var].fillna('Other/Unknown') == level).mean() * 100:.1f})",
                f"Excluded (n={len(exc):,})": f"{int((exc[var].fillna('Other/Unknown') == level).sum())} ({(exc[var].fillna('Other/Unknown') == level).mean() * 100:.1f})",
                "P value": p_cat(df, var, level),
            })
    return rows


def style_run(run, size=6.5, bold=False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def set_cell_border(cell, **kwargs) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, attrs in kwargs.items():
        tag = "w:" + edge
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        for k, v in attrs.items():
            node.set(qn("w:" + k), str(v))


def set_cell_margins(cell, top=60, start=80, bottom=60, end=80) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
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


def style_cell(cell, bold=False, align="center") -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
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
            set_cell_border(cell, top={"val": "nil"}, left={"val": "nil"}, right={"val": "nil"},
                            bottom={"val": "single", "sz": "4", "color": "D9D9D9"})
    for cell in table.rows[0].cells:
        set_cell_border(cell, top={"val": "single", "sz": "12", "color": "000000"},
                        bottom={"val": "single", "sz": "8", "color": "000000"},
                        left={"val": "nil"}, right={"val": "nil"})
    for cell in table.rows[-1].cells:
        set_cell_border(cell, bottom={"val": "single", "sz": "12", "color": "000000"},
                        left={"val": "nil"}, right={"val": "nil"})


def add_panel_table(doc: Document, title: str, rows: list[dict[str, str]]) -> None:
    p = doc.add_paragraph()
    run = p.add_run(title)
    style_run(run, size=6.5, bold=True)
    headers = list(rows[0].keys())
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    widths = [6.2, 4.4, 4.4, 2.2]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        table.rows[0].cells[i].width = Cm(widths[i])
        style_cell(table.rows[0].cells[i], bold=True, align="left" if i == 0 else "center")
    for row in rows:
        cells = table.add_row().cells
        for i, h in enumerate(headers):
            cells[i].text = str(row[h])
            cells[i].width = Cm(widths[i])
            style_cell(cells[i], align="left" if i == 0 else "center")
    set_table_borders(table)


def write_docx(panel_a: list[dict[str, str]], panel_b: list[dict[str, str]]) -> None:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.2)
    section.right_margin = Cm(1.2)

    title = doc.add_paragraph()
    r = title.add_run("Supplementary Table. Baseline characteristics of included and excluded early invasive mechanical ventilation patients.")
    style_run(r, size=6.5, bold=True)
    add_panel_table(doc, "Panel A. MIMIC-IV", panel_a)
    doc.add_paragraph()
    add_panel_table(doc, "Panel B. eICU", panel_b)

    note = doc.add_paragraph()
    note_text = (
        "Data are expressed as the mean ± SD, median (interquartile range), or n (%). "
        "Included patients were those in the final analysis cohorts with at least three valid paired P/F-VR measurements within 48 h after ICU admission and hospital mortality available. "
        "Excluded patients were early invasive mechanical ventilation patients not included in the final cohorts, mainly because of insufficient paired P/F-VR measurements or missing hospital mortality. "
        "P values compare included and excluded patients within each database. "
        "APACHE-IV, Acute Physiology and Chronic Health Evaluation IV; BMI, body mass index; CHF, congestive heart failure; COPD, chronic obstructive pulmonary disease; ICU, intensive care unit; P/F, PaO2/FiO2; RRT, renal replacement therapy; SOFA, Sequential Organ Failure Assessment; VR, ventilatory ratio."
    )
    note.add_run(note_text)
    for run in note.runs:
        style_run(run, size=6.5)
    doc.save(OUTDIR / "TableS_included_vs_excluded_TNR_xiaoliu.docx")


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    mimic = build_mimic_early()
    eicu = build_eicu_early()
    final_m = set(pd.read_csv(FINAL_MIMIC, usecols=["stay_id"])["stay_id"])
    final_e = set(pd.read_csv(FINAL_EICU, usecols=["stay_id"])["stay_id"])
    mimic["included"] = mimic["stay_id"].isin(final_m).astype(int)
    eicu["included"] = eicu["stay_id"].isin(final_e).astype(int)

    # Ensure final cohort counts match the manuscript analysis set.
    assert int(mimic["included"].sum()) == 12694, int(mimic["included"].sum())
    assert int(eicu["included"].sum()) == 1491, int(eicu["included"].sum())

    panel_a = make_rows(mimic, "MIMIC-IV")
    panel_b = make_rows(eicu, "eICU")
    table = pd.concat([
        pd.DataFrame(panel_a).rename(columns={
            [c for c in pd.DataFrame(panel_a).columns if c.startswith("Included")][0]: "Included",
            [c for c in pd.DataFrame(panel_a).columns if c.startswith("Excluded")][0]: "Excluded",
        }).assign(Database="MIMIC-IV"),
        pd.DataFrame(panel_b).rename(columns={
            [c for c in pd.DataFrame(panel_b).columns if c.startswith("Included")][0]: "Included",
            [c for c in pd.DataFrame(panel_b).columns if c.startswith("Excluded")][0]: "Excluded",
        }).assign(Database="eICU"),
    ], ignore_index=True)
    table = table[["Database"] + [c for c in table.columns if c != "Database"]]
    table.to_csv(OUTDIR / "TableS_included_vs_excluded.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "TableS_included_vs_excluded.xlsx") as writer:
        pd.DataFrame(panel_a).to_excel(writer, sheet_name="MIMIC-IV", index=False)
        pd.DataFrame(panel_b).to_excel(writer, sheet_name="eICU", index=False)
        mimic.to_excel(writer, sheet_name="MIMIC_source", index=False)
        eicu.to_excel(writer, sheet_name="eICU_source", index=False)
    write_docx(panel_a, panel_b)

    log_lines = [
        "# Included vs excluded supplementary table generation log",
        "",
        f"MIMIC-IV early invasive mechanical ventilation cohort: n={len(mimic):,}",
        f"- Included final analysis cohort: n={int(mimic['included'].sum()):,}",
        f"- Excluded from final analysis cohort: n={int((mimic['included'] == 0).sum()):,}",
        f"- Hospital deaths in included: n={int(mimic.loc[mimic['included'] == 1, 'hospital_expire_flag'].sum()):,}",
        f"- Hospital deaths in excluded with available outcome: n={int(mimic.loc[mimic['included'] == 0, 'hospital_expire_flag'].fillna(0).sum()):,}",
        "",
        f"eICU early invasive mechanical ventilation cohort: n={len(eicu):,}",
        f"- Included final analysis cohort: n={int(eicu['included'].sum()):,}",
        f"- Excluded from final analysis cohort: n={int((eicu['included'] == 0).sum()):,}",
        f"- Hospital deaths in included: n={int(eicu.loc[eicu['included'] == 1, 'hospital_expire_flag'].sum()):,}",
        f"- Hospital deaths in excluded with available outcome: n={int(eicu.loc[eicu['included'] == 0, 'hospital_expire_flag'].fillna(0).sum()):,}",
        "",
        "Definition:",
        "- Included: final analysis cohort with >=3 valid paired P/F-VR measurements within 48 h and hospital mortality available.",
        "- Excluded: early invasive mechanical ventilation patients not included in the final cohort.",
        "",
        "Outputs:",
        "- TableS_included_vs_excluded_TNR_xiaoliu.docx",
        "- TableS_included_vs_excluded.xlsx",
        "- TableS_included_vs_excluded.csv",
    ]
    (OUTDIR / "TableS_included_vs_excluded_generation_log.md").write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
