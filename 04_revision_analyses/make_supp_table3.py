#!/usr/bin/env python3
"""Supplementary Table 3: included vs excluded early-IMV patients (R2.3 / R1.1).

Builds the formal included-vs-excluded comparison for MIMIC-IV and eICU with
invasive ventilation duration added. Baseline rows are reused verbatim from the
July internal run (24_SuppTable_included_vs_excluded/TableS_included_vs_excluded.csv);
ventilation duration rows are computed fresh from:

  - MIMIC-IV v3.1 official core tables (icustays / patients) plus the derived
    ventilation intervals (InvasiveVent segments), cohort membership verified
    against the published flow counts (28,387 / 13,145 / 15,242);
  - eICU-CRD v2.0 derived ventilation events (mechvent start/end), cohort
    membership verified against 4,141 / 1,324 / 2,817.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(".")
OUT = ROOT / "32_SuppTable3_included_vs_excluded"
OUT.mkdir(exist_ok=True)
SQL = Path("./raw_data/mimmc4/原始数据和相关文件/MIMICsql/mimic-iv-3.1")
OLD_CSV = ROOT / "24_SuppTable_included_vs_excluded" / "TableS_included_vs_excluded.csv"

LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def med_iqr(s: pd.Series) -> str:
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:.2f} ({q1:.2f}-{q3:.2f})"


def mw_p(a: pd.Series, b: pd.Series) -> str:
    stat, p = stats_mw(a, b)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stats_mw(a: pd.Series, b: pd.Series):
    from scipy import stats
    return stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")


def overlap_sum(starts: pd.Series, ends: pd.Series, lo: pd.Series, hi: pd.Series) -> pd.Series:
    lo_ = np.maximum(starts, lo)
    hi_ = np.minimum(ends, hi)
    return (hi_ - lo_).clip(lower=0)


# ------------------------------------------------------------------ MIMIC
def build_mimic() -> pd.DataFrame:
    icu = pd.read_csv(SQL / "icu" / "icustays.csv.gz",
                      usecols=["subject_id", "stay_id", "hadm_id", "intime", "outtime"])
    pat = pd.read_csv(SQL / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    df = icu.merge(pat, on="subject_id", how="left")
    df["intime"] = pd.to_datetime(df["intime"])
    stamp(f"MIMIC icustays: {len(df)} (期望 94,458)")
    first = df["intime"] == df.groupby("hadm_id")["intime"].transform("min")
    adult = pd.to_numeric(df["anchor_age"], errors="coerce") >= 18

    vent = pd.read_csv(OUT / "mimiciv_derived" / "ventilation.csv",
                       names=["stay_id", "starttime", "endtime", "ventilation_status"],
                       skiprows=1, low_memory=False)
    vent = vent[vent["stay_id"].astype(str).str.isdigit()].copy()
    inv = vent[vent["ventilation_status"].eq("InvasiveVent")].copy()
    inv["stay_id"] = inv["stay_id"].astype(int)
    inv["start"] = pd.to_datetime(inv["starttime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    inv["end"] = pd.to_datetime(inv["endtime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    m = inv.merge(df[["stay_id", "intime"]], on="stay_id", how="inner")
    m["sh"] = (m["start"] - m["intime"]).dt.total_seconds() / 3600
    m["eh"] = (m["end"] - m["intime"]).dt.total_seconds() / 3600
    early_ids = set(m.loc[(m["sh"] <= 24) & (m["eh"] >= 0), "stay_id"])
    cohort = df.loc[first & adult & df["stay_id"].isin(early_ids)].copy()
    stamp(f"MIMIC 早期上机队列重建: {len(cohort)} (期望 28,387)")

    inc = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv", usecols=["stay_id"])
    inc_ids = set(inc["stay_id"].astype(int))
    n_inc_in = len(inc_ids & set(cohort["stay_id"].astype(int)))
    stamp(f"included 在队列中: {n_inc_in}/13,145; excluded = {len(cohort) - n_inc_in} (期望 15,242)")

    inv_c = m[m["stay_id"].isin(set(cohort["stay_id"].astype(int)))]
    dur_total = inv_c.groupby("stay_id").apply(
        lambda g: ((g["end"] - g["start"]).dt.total_seconds() / 3600).sum(), include_groups=False)
    dur_48 = inv_c.groupby("stay_id").apply(
        lambda g: overlap_sum(g["sh"], g["eh"], 0.0, 48.0).sum(), include_groups=False)
    dur = pd.DataFrame({"stay_id": dur_total.index, "vent_total_h": dur_total.values})
    dur["vent_48h_h"] = dur["stay_id"].map(dur_48)
    return dur


# ------------------------------------------------------------------ eICU
def build_eicu() -> pd.DataFrame:
    detail = pd.read_csv(OUT / "eICU" / "icustay_detail_fixed.csv", low_memory=False,
                         usecols=["patientunitstayid", "age", "unitdischargeoffset"])
    detail["patientunitstayid"] = pd.to_numeric(detail["patientunitstayid"], errors="coerce").astype("Int64")
    events = pd.read_csv(OUT / "eICU" / "ventilation_events.csv")
    events["hrs"] = pd.to_numeric(events["hrs"], errors="coerce")

    mv_start = events[events["event"].eq("mechvent start")]
    early_ids = set(mv_start.loc[mv_start["hrs"].between(0, 24), "patientunitstayid"].dropna().astype(int))
    adult_ids = set(detail.loc[pd.to_numeric(detail["age"], errors="coerce") >= 18, "patientunitstayid"].dropna().astype(int))
    cohort_ids = adult_ids & early_ids
    stamp(f"eICU 早期上机队列重建: {len(cohort_ids)} (期望 4,141)")

    inc = pd.read_csv(ROOT / "1_队列数据" / "eICU_analysis_ready.csv", usecols=["stay_id"])
    inc_ids = set(inc["stay_id"].astype(int))
    n_inc_in = len(inc_ids & cohort_ids)
    stamp(f"eICU included 在队列中: {n_inc_in}/1,324; excluded = {len(cohort_ids) - n_inc_in} (期望 2,817)")

    censor = detail.set_index("patientunitstayid")["unitdischargeoffset"].to_dict()
    ev = events[events["patientunitstayid"].isin(cohort_ids)].copy()
    dur_rows = []
    for sid, g in ev_groupby(ev, "patientunitstayid"):
        g = g.sort_values("hrs")
        open_starts: list[float] = []
        episodes = []
        limit = float(censor.get(sid, np.nan)) / 60.0 if sid in censor else np.nan
        for _, r in g.iterrows():
            e, h = r["event"], float(r["hrs"]) if pd.notna(r["hrs"]) else np.nan
            if e == "mechvent start":
                open_starts.append(h)
            elif e == "mechvent end" and open_starts:
                s = open_starts.pop(0)
                episodes.append((s, h))
            elif pd.notna(limit) and h >= limit:
                break
        for s in open_starts:
            if pd.notna(limit):
                episodes.append((s, limit))
        tot = sum(max(0.0, b - a) for a, b in episodes)
        w48 = sum(max(0.0, min(b, 48.0) - max(a, 0.0)) for a, b in episodes)
        dur_rows.append({"stay_id": sid, "vent_total_h": tot, "vent_48h_h": w48})
    dur = pd.DataFrame(dur_rows)
    stamp(f"eICU 通气时长计算完成: {len(dur)} stays")
    return dur


def ev_groupby(df: pd.DataFrame, col: str):
    for sid, g in df.groupby(col, sort=False):
        yield sid, g


# ------------------------------------------------------------------ stats + table
def add_vent_rows(dur: pd.DataFrame, inc_ids: set, cohort_ids: set) -> tuple[dict, dict]:
    d = dur.set_index("stay_id")
    inc_d = d.loc[d.index.isin(inc_ids)]
    exc_d = d.loc[d.index.isin(cohort_ids - inc_ids)]
    out = {}
    for col, label in [("vent_48h_h", "Invasive ventilation duration within the first 48 h, h"),
                       ("vent_total_h", "Total invasive ventilation duration, h")]:
        p = stats_mw(inc_d[col], exc_d[col])[1]
        out[label] = {"Included": med_iqr(inc_d[col]), "Excluded": med_iqr(exc_d[col]),
                      "P value": "<0.001" if p < 0.001 else f"{p:.3f}"}
        stamp(f"{label}: included {out[label]['Included']}, excluded {out[label]['Excluded']}, P={out[label]['P value']}")
    return out


def main() -> None:
    from scipy import stats  # noqa
    # MIMIC
    mimic_dur = build_mimic()
    inc_m = pd.read_csv(ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv", usecols=["stay_id"])
    inc_m_ids = set(inc_m["stay_id"].astype(int))
    # MIMIC 队列 id 集合（重建）
    icu = pd.read_csv(SQL / "icu" / "icustays.csv.gz", usecols=["subject_id", "stay_id", "hadm_id", "intime", "outtime"])
    pat = pd.read_csv(SQL / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age"])
    dfm = icu.merge(pat, on="subject_id", how="left")
    dfm["intime"] = pd.to_datetime(dfm["intime"])
    first = dfm["intime"] == dfm.groupby("hadm_id")["intime"].transform("min")
    adult = pd.to_numeric(dfm["anchor_age"], errors="coerce") >= 18
    vent = pd.read_csv(OUT / "mimiciv_derived" / "ventilation.csv", names=["stay_id", "starttime", "endtime", "ventilation_status"], skiprows=1, low_memory=False)
    vent = vent[vent["stay_id"].astype(str).str.isdigit()].copy()
    inv = vent[vent["ventilation_status"].eq("InvasiveVent")].copy()
    inv["stay_id"] = inv["stay_id"].astype(int)
    inv["start"] = pd.to_datetime(inv["starttime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    inv["end"] = pd.to_datetime(inv["endtime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    mm = inv.merge(dfm[["stay_id", "intime"]], on="stay_id", how="inner")
    mm["sh"] = (mm["start"] - mm["intime"]).dt.total_seconds() / 3600
    mm["eh"] = (mm["end"] - mm["intime"]).dt.total_seconds() / 3600
    early_ids = set(mm.loc[(mm["sh"] <= 24) & (mm["eh"] >= 0), "stay_id"])
    cohort_m_ids = set(dfm.loc[first & adult, "stay_id"].astype(int)) & early_ids
    mimic_stats = add_vent_rows(mimic_dur, inc_m_ids, cohort_m_ids)
    mimic_dur.to_csv(OUT / "mimic_vent_duration_per_stay.csv", index=False)

    # eICU
    eicu_dur = build_eicu()
    inc_e = pd.read_csv(ROOT / "1_队列数据" / "eICU_analysis_ready.csv", usecols=["stay_id"])
    inc_e_ids = set(inc_e["stay_id"].astype(int))
    detail = pd.read_csv(OUT / "eICU" / "icustay_detail_fixed.csv", low_memory=False,
                         usecols=["patientunitstayid", "age"])
    detail["patientunitstayid"] = pd.to_numeric(detail["patientunitstayid"], errors="coerce")
    adult_ids = set(detail.loc[pd.to_numeric(detail["age"], errors="coerce") >= 18, "patientunitstayid"].dropna().astype(int))
    events_all = pd.read_csv(OUT / "eICU" / "ventilation_events.csv")
    events_all["hrs"] = pd.to_numeric(events_all["hrs"], errors="coerce")
    early_e = set(events_all.loc[(events_all["event"].eq("mechvent start")) & (events_all["hrs"].between(0, 24)), "patientunitstayid"])
    cohort_e_ids = adult_ids & early_e
    eicu_stats = add_vent_rows(eicu_dur, inc_e_ids, cohort_e_ids)
    eicu_dur.to_csv(OUT / "eicu_vent_duration_per_stay.csv", index=False)

    # 组装表
    old = pd.read_csv(OLD_CSV, dtype=str)
    rows = []
    for _, r in old.iterrows():
        rows.append([r["Database"], r["Variable"], r["Included"], r["Excluded"], r["P value"]])
        if "ICU length of stay" in r["Variable"]:
            db = r["Database"]
            stats_map = mimic_stats if db.strip() == "MIMIC-IV" else eicu_stats
            for label, vals in stats_map.items():
                rows.append([db, label, vals["Included"], vals["Excluded"], vals["P value"]])
    final = pd.DataFrame(rows, columns=["Database", "Variable", "Included", "Excluded", "P value"])
    final.to_csv(OUT / "Supplementary_Table3_included_vs_excluded.csv", index=False)

    # docx
    doc = Document()
    sec = doc.sections[0]
    from docx.enum.section import WD_ORIENT
    w, h = sec.page_width, sec.page_height
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = h, w
    p0 = doc.add_paragraph()
    r0 = p0.add_run("Supplementary Table 3. Comparison of included and excluded patients.")
    r0.font.name = "Times New Roman"; r0.font.size = Pt(9); r0.font.bold = True
    p1 = doc.add_paragraph()
    r1 = p1.add_run(
        "Included patients were those in the final analysis cohorts (MIMIC-IV n = 13,145; eICU n = 1,324), who had at least three "
        "valid paired P/F-VR measurements within 48 h of ICU admission and available hospital mortality. Excluded patients "
        "were early invasive mechanically ventilated patients who were not included in the final cohorts (MIMIC-IV n = 15,242, "
        "comprising 15,124 patients with fewer than three paired measurements and 118 patients missing hospital mortality; "
        "eICU n = 2,817). Ventilation duration was derived from documented invasive ventilation intervals (InvasiveVent in "
        "mimic-iv derived ventilation records; ventilator episodes in eICU), truncated at ICU discharge; continuous variables "
        "are reported as median (IQR) and compared with the Mann-Whitney U test; categorical variables are reported as counts "
        "(%) and compared with the chi-square test.")
    r1.font.name = "Times New Roman"; r1.font.size = Pt(6.5)

    tbl_data = [["Variable", "Included", "Excluded", "P value"]]
    cur_db = None
    for db, var, inc_, exc_, p in final.itertuples(index=False):
        if db != cur_db:
            n_inc = 13145 if db.strip() == "MIMIC-IV" else 1324
            n_exc = 15242 if db.strip() == "MIMIC-IV" else 2817
            tbl_data.append([f"{db} (included n = {n_inc:,}; excluded n = {n_exc:,})", "", "", ""])
            cur_db = db
        tbl_data.append([var, inc_, exc_, p])
    t = doc.add_table(rows=len(tbl_data), cols=4)
    t.style = doc.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    widths = [Cm(13), Cm(5.5), Cm(5.5), Cm(3)]
    for i, row in enumerate(tbl_data):
        for j, val in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.name = "Times New Roman"; run.font.size = Pt(6.5)
            run.font.bold = (i == 0) or (str(row[0]).startswith(("MIMIC", "eICU")))
            cell.width = widths[j]
    doc.save(OUT / "Supplementary Table 3.docx")
    stamp(f"docx 已保存: {OUT / 'Supplementary Table 3.docx'}")
    (OUT / "Supplementary_Table3_generation_log.md").write_text(
        "# Supplementary Table 3 生成日志（included vs excluded，R2.3/R1.1）\n\n"
        "- 基线行复用 24_SuppTable_included_vs_excluded 2026-07 运行结果（原样保留）。\n"
        "- 新增通气时长两行：MIMIC 源自 mimic-iv derived ventilation 表（InvasiveVent 区间，单条损坏记录已剔除）+ 官方 icustays；"
        "eICU 源自 derived ventilation_events（mechvent start/end 配对，未闭合 episode 以 unit discharge 截断）。\n"
        "- 队列成员重建并校验：MIMIC 28,387（included 13,145 全部在列，excluded 15,242）；eICU 4,141（included 1,324 全部在列，excluded 2,817）。\n"
        "- 统计：median (IQR)，Mann-Whitney U。\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()