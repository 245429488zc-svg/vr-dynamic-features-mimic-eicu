#!/usr/bin/env python3
"""Round-2 master, part 5: assemble the NEW Table 3 (prediction ladder, landmark
framework) and write its TNR docx.

Rows (before = Base + P/F; after = + VR initial & VR delta; per-1-SD):
  1. MIMIC-IV landmark, fully adjusted spec — with the 48-h-determinable sepsis flag
  2. MIMIC-IV landmark, harmonized reduced — sepsis48 slot (48-h Sepsis-3 fit side)
  3. eICU landmark, transported fixed coeffs — admissionDx-recorded sepsis
Delta/IDI/NRI with B=2000 bootstrap CIs (round2_bootstrap.csv) and DeLong P
(delong_and_contrast.csv). The association-framework continuity value is stated
in the table footnote, not as a row.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTCOME = "hospital_expire_flag"
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)




def build_docx(df: pd.DataFrame, title_text: str, note_text: str, out_path: Path) -> None:
    """Self-contained landscape TNR docx writer (replaces the legacy external helper)."""
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
            rn = c.paragraphs[0].add_run("" if pd.isna(row[h]) else str(row[h]))
            rn.font.name = "Times New Roman"; rn.font.size = Pt(7)
            if j > 0:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    note = doc.add_paragraph(); note.paragraph_format.space_before = Pt(6)
    rn = note.add_run(note_text); rn.font.name = "Times New Roman"; rn.font.size = Pt(6.5)
    doc.save(out_path)


def main() -> None:

    # ladders
    lad = pd.read_csv(OUTDIR / "round2_ladders.csv")           # M1 (full-spec & harm-nested & eICU/refit)
    lad2 = pd.read_csv(OUTDIR / "round2_harm_ladders.csv")     # part-2 (harm own-cohort + transported)
    bt = pd.read_csv(OUTDIR / "round2_bootstrap.csv")
    dl = pd.read_csv(OUTDIR / "delong_and_contrast.csv")
    lm_cv = pd.read_csv(OUTDIR / "landmark_incremental_mimic.csv")
    bmap = dict(zip(bt["setting"], zip(bt["delta_auc"], bt["delta_auc_ci"], bt["nri"], bt["nri_ci"],
                                       bt["nri_events"], bt["nri_nonevents"], bt["idi"], bt["idi_ci"])))
    dmap = dict(zip(dl["setting"], dl["delong_p"]))

    def ladder_pick(df: pd.DataFrame, row_label: str, before: str, after: str) -> tuple[str, str]:
        b = df.loc[(df["Row"] == row_label) & (df["Model"] == before), "AUROC"].iloc[0]
        a = df.loc[(df["Row"] == row_label) & (df["Model"] == after), "AUROC"].iloc[0]
        return b, a

    def rng_fmt_boot(ci: str) -> str:
        lo, hi = ci.rsplit("-", 1)
        return f"{float(lo):.3f}-{float(hi):.3f}"

    def harm_row(row_label: str, boot_key: str, delong_key: str,
                 before: str = "Base + P/F", after: str = "Base + P/F + VR") -> dict:
        b, a = ladder_pick(lad, row_label, before, after)
        d, dci, nri, nri_ci, nre, nrn, idi, idi_ci = bmap[boot_key]
        p = dmap.get(delong_key, "")
        return {"Analysis": row_label, "AUROC before": b, "AUROC after": a,
                "Delta AUROC (95% CI)": f"{float(d):.3f} ({rng_fmt_boot(dci)})",
                "DeLong P value": p,
                "IDI (95% CI)": f"{idi} ({idi_ci})", "NRI (95% CI)": f"{nri} ({nri_ci})",
                "NRI events/non-events": f"{nre}/{nrn}"}

    rows = [
        harm_row("MIMIC landmark FULL SPEC (internal)",
                 "S1 MIMIC landmark FULL-spec ladder (VR step)", "S1 MIMIC landmark FULL-spec (VR step)"),
    ]
    # harm row from part-2's own-cohort ladder
    b = lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F"), "AUROC"].iloc[0]
    a = lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "AUROC"].iloc[0]
    d, dci, nri, nri_ci, nre, nrn, idi, idi_ci = bmap["S2 MIMIC landmark HARMONIZED ladder (VR step)"]
    p = dmap["S2 MIMIC landmark HARMONIZED (VR step)"]
    n_h = int(lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_h = int(lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    rows.append({"Analysis": "MIMIC-IV landmark HARMONIZED", "AUROC before": b, "AUROC after": a,
                 "Delta AUROC (95% CI)": f"{float(d):.3f} ({rng_fmt_boot(dci)})",
                 "DeLong P value": p,
                 "IDI (95% CI)": f"{idi} ({idi_ci})", "NRI (95% CI)": f"{nri} ({nri_ci})",
                 "NRI events/non-events": f"{nre}/{nrn}"})

    # eICU transported from part-2's transported (own-cohort coeffs)
    b_e = lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F"), "AUROC"].iloc[0]
    a_e = lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "AUROC"].iloc[0]
    d, dci, nri, nri_ci, nre, nrn, idi, idi_ci = bmap["S3 eICU landmark transported (fixed coeffs)"]
    p = dmap["S3 eICU landmark TRANSPORTED (fixed coeffs, VR step)"]
    n_e = int(lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_e = int(lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    def rng_fmt(ci: str) -> str:
        lo, hi = ci.rsplit("-", 1)
        return f"{float(lo):.3f}-{float(hi):.3f}"
    rows.append({"Analysis": "eICU landmark (transported, fixed MIMIC coefficients)",
                 "AUROC before": b_e, "AUROC after": a_e,
                 "Delta AUROC (95% CI)": f"{float(d):.3f} ({rng_fmt(dci)})",
                 "DeLong P value": p,
                 "IDI (95% CI)": f"{idi} ({idi_ci})", "NRI (95% CI)": f"{nri} ({nri_ci})",
                 "NRI events/non-events": f"{nre}/{nrn}"})

    t3 = pd.DataFrame(rows)
    n_f = int(lad.loc[(lad["Row"] == "MIMIC landmark FULL SPEC (internal)") &
                      (lad["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_f = int(lad.loc[(lad["Row"] == "MIMIC landmark FULL SPEC (internal)") &
                      (lad["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    t3.insert(1, "N", [n_f, n_h, n_e])
    t3.insert(2, "Deaths", [d_f, d_h, d_e])
    t3.to_csv(OUTDIR / "Table3_round2.csv", index=False)
    stamp("Table3 rows assembled:")
    print(t3.to_string(index=False))

    note = (
        "Table 3 reports the prediction analyses within the 48-hour landmark framework: patients alive and "
        "hospitalized at 48 hours after ICU admission, with hospital death after the landmark time as the outcome, "
        "following the Round-2 rebuild (observation window truncated at the earliest of death, terminal extubation, "
        "ICU discharge, or hospital discharge; vital status ascertained from the official admissions tables; "
        "569 MIMIC-IV stays no longer met the three-pair requirement after window-boundary truncation and were "
        "excluded, and hospital mortality was re-ascertained from the official admissions tables for every "
        "eligible stay — the legacy outcome files lacked it for 110 MIMIC-IV and 167 eICU stays). "
        "Before = Base + P/F model; After = additional inclusion of VR initial and VR delta (per 1-SD increase, "
        "modeled conditional on VR initial). Row 1 uses the fully adjusted specification (age, sex, Charlson index, "
        "sepsis (Sepsis-3, 48-hour-determinable: suspected-infection onset and qualifying SOFA assessment both "
        "within the first 48 hours), COPD, CHF, SOFA, PEEP, driving pressure, Vt/PBW, heart rate, MAP, temperature, "
        "lactate, BUN, creatinine, WBC, bicarbonate, initial and minimum P/F); demographics and comorbidity "
        "variables (age, sex, Charlson index, COPD, CHF) are fixed admission-level (history) characteristics, "
        "and every time-varying covariate, including the 48-hour-determinable sepsis flag, is determined within "
        "the first 48 hours. Rows 2-3 use the harmonized reduced "
        "specification (Row 1 minus SOFA, driving pressure, and lactate), with sepsis determinable by the 48-hour "
        "landmark time in both databases: the Row-1 48-hour Sepsis-3 definition in MIMIC-IV, and sepsis among the "
        "recorded admission diagnoses (admissionDx) in eICU. "
        "Row 2 is fitted in MIMIC-IV and transported "
        "unchanged (fixed coefficients and the MIMIC-fitted scaler) to the eICU landmark cohort in Row 3. "
        "Delta/IDI/NRI are paired-model statistics with bootstrap 95% confidence intervals (2,000 draws; derivation "
        "settings refit per resample; eICU transported evaluated with fixed coefficients per resample). "
        "For continuity, the association-framework ladder (corrected specification: the fully adjusted 18 "
        "covariates including COPD and CHF) on the rebuilt complete-case cohort (n=10,428) gives apparent AUROC "
        "0.842 to 0.847 (VR-step Delta 0.0050, 95% CI 0.0031-0.0073; 5-fold cross-validated AUROC 0.840 to "
        "0.844, optimism <=0.003, out-of-sample calibration -0.018/0.983). Within the landmark framework, the "
        "fully adjusted models showed 5-fold cross-validated AUROCs of "
        f"{lm_cv.loc[lm_cv.Model.eq('Base'), 'AUROC (CV)'].iloc[0]:.3f} (Base), "
        f"{lm_cv.loc[lm_cv.Model.eq('Base + P/F'), 'AUROC (CV)'].iloc[0]:.3f} (Base + P/F), and "
        f"{lm_cv.loc[lm_cv.Model.eq('Base + P/F + VR'), 'AUROC (CV)'].iloc[0]:.3f} "
        "(Base + P/F + VR); calibration and optimism estimates are reported in the analysis log."
    )
    build_docx(t3, "Table 3. Incremental predictive value of early VR features beyond P/F ratio (48-hour landmark framework).",
                  note, OUTDIR / "Table3_round2_TNR_xiaoliu.docx")
    (OUTDIR / "master_part5_log.md").write_text("# Table 3 assembly\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Table 3 docx 完成")


if __name__ == "__main__":
    main()
