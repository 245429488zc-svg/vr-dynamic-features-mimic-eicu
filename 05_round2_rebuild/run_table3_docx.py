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
    from docx.oxml.ns import qn
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
    # 列宽以 twips 计, 合计 24.74 cm = Letter 横版可用宽; 显式写入 tblGrid 渲染器才认
    widths_cm = [5.9, 1.5, 1.6, 2.4, 2.4, 3.6, 2.2, 1.6, 3.5]
    w_twips = [int(round(w * 567)) for w in widths_cm[:len(headers)]]
    grid = table._element.find(qn("w:tblGrid"))
    for g, w in zip(grid.findall(qn("w:gridCol")), w_twips):
        g.set(qn("w:w"), str(w))
    tblPr = table._element.find(qn("w:tblPr"))
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        from docx.oxml import OxmlElement
        tblW = OxmlElement("w:tblW"); tblPr.append(tblW)
    tblW.set(qn("w:w"), str(sum(w_twips))); tblW.set(qn("w:type"), "dxa")
    tblLayout = tblPr.find(qn("w:tblLayout"))
    if tblLayout is None:
        from docx.oxml import OxmlElement
        tblLayout = OxmlElement("w:tblLayout"); tblPr.append(tblLayout)
    tblLayout.set(qn("w:type"), "fixed")
    table.autofit = False
    for j, w in enumerate(w_twips):
        for row_cells in table.rows:
            if j < len(row_cells.cells):
                row_cells.cells[j].width = Cm(w_twips[j] / 567.0)
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

    def fmt_p(p: str) -> str:
        try:
            v = float(p)
        except (TypeError, ValueError):
            return str(p)
        return "<0.001" if v < 0.001 else f"{v:.3f}"

    def row_dict(label: str, b: str, a: str, boot_key: str, delong_key: str) -> dict:
        d, dci, nri, nri_ci, nre, nrn, idi, idi_ci = bmap[boot_key]
        lo, hi = nri_ci.rsplit("-", 1)
        return {"Cohort / setting": label, "AUROC before": b, "AUROC after": a,
                "Delta AUROC": f"{float(d):.3f} ({rng_fmt_boot(dci)})",
                "DeLong P value": fmt_p(dmap.get(delong_key, "")),
                "IDI": f"{float(idi):.3f}", "NRI": f"{float(nri):.3f} ({float(lo):.3f}-{float(hi):.3f})",
                "_evne": f"{nre}/{nrn}"}

    rows = [
        row_dict("MIMIC-IV internal model",
                 *ladder_pick(lad, "MIMIC landmark FULL SPEC (internal)", "Base + P/F", "Base + P/F + VR"),
                 "S1 MIMIC landmark FULL-spec ladder (VR step)", "S1 MIMIC landmark FULL-spec (VR step)"),
    ]
    # harmonized row from part-2's own-cohort ladder
    b = lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F"), "AUROC"].iloc[0]
    a = lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "AUROC"].iloc[0]
    rows.append(row_dict("MIMIC-IV harmonized reduced model", b, a,
                         "S2 MIMIC landmark HARMONIZED ladder (VR step)", "S2 MIMIC landmark HARMONIZED (VR step)"))

    # eICU transported from part-2's transported (own-cohort coeffs)
    b_e = lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F"), "AUROC"].iloc[0]
    a_e = lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "AUROC"].iloc[0]
    rows.append(row_dict("eICU external validation (transported)", b_e, a_e,
                         "S3 eICU landmark transported (fixed coeffs)",
                         "S3 eICU landmark TRANSPORTED (fixed coeffs, VR step)"))

    t3 = pd.DataFrame(rows)
    evne = t3.pop("_evne")
    n_f = int(lad.loc[(lad["Row"] == "MIMIC landmark FULL SPEC (internal)") &
                      (lad["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_f = int(lad.loc[(lad["Row"] == "MIMIC landmark FULL SPEC (internal)") &
                      (lad["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    n_h = int(lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_h = int(lad2.loc[(lad2["Row"] == "MIMIC harm (own cohort)") & (lad2["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    n_e = int(lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "N"].iloc[0])
    d_e = int(lad2.loc[(lad2["Row"] == "eICU landmark (transported)") & (lad2["Model"] == "Base + P/F + VR"), "Deaths"].iloc[0])
    t3.insert(1, "N", [n_f, n_h, n_e])
    t3.insert(2, "Deaths", [d_f, d_h, d_e])
    t3.to_csv(OUTDIR / "Table3_round2.csv", index=False)
    stamp("Table3 rows assembled:")
    print(t3.assign(evne=evne.values).to_string(index=False))

    ev1, ev2, ev3 = evne.values
    note = (
        "The comparison evaluates the incremental value of adding VR initial and VR delta to the Base + P/F model "
        "among patients alive and hospitalized at 48 hours after ICU admission, with in-hospital death after 48 "
        "hours as the outcome (the 48-hour landmark framework; cohort flow and the explicit window truncation are "
        "shown in Figure 1). In rows 1–2 the Base model comprised the fully adjusted covariates stated in the "
        "Methods, with sepsis as the 48-hour-determinable Sepsis-3 flag (suspected-infection onset and qualifying "
        "SOFA assessment within the first 48 hours); the harmonized reduced row omitted SOFA, driving pressure, "
        "and lactate, used the same MIMIC-IV sepsis definition in the fitting cohort and recorded admission "
        "diagnoses in eICU, and was transported with fixed coefficients and the MIMIC-IV-fitted scaler. P/F "
        "variables were the initial and minimum values; VR variables were VR initial and VR delta (per 1-SD "
        "increase, delta modeled conditional on initial; see the Methods). ΔAUROC was calculated as the AUROC "
        "after adding VR minus the AUROC before; 95% confidence intervals were estimated by bootstrap resampling "
        "(2,000 draws; derivation settings refit per resample, the transported setting evaluated with fixed "
        "coefficients per resample). IDI, integrated discrimination improvement; NRI denotes the category-free net "
        "reclassification improvement, with event and non-event components of "
        f"{ev1.replace(chr(45), chr(8722))}, {ev2.replace(chr(45), chr(8722))}, and {ev3.replace(chr(45), chr(8722))}. "
        "For continuity with the association framework, the same VR step on the rebuilt complete-case cohort "
        "(n = 10,428) increased apparent AUROC from 0.842 to 0.847 (Δ 0.005, 95% CI 0.003–0.007; cross-validated "
        "0.840 to 0.844); removing sepsis from both sides of the harmonized specification yielded directionally "
        "consistent results (Δ 0.006 and 0.008; Supplementary Table 6)."
    )
    build_docx(t3, "Table 3. Incremental value of early ventilatory ratio features beyond P/F ratio in the 48-hour landmark framework.",
                  note, OUTDIR / "Table3_round2_TNR_xiaoliu.docx")
    (OUTDIR / "master_part5_log.md").write_text("# Table 3 assembly\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Table 3 docx 完成")


if __name__ == "__main__":
    main()
