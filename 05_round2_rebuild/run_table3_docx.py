#!/usr/bin/env python3
"""Round-2 master, part 5: assemble the NEW Table 3 (prediction ladder, landmark
framework) and write its TNR docx.

Rows (before = Base + P/F; after = + VR initial & VR delta; per-1-SD):
  1. MIMIC-IV landmark, fully adjusted spec   (n=9,882)
  2. MIMIC-IV landmark, harmonized reduced    (n=10,663)
  3. eICU landmark, transported fixed coeffs  (n=1,270)
Delta/IDI/NRI with B=2000 bootstrap CIs (round2_bootstrap.csv) and DeLong P
(delong_and_contrast.csv). The association-framework continuity value is stated
in the table footnote, not as a row.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUTDIR = ROOT / "43_第二轮重分析"
OUTCOME = "hospital_expire_flag"
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def main() -> None:
    inc_spec = importlib.util.spec_from_file_location(
        "m4", ROOT / "13_Table4正式版" / "make_table4_sensitivity.py")
    m4 = importlib.util.module_from_spec(inc_spec)
    inc_spec.loader.exec_module(m4)

    inc_spec2 = importlib.util.spec_from_file_location(
        "harmod", OUTDIR / ".." / "43_第二轮重分析" / "run_master_part2.py")
    # ladders
    lad = pd.read_csv(OUTDIR / "round2_ladders.csv")           # M1 (full-spec & harm-nested & eICU/refit)
    lad2 = pd.read_csv(OUTDIR / "round2_harm_ladders.csv")     # part-2 (harm own-cohort + transported)
    bt = pd.read_csv(OUTDIR / "round2_bootstrap.csv")
    dl = pd.read_csv(OUTDIR / "delong_and_contrast.csv")
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
    p = dmap["S3b eICU landmark TRANSPORTED (fixed coeffs, VR step)"]
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
    t3.insert(1, "N", [9882, n_h, n_e])
    t3.insert(2, "Deaths", [1696, d_h, d_e])
    t3.to_csv(OUTDIR / "Table3_round2.csv", index=False)
    stamp("Table3 rows assembled:")
    print(t3.to_string(index=False))

    note = (
        "Table 3 reports the prediction analyses within the 48-hour landmark framework: patients alive and "
        "hospitalized at 48 hours after ICU admission, with hospital death after the landmark time as the outcome, "
        "following the Round-2 rebuild (observation window truncated at the earliest of death, terminal extubation, "
        "ICU discharge, or hospital discharge; vital status ascertained from the official admissions tables; "
        "569 MIMIC-IV stays that no longer met the three-pair requirement after truncation were excluded, and 569 "
        "stays with legacy missing mortality in the rebuilt outcome file were re-included via the official join). "
        "Before = Base + P/F model; After = additional inclusion of VR initial and VR delta (per 1-SD increase, "
        "modeled conditional on VR initial). Row 1 uses the fully adjusted specification (age, sex, Charlson index, "
        "sepsis (Sepsis-3), COPD, CHF, SOFA, PEEP, driving pressure, Vt/PBW, heart rate, MAP, temperature, lactate, "
        "BUN, creatinine, WBC, bicarbonate, initial and minimum P/F). Rows 2-3 use the harmonized reduced "
        "specification (Row 1 minus SOFA, driving pressure, and lactate, with sepsis defined by recorded admission "
        "diagnosis [ICD-coded in MIMIC-IV, admissionDx text in eICU]); Row 2 is fitted in MIMIC-IV and transported "
        "unchanged (fixed coefficients and the MIMIC-fitted scaler) to the eICU landmark cohort in Row 3. "
        "Delta/IDI/NRI are paired-model statistics with bootstrap 95% confidence intervals (2,000 draws; derivation "
        "settings refit per resample; eICU transported evaluated with fixed coefficients per resample). "
        "For continuity, the association-framework ladder on the rebuilt complete-case cohort (n=10,349) gives "
        "AUROC 0.842 to 0.848 (Delta 0.0052, 95% CI 0.0031-0.0077; 5-fold CV 0.840 to 0.846, optimism 0.002, "
        "out-of-sample calibration -0.016/0.986). Within the landmark framework, the fully adjusted models showed "
        "5-fold cross-validated AUROCs of 0.827 (Base), 0.827 (Base + P/F), and 0.831 (Base + P/F + VR), optimism "
        "<=0.003, with out-of-sample calibration -0.034/0.973."
    )
    m4.OUTDIR = OUTDIR
    m4.build_docx(t3, "Table 3. Incremental predictive value of early VR features beyond P/F ratio (48-hour landmark framework).",
                  note, OUTDIR / "Table3_round2_TNR_xiaoliu.docx")
    (OUTDIR / "master_part5_log.md").write_text("# Table 3 assembly\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Table 3 docx 完成")


if __name__ == "__main__":
    main()