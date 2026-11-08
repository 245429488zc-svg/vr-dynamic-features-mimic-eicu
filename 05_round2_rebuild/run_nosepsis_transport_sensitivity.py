#!/usr/bin/env python3
"""Sensitivity: harmonized/transported 48-h landmark ladder WITHOUT the sepsis covariate.

Rationale (Reviewer 2, Comment 2): the harmonized sepsis slot (sepsis48) is the 48-h
Sepsis-3 definition in MIMIC-IV but the admissionDx text definition in eICU — related,
not identical constructions (prevalence ~71.5% in the MIMIC fit cohort vs ~16.1% in the
transported eICU landmark cohort). This script removes sepsis from BOTH sides of the
harmonized/transported specification and re-runs the ladder end-to-end, so that the
"report how results change" requirement is met with a single-variable-difference test.

Settings:
  N2 MIMIC harm (own cohort), NO-sepsis harmonized spec: Base(+PF/+VR) apparent fits,
     ΔAUROC/IDI/NRI with stratified bootstrap refit (B = 2,000), DeLong P (published
     algorithm imported from 03_main_analyses).
  N3 eICU landmark, transported (fixed MIMIC no-sepsis coefficients + MIMIC-fitted
     scaler), fixed-scalar bootstrap (no refit), DeLong P.

Outputs (in OUTDIR): nosepsis_transport_sensitivity.csv + run_nosepsis_transport_log.md.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTCOME = "hospital_expire_flag"
RNG = 20260721
B = 2000

# Harmonized spec WITHOUT sepsis (14 covariates): both databases identically defined
# and admitted-time/0-48h-determinable.
BASE_H_NOSEP = ["age", "sex", "charlson", "copd", "chf", "peep",
                "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def nri_idi(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> dict:
    y = y.astype(int)
    ev = y == 1
    de = p_new[ev] - p_old[ev]
    dn = p_new[~ev] - p_old[~ev]
    nri_event = float((de > 0).mean() - (de < 0).mean())
    nri_nonevent = float((dn < 0).mean() - (dn > 0).mean())
    idi = float(de.mean() - dn.mean())
    return {"nri": nri_event + nri_nonevent, "nri_events": nri_event,
            "nri_nonevents": nri_nonevent, "idi": idi}


def _fit(df: pd.DataFrame, vars_):
    scaler = StandardScaler().fit(df[vars_])
    m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    m.fit(scaler.transform(df[vars_]), df[OUTCOME].astype(int))
    return scaler, m


def boot_internal_no_sepsis(df: pd.DataFrame, label: str) -> dict:
    y = df[OUTCOME].astype(int).values
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(RNG)
    da, nris, idi_ = [], [], []
    for _ in range(B):
        sel = np.concatenate([idx1[rng.randint(0, len(idx1), len(idx1))],
                              idx0[rng.randint(0, len(idx0), len(idx0))]])
        sub = df.iloc[sel]
        ysub = sub[OUTCOME].astype(int).values
        so, mo = _fit(sub, BASE_H_NOSEP + PF)
        sn, mn = _fit(sub, BASE_H_NOSEP + PF + VR)
        po = mo.predict_proba(so.transform(sub[BASE_H_NOSEP + PF]))[:, 1]
        pn = mn.predict_proba(sn.transform(sub[BASE_H_NOSEP + PF + VR]))[:, 1]
        da.append(roc_auc_score(ysub, pn) - roc_auc_score(ysub, po))
        nris.append(nri_idi(ysub, po, pn))
    da = np.array(da)
    nri = np.array([r["nri"] for r in nris]); ev = np.array([r["nri_events"] for r in nris])
    ne = np.array([r["nri_nonevents"] for r in nris]); idiv = np.array([r["idi"] for r in nris])
    out = {"setting": label, "B": B,
           "delta_auc": f"{np.mean(da):.4f}",
           "delta_auc_ci": f"{np.percentile(da, 2.5):.4f}-{np.percentile(da, 97.5):.4f}",
           "nri": f"{np.mean(nri):.3f}", "nri_ci": f"{np.percentile(nri, 2.5):.3f}-{np.percentile(nri, 97.5):.3f}",
           "nri_events": f"{np.mean(ev):.3f}", "nri_nonevents": f"{np.mean(ne):.3f}",
           "idi": f"{np.mean(idiv):.3f}", "idi_ci": f"{np.percentile(idiv, 2.5):.3f}-{np.percentile(idiv, 97.5):.3f}"}
    stamp(f"[{label}] ΔAUROC {out['delta_auc']} ({out['delta_auc_ci']}); NRI {out['nri']} ({out['nri_ci']}); "
          f"events {out['nri_events']}, non-events {out['nri_nonevents']}; IDI {out['idi']} ({out['idi_ci']})")
    return out


def main() -> None:
    # published DeLong machinery (same algorithm as the original submission)
    spec = importlib.util.spec_from_file_location(
        "incmod_nosep", ROOT / "03_main_analyses" /
        "run_revised_incremental_pf_vr_pf_initial_min.py")
    inc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inc)
    inc.ROOT = ROOT

    harm = pd.read_csv(OUTDIR / "harm_landmark_mimic.csv")
    if "sepsis48" not in harm.columns:
        raise SystemExit("harm_landmark_mimic.csv lacks sepsis48 — run run_master_part2 (sepsis48 build) first")

    # eICU landmark frame mirror of run_master_part2 (native first-day means; R2-2)
    eic_lm = pd.read_csv(OUTDIR / "eicu_landmark_round2.csv")
    eic_lm = eic_lm[eic_lm["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy() if \
        "hospitaldischargestatus" in eic_lm.columns else eic_lm
    # landmark frame is native-normalized by run_master_part1 (R2-2); legacy handling removed

    rows = []

    # ---------- N2: harm own-cohort ladder without sepsis ----------
    po, _ = inc.fit_predict(harm, BASE_H_NOSEP + PF)
    pn, _ = inc.fit_predict(harm, BASE_H_NOSEP + PF + VR)
    y2 = harm[OUTCOME].astype(int).values
    z2, p2, a02, a12 = inc.delong_test(y2, po, pn)
    stamp(f"[N2] dAUROC {a12 - a02:.4f}, DeLong P = {p2:.3g}")
    stamp(f"[N2] MIMIC harm landmark, NO-sepsis spec (sepsis48 in cohort: "
          f"{int(harm['sepsis48'].sum())}/{len(harm)} — removed from models)")
    ladder_names = [("Base", BASE_H_NOSEP), ("Base + P/F", BASE_H_NOSEP + PF),
                    ("Base + P/F + VR", BASE_H_NOSEP + PF + VR)]
    import statsmodels.api as sm
    for name, vars_ in ladder_names:
        x = sm.add_constant(pd.DataFrame(StandardScaler().fit_transform(harm[vars_]),
                                         columns=vars_, index=harm.index), has_constant="add")
        y = harm[OUTCOME].astype(int)
        res = sm.Logit(y, x).fit(disp=False, maxiter=300)
        a = roc_auc_score(y, res.predict(x))
        rows.append({"Row": "MIMIC harm landmark NO-sepsis (own cohort)", "Model": name,
                     "N": len(harm), "Deaths": int(y.sum()), "AUROC": f"{a:.3f}",
                     "Brier": f"{brier_score_loss(y, res.predict(x)):.3f}"})
        stamp(f"[N2 ladder] {name}: AUROC={a:.4f}")
    s_n2 = boot_internal_no_sepsis(harm, "N2 MIMIC harm NO-sepsis ladder (VR step)")

    # ---------- N3: transported (fixed no-sepsis coefficients) ----------
    eic_lm = eic_lm.dropna(subset=[OUTCOME] + BASE_H_NOSEP + PF + VR).copy()
    y_e = eic_lm[OUTCOME].astype(int)
    stamp(f"[N3] eICU landmark n={len(eic_lm)}, 48h后死亡={int(y_e.sum())}")
    for name, vars_ in ladder_names:
        so, mo = _fit(harm, vars_)
        p_e = mo.predict_proba(so.transform(eic_lm[vars_]))[:, 1]
        a_e = roc_auc_score(y_e, p_e)
        lg = np.log(np.clip(p_e, 1e-6, 1 - 1e-6) / (1 - np.clip(p_e, 1e-6, 1 - 1e-6)))
        cres = sm.Logit(y_e.astype(float), sm.add_constant(lg)).fit(disp=False, maxiter=300)
        cp = np.asarray(cres.params, dtype=float)
        rows.append({"Row": "eICU landmark NO-sepsis (transported)", "Model": name,
                     "N": len(eic_lm), "Deaths": int(y_e.sum()), "AUROC": f"{a_e:.3f}",
                     "Calibration intercept": f"{cp[0]:.3f}", "Calibration slope": f"{cp[1]:.3f}"})
        stamp(f"[N3 transported] {name}: AUROC={a_e:.4f}, 校准 {cp[0]:.3f}/{cp[1]:.3f}")

    # transported VR-step Δ with fixed-coeff bootstrap + DeLong
    so, mo = _fit(harm, BASE_H_NOSEP + PF)
    sn, mn = _fit(harm, BASE_H_NOSEP + PF + VR)
    y = eic_lm[OUTCOME].astype(int).values
    po, pn = mo.predict_proba(so.transform(eic_lm[BASE_H_NOSEP + PF]))[:, 1], \
        mn.predict_proba(sn.transform(eic_lm[BASE_H_NOSEP + PF + VR]))[:, 1]
    z, p, a0, a1 = inc.delong_test(y, po, pn)
    stamp(f"[N3] dAUROC {a1 - a0:.4f}, DeLong P = {p:.3g}")
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(RNG)
    da, nris, idi_ = [], [], []
    for _ in range(B):
        sel = np.concatenate([idx1[rng.randint(0, len(idx1), len(idx1))],
                              idx0[rng.randint(0, len(idx0), len(idx0))]])
        sub = eic_lm.iloc[sel]
        ysub = sub[OUTCOME].astype(int).values
        po_ = mo.predict_proba(so.transform(sub[BASE_H_NOSEP + PF]))[:, 1]
        pn_ = mn.predict_proba(sn.transform(sub[BASE_H_NOSEP + PF + VR]))[:, 1]
        da.append(roc_auc_score(ysub, pn_) - roc_auc_score(ysub, po_))
        nris.append(nri_idi(ysub, po_, pn_))
    da = np.array(da)
    s_n3 = {"setting": "N3 eICU landmark NO-sepsis transported (fixed coeffs)", "B": B,
            "delta_auc": f"{np.mean(da):.4f}",
            "delta_auc_ci": f"{np.percentile(da, 2.5):.4f}-{np.percentile(da, 97.5):.4f}",
            "nri": f"{np.mean([r['nri'] for r in nris]):.3f}",
            "nri_ci": f"{np.percentile([r['nri'] for r in nris], 2.5):.3f}-{np.percentile([r['nri'] for r in nris], 97.5):.3f}",
            "nri_events": f"{np.mean([r['nri_events'] for r in nris]):.3f}",
            "nri_nonevents": f"{np.mean([r['nri_nonevents'] for r in nris]):.3f}",
            "idi": f"{np.mean([r['idi'] for r in nris]):.3f}",
            "idi_ci": f"{np.percentile([r['idi'] for r in nris], 2.5):.3f}-{np.percentile([r['idi'] for r in nris], 97.5):.3f}"}
    stamp(f"[{s_n3['setting']}] ΔAUROC {s_n3['delta_auc']} ({s_n3['delta_auc_ci']}); NRI {s_n3['nri']} "
          f"({s_n3['nri_ci']}); events {s_n3['nri_events']}, non-events {s_n3['nri_nonevents']}; "
          f"IDI {s_n3['idi']} ({s_n3['idi_ci']})")

    summary = pd.DataFrame([
        {"setting": s_n2["setting"], **{k: s_n2[k] for k in
         ("delta_auc", "delta_auc_ci", "nri", "nri_ci", "nri_events", "nri_nonevents", "idi", "idi_ci")}},
        {"setting": s_n3["setting"], **{k: s_n3[k] for k in
         ("delta_auc", "delta_auc_ci", "nri", "nri_ci", "nri_events", "nri_nonevents", "idi", "idi_ci")}},
        {"setting": "N2 MIMIC harm NO-sepsis DeLong", "delta_auc": f"{a12 - a02:.4f}",
         "delta_auc_ci": "", "nri": "", "nri_ci": "", "nri_events": "", "nri_nonevents": "",
         "idi": "", "idi_ci": f"{p2:.3g}"},
        {"setting": "N3 transported VR-step DeLong", "delta_auc": f"{a1 - a0:.4f}",
         "delta_auc_ci": "", "nri": "", "nri_ci": "", "nri_events": "", "nri_nonevents": "",
         "idi": "", "idi_ci": f"{p:.3g}"},
    ])
    pd.concat([pd.DataFrame(rows), summary.rename(columns={"setting": "Row", "delta_auc": "bootstrap delta_auc"}).fillna("")],
              ignore_index=True).to_csv(OUTDIR / "nosepsis_transport_sensitivity.csv", index=False)

    # ---------- ST6 docx (Supplementary Table 6) ----------
    def build_st6_docx(df: pd.DataFrame, title_text: str, note_text: str, out_path: Path) -> None:
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
        widths = [Cm(6.4), Cm(1.6), Cm(1.7), Cm(2.6), Cm(2.6), Cm(3.4), Cm(2.4), Cm(3.0), Cm(3.0), Cm(2.6)]
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

    def fmt_ci(ci: str) -> str:
        lo, hi = ci.rsplit("-", 1)
        return f"{float(lo):.3f}-{float(hi):.3f}"

    st6 = pd.DataFrame([
        {"Analysis": "MIMIC-IV harmonized landmark, NO-sepsis (own cohort)",
         "N": int(rows[1]["N"]), "Deaths": int(rows[1]["Deaths"]),
         "AUROC before": rows[1]["AUROC"], "AUROC after": rows[2]["AUROC"],
         "Delta AUROC (95% CI)": f"{float(s_n2['delta_auc']):.3f} ({fmt_ci(s_n2['delta_auc_ci'])})",
         "DeLong P value": f"{p2:.3g}",
         "IDI (95% CI)": f"{s_n2['idi']} ({s_n2['idi_ci']})", "NRI (95% CI)": f"{s_n2['nri']} ({s_n2['nri_ci']})",
         "NRI events/non-events": f"{s_n2['nri_events']}/{s_n2['nri_nonevents']}"},
        {"Analysis": "eICU landmark NO-sepsis (transported, fixed MIMIC coefficients)",
         "N": int(rows[4]["N"]), "Deaths": int(rows[4]["Deaths"]),
         "AUROC before": rows[4]["AUROC"], "AUROC after": rows[5]["AUROC"],
         "Delta AUROC (95% CI)": f"{float(s_n3['delta_auc']):.3f} ({fmt_ci(s_n3['delta_auc_ci'])})",
         "DeLong P value": f"{p:.3g}",
         "IDI (95% CI)": f"{s_n3['idi']} ({s_n3['idi_ci']})", "NRI (95% CI)": f"{s_n3['nri']} ({s_n3['nri_ci']})",
         "NRI events/non-events": f"{s_n3['nri_events']}/{s_n3['nri_nonevents']}"},
    ])
    st6_note = (
        "The harmonized sepsis covariate in Table 3 (sepsis48) is the 48-hour-determinable Sepsis-3 definition in "
        "MIMIC-IV and the recorded admission diagnoses (admissionDx) in eICU: related constructions, not identical "
        "definitions, with markedly different positivity prevalence in the two cohorts (7,697/10,770 [71.5%] in the "
        "MIMIC-IV harmonized landmark fit cohort; 194/1,206 [16.1%] in the eICU landmark cohort). In this sensitivity "
        "analysis the sepsis covariate was removed from both sides of the transported specification and the ladder was "
        "re-run end-to-end. Before = Base + P/F model; After = additional inclusion of VR initial and VR delta "
        "(per 1-SD increase, modeled conditional on VR initial). Row 1 is fitted on the MIMIC-IV harmonized landmark "
        "cohort; Row 2 applies the fixed no-sepsis Row-1 coefficients with the MIMIC-IV-fitted scaler to the eICU "
        "landmark cohort. Delta/IDI/NRI are paired-model statistics with bootstrap 95% confidence intervals "
        "(2,000 draws; derivation settings refit per resample; the transported analysis evaluated with fixed "
        "coefficients per resample). The sepsis48 specification (Table 3, Rows 2-3) remains primary: the VR step "
        "preserves its direction and reporting-precision magnitude in both settings without sepsis, and the "
        "transported step remains nominally non-significant under either specification (DeLong P = 0.073 with, "
        "0.177 without).")
    build_st6_docx(st6, "Supplementary Table 6. Sensitivity analysis: the harmonized/transported 48-hour landmark ladder without the sepsis covariate.",
                   st6_note, OUTDIR / "TableS6_nosepsis_TNR_xiaoliu.docx")
    stamp("ST6 docx 完成")

    (OUTDIR / "run_nosepsis_transport_log.md").write_text(
        "# NO-sepsis transported sensitivity (Reviewer 2-2)\n\n"
        "- Harmonized/transported 48-h landmark ladder WITHOUT the sepsis covariate in both databases.\n"
        "- N2: MIMIC harm own cohort (stratified bootstrap refit, B=2,000); N3: fixed no-sepsis "
        "MIMIC coefficients + MIMIC-fitted scaler applied to eICU landmark (fixed-coeff bootstrap).\n\n"
        + "\n".join(LOG), encoding="utf-8")
    stamp("NO-sepsis 敏感性完成")


if __name__ == "__main__":
    main()
