#!/usr/bin/env python3
"""Round-2 master, part 4: DeLong P for the three Table-3 VR steps + a corrected-
specification full-cohort delta bootstrap on the rebuilt association cohort
(for the response-letter "repeat the affected analyses" line).

Imports the PUBLISHED DeLong machinery from
42_GitHub仓库/vr-dynamic-features-code/03_main_analyses/run_revised_incremental_pf_vr_pf_initial_min.py
(compute_midrank / fast_delong / delong_test / fit_predict) so the round-2 table notes
use the same algorithm as the original submission.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTCOME = "hospital_expire_flag"
RNG = 20260721
B = 2000

BASE_FULL = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
             "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
             "bun", "creatinine", "wbc", "bicarbonate"]
BASE_PRED = ["sepsis3_48h" if col == "sepsis3" else col for col in BASE_FULL]
# harmonized slot: sepsis48 = 48-h-determinable sepsis (48-h Sepsis-3 on the MIMIC
# fit side; admissionDx-recorded sepsis on the eICU side in the S3 transport/resample)
BASE_H = ["age", "sex", "charlson", "sepsis48", "copd", "chf", "peep",
          "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-association-bootstrap", action="store_true",
                        help="Recompute the three landmark DeLong comparisons only; the association model is unchanged.")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "incmod", ROOT / "03_main_analyses" /
        "run_revised_incremental_pf_vr_pf_initial_min.py")
    inc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inc)
    inc.ROOT = ROOT
    mod_RNG = getattr(inc, "RNG", RNG)

    mim = pd.read_csv(OUTDIR / "mimic_analysis_ready_round2.csv")
    lm_ids = pd.read_csv(OUTDIR / "landmark_cohort_round2.csv", usecols=["stay_id"])
    lm = mim[mim["stay_id"].isin(lm_ids["stay_id"])].copy()
    harm = pd.read_csv(OUTDIR / "harm_landmark_mimic.csv")
    eic = pd.read_csv(OUTDIR / "eicu_analysis_ready_round2.csv")
    eic_lm = pd.read_csv(OUTDIR / "eicu_landmark_round2.csv")
    eic_lm = eic_lm[eic_lm["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy()
    # landmark frame is native-normalized by run_master_part1 (R2-2); legacy handling removed
    eic_lm = eic_lm.dropna(subset=[OUTCOME] + BASE_H + PF + VR).copy()
    stamp(f"对比臂: MIMIC-lm {len(lm)}, harm-lm {len(harm)}, eICU-lm {len(eic_lm)}")

    rows = []
    for label, df, base in [
        ("S1 MIMIC landmark FULL-spec (VR step)", lm, BASE_PRED),
        ("S2 MIMIC landmark HARMONIZED (VR step)", harm, BASE_H),
    ]:
        po, _ = inc.fit_predict(df, base + PF)
        pn, _ = inc.fit_predict(df, base + PF + VR)
        y = df[OUTCOME].astype(int).values
        z, p, a0, a1 = inc.delong_test(y, po, pn)
        rows.append({"setting": label, "delta": f"{a1 - a0:.4f}", "delong_p": f"{p:.3g}", "mean": "", "ci": ""})
        stmt(f"[{label}] dAUROC {a1 - a0:.4f}, DeLong P = {p:.3g}") if False else stamp(f"[{label}] dAUROC {a1 - a0:.4f}, DeLong P = {p:.3g}")

    # S3: TRUE transported step (harm-own fixed coefficients applied to eICU landmark)
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS
    _so = _SS().fit(harm[BASE_H + PF]); _mo = _LR(penalty=None, max_iter=3000, random_state=RNG).fit(
        _so.transform(harm[BASE_H + PF]), harm[OUTCOME].astype(int))
    _sn = _SS().fit(harm[BASE_H + PF + VR]); _mn = _LR(penalty=None, max_iter=3000, random_state=RNG).fit(
        _sn.transform(harm[BASE_H + PF + VR]), harm[OUTCOME].astype(int))
    y_e = eic_lm[OUTCOME].astype(int).values
    _po = _mo.predict_proba(_so.transform(eic_lm[BASE_H + PF]))[:, 1]
    _pn = _mn.predict_proba(_sn.transform(eic_lm[BASE_H + PF + VR]))[:, 1]
    _z, _p, _a0, _a1 = inc.delong_test(y_e, _po, _pn)
    rows.append({"setting": "S3 eICU landmark TRANSPORTED (fixed coeffs, VR step)",
                 "delta": f"{_a1 - _a0:.4f}", "delong_p": f"{_p:.3g}", "mean": "", "ci": ""})
    stamp(f"[S3 transported] dAUROC {_a1 - _a0:.4f}, DeLong P = {_p:.3g}")

    if not args.skip_association_bootstrap:
        # Unchanged association model: run only when its independent bootstrap is needed.
        cc = mim.dropna(subset=[OUTCOME] + BASE_FULL + PF + VR).copy()
        y = cc[OUTCOME].astype(int).values
        i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
        rng = np.random.default_rng(RNG)
        ds = []
        for _ in range(B):
            sel = np.concatenate([rng.choice(i1, len(i1), True), rng.choice(i0, len(i0), True)])
            sub = cc.iloc[sel]
            po, _ = inc.fit_predict(sub, BASE_FULL + PF)
            pn, _ = inc.fit_predict(sub, BASE_FULL + PF + VR)
            ysub = sub[OUTCOME].astype(int).values
            ds.append(roc_auc_score(ysub, pn) - roc_auc_score(ysub, po))
        a = np.array(ds)
        rows.append({"setting": "对照·重建关联队列全队列 FULL-spec ladder (VR step, corrected spec)",
                     "mean": f"{np.mean(a):.4f}",
                     "ci": f"{np.percentile(a, 2.5):.4f}-{np.percentile(a, 97.5):.4f}"})
        stamp(f"[对照臂] Δ{np.mean(a):.4f} ({np.percentile(a, 2.5):.4f}-{np.percentile(a, 97.5):.4f})")
    else:
        stamp("[对照臂] skipped: association model and covariates unchanged")

    pd.DataFrame(rows).to_csv(OUTDIR / "delong_and_contrast.csv", index=False)
    (OUTDIR / "delong_and_contrast_log.md").write_text(
        "# DeLong P 与对照 bootstrap（round-2, 复用已发表算法）\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Part 4 完成")


if __name__ == "__main__":
    main()
