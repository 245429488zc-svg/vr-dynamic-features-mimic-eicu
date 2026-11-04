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

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUTDIR = ROOT / "43_第二轮重分析"
OUTCOME = "hospital_expire_flag"
RNG = 20260721
B = 2000

BASE_FULL = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
             "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
             "bun", "creatinine", "wbc", "bicarbonate"]
BASE_H = ["age", "sex", "charlson", "sepsis_h", "copd", "chf", "peep",
          "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "incmod", ROOT / "42_GitHub仓库" / "vr-dynamic-features-code" / "03_main_analyses" /
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
    for col in ["hr", "map", "temp", "rr_vitalsign", "spo2"]:
        if col in eic_lm.columns:
            eic_lm = eic_lm.drop(columns=[col])
    for a, b in {"hr_native": "hr", "map_native": "map", "temp_native": "temp"}.items():
        if a in eic_lm.columns:
            eic_lm = eic_lm.rename(columns={a: b})
    eic_lm = eic_lm.dropna(subset=[OUTCOME] + BASE_H + PF + VR).copy()
    stamp(f"对比臂: MIMIC-lm {len(lm)}, harm-lm {len(harm)}, eICU-lm {len(eic_lm)}")

    rows = []
    for label, df, base in [
        ("S1 MIMIC landmark FULL-spec (VR step)", lm, BASE_FULL),
        ("S2 MIMIC landmark HARMONIZED (VR step)", harm, BASE_H),
        ("S3 eICU landmark transported (VR step)", eic_lm, BASE_H),
    ]:
        po, _ = inc.fit_predict(df, base + PF)
        pn, _ = inc.fit_predict(df, base + PF + VR)
        y = df[OUTCOME].astype(int).values
        z, p, a0, a1 = inc.delong_test(y, po, pn)
        rows.append({"setting": label, "delta": f"{a1 - a0:.4f}", "delong_p": f"{p:.3g}"})
        stamp(f"[{label}] ΔAUROC {a1 - a0:.4f}, DeLong P = {p:.3g}")

    # 对照臂：重建关联队列 + 修正规格（BASE_FULL 18协变量）全队列 delta bootstrap
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

    pd.DataFrame(rows).to_csv(OUTDIR / "delong_and_contrast.csv", index=False)
    (OUTDIR / "delong_and_contrast_log.md").write_text(
        "# DeLong P 与对照 bootstrap（round-2, 复用已发表算法）\n\n" + "\n".join(LOG), encoding="utf-8")
    stamp("Part 4 完成")


if __name__ == "__main__":
    main()