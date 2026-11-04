#!/usr/bin/env python3
"""Round-2 master, part 2 (B): bootstrap CIs (B=2000) for the three Table-3 VR steps.

Settings (mirroring 35_增量bootstrap's design on the ROUND-2 rebuilt world):
  S1 MIMIC landmark FULL-spec ladder : old = Base+P/F (20), new = +VR (22);
     stratified bootstrap WITH refit (like the derivation setting).
  S2 MIMIC landmark HARMONIZED ladder: old = Base_h+P/F (17), new = +VR (19);
     stratified bootstrap WITH refit.
  S3 eICU landmark TRANSPORTED       : fixed MIMIC coefficients applied to resampled
     eICU rows (no refit), like the original eICU handling.

For each: ΔAUROC 95% CI; category-free NRI (events/non-events), IDI with CIs.
Outputs: round2_bootstrap.csv + log md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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


def ladder_probs(df: pd.DataFrame, base: list[str], label: str) -> tuple[np.ndarray, np.ndarray, object]:
    """old = base+PF; new = base+PF+VR; returns (p_old_full, p_new_full, fit_new)."""
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression as LR
    def _fit(vars_):
        scaler = StandardScaler().fit(df[vars_])
        m = LR(penalty=None, max_iter=3000, random_state=RNG)
        m.fit(scaler.transform(df[vars_]), df[OUTCOME].astype(int))
        return scaler, m
    so, mo = _fit(base + PF)
    sn, mn = _fit(base + PF + VR)
    po = mo.predict_proba(so.transform(df[base + PF]))[:, 1]
    pn = mn.predict_proba(sn.transform(df[base + PF + VR]))[:, 1]
    return po, pn, (so, mo)


def boot_internal(df: pd.DataFrame, base: list[str], label: str) -> dict:
    import statsmodels.api as sm
    y = df[OUTCOME].astype(int).values
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(RNG)
    d_aucs, nris, ids_ = [], [], []
    for _ in range(B):
        s1 = idx1[rng.randint(0, len(idx1), len(idx1))]
        s0 = idx0[rng.randint(0, len(idx0), len(idx0))]
        sel = np.concatenate([s1, s0])
        sub = df.iloc[sel]
        def _fit(vars_):
            scaler = StandardScaler().fit(sub[vars_])
            m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
            m.fit(scaler.transform(sub[vars_]), sub[OUTCOME].astype(int))
            return scaler, m
        so_, mo_ = _fit(base + PF)
        sn_, mn_ = _fit(base + PF + VR)
        po = mo_.predict_proba(so_.transform(sub[base + PF]))[:, 1]
        pn = mn_.predict_proba(sn_.transform(sub[base + PF + VR]))[:, 1]
        ysub = sub[OUTCOME].astype(int).values
        d_aucs.append(roc_auc_score(ysub, pn) - roc_auc_score(ysub, po))
        r = nri_idi(ysub, po, pn)
        nris.append(r)
    da = np.array(d_aucs)
    lo_nri = np.array([x["nri"] for x in nris]); ev = np.array([x["nri_events"] for x in nris])
    ne = np.array([x["nri_nonevents"] for x in nris]); idi = np.array([x["idi"] for x in nris])
    out = {"setting": label, "B": B,
           "delta_auc": f"{np.mean(da):.4f}",
           "delta_auc_ci": f"{np.percentile(da, 2.5):.4f}-{np.percentile(da, 97.5):.4f}",
           "nri": f"{np.mean(lo_nri):.3f}",
           "nri_ci": f"{np.percentile(lo_nri, 2.5):.3f}-{np.percentile(lo_nri, 97.5):.3f}",
           "nri_events": f"{np.mean(ev):.3f}", "nri_nonevents": f"{np.mean(ne):.3f}",
           "idi": f"{np.mean(idi):.3f}", "idi_ci": f"{np.percentile(idi, 2.5):.3f}-{np.percentile(idi, 97.5):.3f}"}
    stamp(f"[{label}] ΔAUROC {out['delta_auc']} ({out['delta_auc_ci']}); NRI {out['nri']} ({out['nri_ci']}); "
          f"events {out['nri_events']}, non-events {out['nri_nonevents']}; IDI {out['idi']} ({out['idi_ci']})")
    return out


def boot_transport(df_eicu: pd.DataFrame, coeffs: object, base: list[str], label: str) -> dict:
    """Fixed-coefficient transport bootstrap: resample eICU rows, no refit."""
    import statsmodels.api as sm
    y = df_eicu[OUTCOME].astype(int).values
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(RNG)
    scaler_m, models = coeffs
    d_aucs, nris, ids_ = [], [], []
    for _ in range(B):
        sel = np.concatenate([idx1[rng.randint(0, len(idx1), len(idx1))],
                              idx0[rng.randint(0, len(idx0), len(idx0))]])
        sub = df_eicu.iloc[sel]
        ysub = sub[OUTCOME].astype(int).values
        po = models[0].predict_proba(models[0].scaler_.transform(sub[base + PF]) if False else None)[:, 1] if False else None  # placeholder
        break
    return {}

def main() -> None:
    mim = pd.read_csv(OUTDIR / "mimic_analysis_ready_round2.csv")

    # S1: MIMIC landmark FULL-spec
    lm = pd.read_csv(OUTDIR / "landmark_cohort_round2.csv")
    lm = mim[mim["stay_id"].isin(lm["stay_id"])].copy()
    s1 = boot_internal(lm, BASE_FULL, "S1 MIMIC landmark FULL-spec ladder (VR step)")

    # S3: eICU transported (fit on harm-own, apply fixed)
    harm = pd.read_csv(OUTDIR / "harm_landmark_mimic.csv")
    eic = pd.read_csv(OUTDIR / "eicu_analysis_ready_round2.csv")
    eic_lm = pd.read_csv(OUTDIR / "eicu_landmark_round2.csv")
    eic_lm = eic_lm[eic_lm["hospitaldischargestatus"].isin(["Alive", "Expired"])].copy()
    poll = [c for c in ["hr", "map", "temp", "rr_vitalsign", "spo2"] if c in eic_lm.columns]
    eic_lm = eic_lm.drop(columns=poll)
    for a, b in {"hr_native": "hr", "map_native": "map", "temp_native": "temp"}.items():
        if a in eic_lm.columns:
            eic_lm = eic_lm.rename(columns={a: b})
    eic_lm = eic_lm.dropna(subset=[OUTCOME] + BASE_H + PF + VR).copy()
    stamp(f"bootstrap eICU-lm: n={len(eic_lm)}")
    harm["sepsis_h"] = harm["sepsis_h"].astype(int)

    # fixed-coeff transport bootstrap (mirror: resample eICU rows, no refit)
    def fit_scaled(df, vars_):
        scaler = StandardScaler().fit(df[vars_])
        m = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
        m.fit(scaler.transform(df[vars_]), df[OUTCOME].astype(int))
        return scaler, m
    sc_o, m_o = fit_scaled(harm, BASE_H + PF)
    sc_n, m_n = fit_scaled(harm, BASE_H + PF + VR)
    y = eic_lm[OUTCOME].astype(int).values
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.RandomState(RNG)
    d_aucs, nris, idi_ar = [], [], []
    for _ in range(B):
        sel = np.concatenate([idx1[rng.randint(0, len(idx1), len(idx1))],
                              idx0[rng.randint(0, len(idx0), len(idx0))]])
        sub = eic_lm.iloc[sel]
        ysub = sub[OUTCOME].astype(int).values
        po = m_o.predict_proba(sc_o.transform(sub[BASE_H + PF]))[:, 1]
        pn = m_n.predict_proba(sc_n.transform(sub[BASE_H + PF + VR]))[:, 1]
        d_aucs.append(roc_auc_score(ysub, pn) - roc_auc_score(ysub, po))
        r = nri_idi(ysub, po, pn)
        nris.append(r)
    da = np.array(d_aucs)
    nr_ = np.array([x["nri"] for x in nris]); ev_ = np.array([x["nri_events"] for x in nris])
    ne_ = np.array([x["nri_nonevents"] for x in nris]); idi_ = np.array([x["idi"] for x in nris])
    s3 = {"setting": "S3 eICU landmark transported (fixed coeffs)", "B": B,
          "delta_auc": f"{np.mean(da):.4f}",
          "delta_auc_ci": f"{np.percentile(da, 2.5):.4f}-{np.percentile(da, 97.5):.4f}",
          "nri": f"{np.mean(nr_):.3f}",
          "nri_ci": f"{np.percentile(nr_, 2.5):.3f}-{np.percentile(nr_, 97.5):.3f}",
          "nri_events": f"{np.mean(ev_):.3f}", "nri_nonevents": f"{np.mean(ne_):.3f}",
          "idi": f"{np.mean(idi_):.3f}", "idi_ci": f"{np.percentile(idi_, 2.5):.3f}-{np.percentile(idi_, 97.5):.3f}"}
    stamp(f"[S3] ΔAUROC {s3['delta_auc']} ({s3['delta_auc_ci']}); NRI {s3['nri']} ({s3['nri_ci']}); "
          f"events {s3['nri_events']}, non-events {s3['nri_nonevents']}; IDI {s3['idi']} ({s3['idi_ci']})")

    # S2: harm ladder own-cohort
    s2 = boot_internal(harm, BASE_H, "S2 MIMIC landmark HARMONIZED ladder (VR step)")

    pd.DataFrame([s1, s2, s3]).to_csv(OUTDIR / "round2_bootstrap.csv", index=False)
    (OUTDIR / "round2_bootstrap_log.md").write_text(
        "# Bootstrap CIs（B=2000，round-2 重建世界）\n\n"
        "- S1/S2：按结局分层有放回重采样并重拟合；S3：MIMIC 固定系数应用于重采样的 eICU 行。\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp("bootstrap 完成")


if __name__ == "__main__":
    main()