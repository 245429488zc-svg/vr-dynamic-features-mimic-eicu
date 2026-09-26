#!/usr/bin/env python3
"""Recompute PF/VR incremental value under revised complete-case models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "12_Table3正式版"
MIMIC_PATH = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
EICU_PATH = ROOT / "1_队列数据" / "eICU_analysis_ready.csv"
OUTCOME = "hospital_expire_flag"
RNG = 20260721

BASE_A = [
    "age", "sex", "charlson", "sepsis3", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
BASE_B = [
    "age", "sex", "charlson", "sepsis3", "peep", "vt_pbw", "hr",
    "map", "temp", "bun", "creatinine", "wbc", "bicarbonate",
]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]


def compute_midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=float)
    out[j] = t
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    preds = np.vstack((pred_a, pred_b))[:, order]
    label_1_count = int(y_true.sum())
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p), float(aucs[0]), float(aucs[1])


def idi_nri(y, p_old, p_new):
    y = np.asarray(y).astype(int)
    old_event = p_old[y == 1].mean()
    old_nonevent = p_old[y == 0].mean()
    new_event = p_new[y == 1].mean()
    new_nonevent = p_new[y == 0].mean()
    idi = (new_event - new_nonevent) - (old_event - old_nonevent)

    event_diff = p_new[y == 1] - p_old[y == 1]
    nonevent_diff = p_new[y == 0] - p_old[y == 0]
    nri_event = (event_diff > 0).mean() - (event_diff < 0).mean()
    nri_nonevent = (nonevent_diff < 0).mean() - (nonevent_diff > 0).mean()
    nri = nri_event + nri_nonevent
    return float(idi), float(nri), float(nri_event), float(nri_nonevent)


def fit_predict(train, variables, test=None):
    scaler = StandardScaler().fit(train[variables])
    x_train = scaler.transform(train[variables])
    y_train = train[OUTCOME].astype(int).values
    lr = LogisticRegression(penalty=None, max_iter=3000, random_state=RNG)
    lr.fit(x_train, y_train)
    p_train = lr.predict_proba(x_train)[:, 1]
    p_test = None
    if test is not None:
        p_test = lr.predict_proba(scaler.transform(test[variables]))[:, 1]
    return p_train, p_test


def build_models(df, base_vars):
    model_vars = {
        "Base": base_vars,
        "Base + P/F": base_vars + PF,
        "Base + VR": base_vars + VR,
        "Base + P/F + VR": base_vars + PF + VR,
    }
    return model_vars


def evaluate_dataset(train, model_vars, label, test=None):
    probs_train = {}
    probs_test = {}
    perf_rows = []
    y_train = train[OUTCOME].astype(int).values
    y_test = test[OUTCOME].astype(int).values if test is not None else None
    for name, vars_ in model_vars.items():
        p_train, p_test = fit_predict(train, vars_, test)
        probs_train[name] = p_train
        if p_test is not None:
            probs_test[name] = p_test
        row = {
            "Database": label,
            "Model": name,
            "N": int(len(train)),
            "Events": int(y_train.sum()),
            "AUROC": roc_auc_score(y_train, p_train),
            "Brier": brier_score_loss(y_train, p_train),
        }
        if p_test is not None:
            row.update({
                "External N": int(len(test)),
                "External events": int(y_test.sum()),
                "External AUROC": roc_auc_score(y_test, p_test),
                "External Brier": brier_score_loss(y_test, p_test),
            })
        perf_rows.append(row)
    return pd.DataFrame(perf_rows), probs_train, probs_test


def comparison_rows(y, probs, db):
    pairs = [
        ("P/F over Base", "Base", "Base + P/F"),
        ("VR over Base", "Base", "Base + VR"),
        ("VR over Base + P/F", "Base + P/F", "Base + P/F + VR"),
        ("P/F over Base + VR", "Base + VR", "Base + P/F + VR"),
    ]
    rows = []
    for label, old, new in pairs:
        z, p, auc_old, auc_new = delong_test(y, probs[old], probs[new])
        idi, nri, nri_event, nri_nonevent = idi_nri(y, probs[old], probs[new])
        rows.append({
            "Comparison": label,
            "Database": db,
            "AUROC before": auc_old,
            "AUROC after": auc_new,
            "Delta AUROC": auc_new - auc_old,
            "DeLong z": z,
            "P value": p,
            "IDI": idi,
            "NRI": nri,
            "NRI events": nri_event,
            "NRI nonevents": nri_nonevent,
        })
    return rows


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    mimic = pd.read_csv(MIMIC_PATH)
    eicu = pd.read_csv(EICU_PATH)

    a_vars = BASE_A + PF + VR
    b_vars = BASE_B + PF + VR
    mimic_a = mimic.dropna(subset=[OUTCOME] + a_vars).copy()
    mimic_b = mimic.dropna(subset=[OUTCOME] + b_vars).copy()
    eicu_b = eicu.dropna(subset=[OUTCOME] + b_vars).copy()

    a_model_vars = build_models(mimic_a, BASE_A)
    b_model_vars = build_models(mimic_b, BASE_B)

    perf_a, probs_a, _ = evaluate_dataset(mimic_a, a_model_vars, "MIMIC internal")
    perf_b, probs_b_m, probs_b_e = evaluate_dataset(mimic_b, b_model_vars, "MIMIC derivation", eicu_b)

    comp_rows = []
    comp_rows.extend(comparison_rows(mimic_a[OUTCOME].values, probs_a, "MIMIC internal"))
    comp_rows.extend(comparison_rows(mimic_b[OUTCOME].values, probs_b_m, "MIMIC derivation"))
    comp_rows.extend(comparison_rows(eicu_b[OUTCOME].values, probs_b_e, "eICU external"))

    perf = pd.concat([perf_a, perf_b], ignore_index=True)
    comp = pd.DataFrame(comp_rows)
    perf.to_csv(OUTDIR / "Table3_model_ladder_pf_initial_min.csv", index=False)
    comp.to_csv(OUTDIR / "Table3_incremental_value_Delong_IDI_NRI_pf_initial_min.csv", index=False)

    pub = comp.copy()
    for c in ["AUROC before", "AUROC after", "Delta AUROC", "DeLong z", "IDI", "NRI", "NRI events", "NRI nonevents"]:
        pub[c] = pub[c].map(lambda x: f"{x:.4f}" if c not in ["DeLong z"] else f"{x:.3f}")
    pub["P value"] = comp["P value"].map(fmt_p)
    pub.to_csv(OUTDIR / "Table3_incremental_value_pub_pf_initial_min.csv", index=False)

    log = [
        "# 新版 P/F 与 VR 增量价值分析",
        "",
        "- 使用新版低缺失完整病例口径重新计算。",
        "- P/F 特征：pf_initial + pf_min。",
        "- VR 特征：vr_initial + vr_delta。",
        "- 比较包括 VR over Base + P/F，以及 P/F over Base + VR。",
        "- DeLong/IDI/NRI 均基于新版模型预测概率重新计算，旧 Table 3 不再沿用。",
        "",
        "## 关键结论",
        "",
    ]
    key = comp[comp["Comparison"].eq("VR over Base + P/F")]
    for _, r in key.iterrows():
        log.append(
            f"- {r['Database']}: adding VR to Base+P/F changed AUROC "
            f"{r['AUROC before']:.3f} -> {r['AUROC after']:.3f}; "
            f"DeLong P={fmt_p(r['P value'])}; IDI={r['IDI']:.4f}; NRI={r['NRI']:.4f}."
        )
    (OUTDIR / "pf_initial_min_incremental_recalculation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote revised incremental analysis to {OUTDIR}")


if __name__ == "__main__":
    main()
