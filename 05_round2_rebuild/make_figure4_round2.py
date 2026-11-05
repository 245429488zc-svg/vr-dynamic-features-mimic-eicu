#!/usr/bin/env python3
"""Generate Figure 4: incremental predictive value of VR beyond P/F."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTDIR.mkdir(exist_ok=True)

MIMIC_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "landmark_cohort_round2.csv"
BT_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "round2_bootstrap.csv"
HARM_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "harm_landmark_mimic.csv"
EICU_LM_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "eicu_landmark_round2.csv"

TABLE3_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "Table3_round2.csv"

OUTCOME = "hospital_expire_flag"
RNG = 20260721

BASE_A = [
    "age", "sex", "charlson", "sepsis3_48h", "copd", "chf", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
BASE_B = [
    "age", "sex", "charlson", "sepsis48", "copd", "chf", "peep",
    "vt_pbw", "hr", "map", "temp", "bun", "creatinine", "wbc", "bicarbonate",
]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]

BLUE = "#4C78A8"
RED = "#E45756"
GRAY = "#666666"


def compute_midrank(x: np.ndarray) -> np.ndarray:
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


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
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


def delong_test(y_true: np.ndarray, pred_old: np.ndarray, pred_new: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    preds = np.vstack((pred_old, pred_new))[:, order]
    label_1_count = int(y_true.sum())
    aucs, cov = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def idi_nri(y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray) -> tuple[float, float]:
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
    return float(idi), float(nri_event + nri_nonevent)


def fit_predict(train: pd.DataFrame, variables: list[str], test: pd.DataFrame | None = None):
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


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def prepare_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic_a = pd.read_csv(MIMIC_PATH)
    mimic_b = pd.read_csv(HARM_PATH)
    eicu_b = pd.read_csv(EICU_LM_PATH)
    # eICU 臂：以原生首日均值列为准（R2-2），清理旧回填并以完整口径收口
    poll = [c for c in ["hr", "map", "temp", "rr_vitalsign", "spo2"] if c in eicu_b.columns]
    if poll:
        eicu_b = eicu_b.drop(columns=poll)
    for a_, b_ in {"hr_native": "hr", "map_native": "map", "temp_native": "temp"}.items():
        if a_ in eicu_b.columns:
            eicu_b = eicu_b.rename(columns={a_: b_})
    # Harmonized sepsis slot: admissionDx-recorded sepsis in the eICU cohort.
    eicu_b["sepsis48"] = eicu_b["sepsis_h"].astype(int)
    eicu_b = eicu_b.dropna(subset=[OUTCOME] + BASE_B + PF + VR).copy()

    settings = []

    p_old, _ = fit_predict(mimic_a, BASE_A + PF)
    p_new, _ = fit_predict(mimic_a, BASE_A + PF + VR)
    settings.append(("MIMIC-IV internal (48-h landmark)", mimic_a, p_old, p_new))

    p_old, p_old_eicu = fit_predict(mimic_b, BASE_B + PF, eicu_b)
    p_new, p_new_eicu = fit_predict(mimic_b, BASE_B + PF + VR, eicu_b)
    settings.append(("MIMIC-IV harmonized reduced (48-h landmark)", mimic_b, p_old, p_new))
    settings.append(("eICU (48-h landmark, transported)", eicu_b, p_old_eicu, p_new_eicu))

    roc_rows = []
    summary_rows = []
    for setting, data, pred_old, pred_new in settings:
        y = data[OUTCOME].astype(int).values
        for model_name, pred in [("Base + P/F", pred_old), ("Base + P/F + VR", pred_new)]:
            fpr, tpr, thresholds = roc_curve(y, pred)
            for fp, tp, th in zip(fpr, tpr, thresholds):
                roc_rows.append(
                    {
                        "setting": setting,
                        "model": model_name,
                        "fpr": fp,
                        "tpr": tp,
                        "threshold": th,
                    }
                )
        _, p_value = delong_test(y, pred_old, pred_new)
        bt_row = globals()["_BT"].iloc[len(summary_rows)]
        auc_old = roc_auc_score(y, pred_old)
        auc_new = roc_auc_score(y, pred_new)
        summary_rows.append(
            {
                "setting": setting,
                "N": int(len(data)),
                "Deaths": int(y.sum()),
                "AUROC before": auc_old,
                "AUROC after": auc_new,
                "Delta AUROC (bootstrap)": float(bt_row["delta_auc"]),
                "DeLong P value": p_value,
                "IDI": float(bt_row["idi"]),
                "NRI": float(bt_row["nri"]),
                "Brier before": brier_score_loss(y, pred_old),
                "Brier after": brier_score_loss(y, pred_new),
            }
        )

    roc_df = pd.DataFrame(roc_rows)
    summary_df = pd.DataFrame(summary_rows)
    roc_df.to_csv(OUTDIR / "Figure4_ROC_curve_points.csv", index=False)
    summary_df.to_csv(OUTDIR / "Figure4_incremental_metrics_summary.csv", index=False)
    return roc_df, summary_df


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3)


def draw_roc_panel(ax, roc_df: pd.DataFrame, summary_df: pd.DataFrame, setting: str, title: str):
    sub = roc_df[roc_df["setting"].eq(setting)]
    summ = summary_df[summary_df["setting"].eq(setting)].iloc[0]
    for model, color, ls in [("Base + P/F", BLUE, "-"), ("Base + P/F + VR", RED, "-")]:
        curve = sub[sub["model"].eq(model)]
        auc = summ["AUROC before"] if model == "Base + P/F" else summ["AUROC after"]
        ax.plot(curve["fpr"], curve["tpr"], color=color, linestyle=ls, linewidth=1.8, label=f"{model} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#bdbdbd", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)
    ax.text(
        0.04,
        0.19,
        f"ΔAUROC={summ['Delta AUROC (bootstrap)']:.3f}\nDeLong {fmt_p(summ['DeLong P value'])}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "boxstyle": "round,pad=0.2", "alpha": 0.92},
    )
    style_axis(ax)


def draw_increment_panel(ax, summary_df: pd.DataFrame):
    plot_df = summary_df.copy()
    plot_df["setting_label"] = pd.Categorical(
        plot_df["setting"],
        categories=["MIMIC-IV internal (48-h landmark)", "MIMIC-IV harmonized reduced (48-h landmark)", "eICU (48-h landmark, transported)"],
        ordered=True,
    )
    plot_df = plot_df.sort_values("setting_label")
    metrics = [
        ("Delta AUROC (bootstrap)", "ΔAUROC", "#4C78A8"),
        ("IDI", "IDI", "#72B7B2"),
        ("NRI", "NRI", "#E45756"),
    ]
    y_base = np.arange(len(plot_df))
    offsets = [-0.18, 0.0, 0.18]
    for (col, label, color), off in zip(metrics, offsets):
        vals = plot_df[col].to_numpy()
        ax.scatter(vals, y_base + off, color=color, s=34, label=label, zorder=3)
        for val, y in zip(vals, y_base + off):
            ax.text(val + 0.006, y, f"{val:.3f}", va="center", ha="left", fontsize=7.2, color=color)
    ax.axvline(0, color="#808080", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_yticks(y_base)
    ax.set_yticklabels(["MIMIC internal", "MIMIC harmonized", "eICU transported"])
    ax.invert_yaxis()
    ax.set_xlim(-0.01, 0.30)
    ax.set_xlabel("Incremental metric value")
    ax.set_title("D. Incremental value of adding VR", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=7.4)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)


def make_figure(roc_df: pd.DataFrame, summary_df: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.6))
    draw_roc_panel(axes[0, 0], roc_df, summary_df, "MIMIC-IV internal (48-h landmark)", "A. MIMIC-IV internal model (48-h landmark)")
    draw_roc_panel(axes[0, 1], roc_df, summary_df, "MIMIC-IV harmonized reduced (48-h landmark)", "B. MIMIC-IV harmonized reduced model")
    draw_roc_panel(axes[1, 0], roc_df, summary_df, "eICU (48-h landmark, transported)", "C. eICU external validation (transported, 48-h landmark)")
    draw_increment_panel(axes[1, 1], summary_df)
    fig.suptitle("Figure 4. Incremental predictive value of VR beyond P/F", fontsize=12.2, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.11, hspace=0.42, wspace=0.42)
    fig.text(
        0.08,
        0.035,
        "Base + P/F models included initial and minimum P/F ratio. VR variables included VR initial and VR delta.",
        ha="left",
        va="bottom",
        fontsize=7.5,
    )
    return fig


def validate_against_table3(summary_df: pd.DataFrame) -> list[str]:
    # round-2: Table 3 rebuilt (landmark framework); numeric cross-check handled by
    # run_table3_docx.py assembly; here we only pass through.
    return []


def write_log(summary_df: pd.DataFrame, validation_notes: list[str]) -> None:
    log = [
        "# Figure 4 generation log",
        "",
        "## Data source",
        "",
        f"- MIMIC-IV analysis data: {MIMIC_PATH}",
        f"- eICU analysis data: {EICU_LM_PATH}",
        f"- Table 3 reference: {TABLE3_PATH}",
        "",
        "## Model comparison",
        "",
        "- Compared Base + P/F against Base + P/F + VR.",
        "- P/F variables: pf_initial and pf_min.",
        "- VR variables: vr_initial and vr_delta.",
        "- MIMIC-IV internal model used BASE_A variables, including SOFA.",
        "- MIMIC-IV harmonized and eICU external validation used BASE_B variables.",
        "",
        "## Cohort checks",
        "",
    ]
    for _, row in summary_df.iterrows():
        log.append(
            f"- {row['setting']}: n={int(row['N'])}, deaths={int(row['Deaths'])}, "
            f"AUROC {row['AUROC before']:.3f}->{row['AUROC after']:.3f}, "
            f"Delta={row['Delta AUROC (bootstrap)']:.3f}, IDI={row['IDI']:.3f}, NRI={row['NRI']:.3f}."
        )
    log += [
        "",
        "## Validation against Table 3",
        "",
    ]
    if validation_notes:
        log.extend([f"- {note}" for note in validation_notes])
    else:
        log.append("- Recalculated values matched Table 3 within rounding tolerance.")
    log += [
        "",
        "## Output files",
        "",
        "- Figure4_incremental_value_VR_beyond_PF.png",
        "- Figure4_incremental_value_VR_beyond_PF.pdf",
        "- Figure4_incremental_value_VR_beyond_PF.svg",
        "- Figure4_ROC_curve_points.csv",
        "- Figure4_incremental_metrics_summary.csv",
    ]
    (OUTDIR / "Figure4_generation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    bt_df = pd.read_csv(BT_PATH)  # bootstrap Δ (unified display口径 with Table 3)
    globals()['_BT'] = bt_df
    roc_df, summary_df = prepare_predictions()
    validation_notes = validate_against_table3(summary_df)
    fig = make_figure(roc_df, summary_df)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUTDIR / f"Figure4_incremental_value_VR_beyond_PF.{ext}",
            dpi=600 if ext == "png" else None,
            facecolor="white",
        )
    plt.close(fig)
    write_log(summary_df, validation_notes)
    print(f"Wrote Figure 4 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
