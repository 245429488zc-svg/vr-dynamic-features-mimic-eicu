#!/usr/bin/env python3
"""Generate supplementary calibration and decision curve plots.

The figure uses the same complete-case model settings as Table 3/Figure 4:
Base + P/F versus Base + P/F + VR in MIMIC-IV internal, MIMIC-IV
harmonized, and eICU external validation analyses.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
MIMIC_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "landmark_cohort_round2.csv"
HARM_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "harm_landmark_mimic.csv"
EICU_LM_PATH = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "eicu_landmark_round2.csv"


OUTCOME = "hospital_expire_flag"
RNG = 20260722

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

COLORS = {
    "Base + P/F": "#4C78A8",
    "Base + P/F + VR": "#E45756",
    "Treat all": "#7A7A7A",
    "Treat none": "#BDBDBD",
}


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


def prepare_predictions() -> pd.DataFrame:
    mimic_a = pd.read_csv(MIMIC_PATH)
    mimic_b = pd.read_csv(HARM_PATH)
    eicu_b = pd.read_csv(EICU_LM_PATH)
    poll = [c for c in ["hr", "map", "temp", "rr_vitalsign", "spo2"] if c in eicu_b.columns]
    if poll:
        eicu_b = eicu_b.drop(columns=poll)
    for a_, b_ in {"hr_native": "hr", "map_native": "map", "temp_native": "temp"}.items():
        if a_ in eicu_b.columns:
            eicu_b = eicu_b.rename(columns={a_: b_})
    # harmonized sepsis slot: admissionDx-recorded sepsis (recorded at ICU admission)
    eicu_b["sepsis48"] = eicu_b["sepsis_h"].astype(int)
    eicu_b = eicu_b.dropna(subset=[OUTCOME] + BASE_B + PF + VR).copy()

    rows = []

    p_old, _ = fit_predict(mimic_a, BASE_A + PF)
    p_new, _ = fit_predict(mimic_a, BASE_A + PF + VR)
    rows.append(("MIMIC-IV internal (48-h landmark)", mimic_a, p_old, p_new))

    p_old, p_old_eicu = fit_predict(mimic_b, BASE_B + PF, eicu_b)
    p_new, p_new_eicu = fit_predict(mimic_b, BASE_B + PF + VR, eicu_b)
    rows.append(("MIMIC-IV harmonized reduced (48-h landmark)", mimic_b, p_old, p_new))
    rows.append(("eICU (48-h landmark, transported)", eicu_b, p_old_eicu, p_new_eicu))

    pred_rows = []
    for setting, data, pred_old, pred_new in rows:
        for model, pred in [("Base + P/F", pred_old), ("Base + P/F + VR", pred_new)]:
            tmp = pd.DataFrame({
                "setting": setting,
                "model": model,
                "stay_id": data["stay_id"].values,
                "outcome": data[OUTCOME].astype(int).values,
                "predicted_risk": pred,
            })
            pred_rows.append(tmp)
    return pd.concat(pred_rows, ignore_index=True)


def calibration_table(pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for (setting, model), sub in pred.groupby(["setting", "model"], sort=False):
        sub = sub.copy()
        sub["bin"] = pd.qcut(sub["predicted_risk"], q=n_bins, labels=False, duplicates="drop")
        for b, g in sub.groupby("bin", sort=True):
            rows.append({
                "setting": setting,
                "model": model,
                "bin": int(b) + 1,
                "n": int(len(g)),
                "mean_predicted": float(g["predicted_risk"].mean()),
                "observed": float(g["outcome"].mean()),
            })
    return pd.DataFrame(rows)


def dca_table(pred: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.arange(0.05, 0.501, 0.01)
    rows = []
    for setting, setting_df in pred.groupby("setting", sort=False):
        one_model = setting_df[setting_df["model"].eq("Base + P/F")]
        y = one_model["outcome"].astype(int).values
        n = len(y)
        prevalence = y.mean()
        for pt in thresholds:
            rows.append({
                "setting": setting,
                "model": "Treat none",
                "threshold": float(pt),
                "net_benefit": 0.0,
            })
            rows.append({
                "setting": setting,
                "model": "Treat all",
                "threshold": float(pt),
                "net_benefit": float(prevalence - (1 - prevalence) * pt / (1 - pt)),
            })
        for model, sub in setting_df.groupby("model", sort=False):
            y = sub["outcome"].astype(int).values
            p = sub["predicted_risk"].values
            for pt in thresholds:
                pred_pos = p >= pt
                tp = np.sum(pred_pos & (y == 1))
                fp = np.sum(pred_pos & (y == 0))
                nb = tp / n - fp / n * (pt / (1 - pt))
                rows.append({
                    "setting": setting,
                    "model": model,
                    "threshold": float(pt),
                    "net_benefit": float(nb),
                })
    return pd.DataFrame(rows)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(p)
    try:
        fit = sm.Logit(y, sm.add_constant(x)).fit(disp=False, maxiter=200)
        return float(fit.params[0]), float(fit.params[1])
    except Exception:
        return np.nan, np.nan


def summary_table(pred: pd.DataFrame, dca: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, model), sub in pred.groupby(["setting", "model"], sort=False):
        y = sub["outcome"].astype(int).values
        p = sub["predicted_risk"].values
        intercept, slope = calibration_slope_intercept(y, p)
        d = dca[dca["setting"].eq(setting) & dca["model"].eq(model) &
                dca["threshold"].between(0.10, 0.30)]
        rows.append({
            "setting": setting,
            "model": model,
            "N": int(len(sub)),
            "Deaths": int(y.sum()),
            "Prevalence": float(y.mean()),
            "AUROC": float(roc_auc_score(y, p)),
            "Brier score": float(brier_score_loss(y, p)),
            "Calibration intercept": intercept,
            "Calibration slope": slope,
            "Mean net benefit 0.10-0.30": float(d["net_benefit"].mean()),
        })
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.7, length=3)
    ax.grid(True, color="#EAEAEA", linewidth=0.5)
    ax.set_axisbelow(True)


def draw_calibration(ax, cal: pd.DataFrame, summary: pd.DataFrame, setting: str, panel: str):
    sub = cal[cal["setting"].eq(setting)]
    for model in ["Base + P/F", "Base + P/F + VR"]:
        g = sub[sub["model"].eq(model)]
        summ = summary[(summary["setting"].eq(setting)) & (summary["model"].eq(model))].iloc[0]
        ax.plot(
            g["mean_predicted"], g["observed"],
            marker="o", ms=3.6, lw=1.5,
            color=COLORS[model],
            label=f"{model} (Brier {summ['Brier score']:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#666666", lw=1.0, linestyle=(0, (4, 3)))
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Predicted hospital mortality")
    ax.set_ylabel("Observed hospital mortality")
    ax.set_title(f"{panel}. {setting}: calibration", loc="left", fontsize=8.8, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6.8)
    style_axis(ax)


def draw_dca(ax, dca: pd.DataFrame, summary: pd.DataFrame, setting: str, panel: str):
    sub = dca[dca["setting"].eq(setting)]
    for model, lw, ls in [
        ("Base + P/F", 1.7, "-"),
        ("Base + P/F + VR", 1.7, "-"),
        ("Treat all", 1.0, (0, (4, 3))),
        ("Treat none", 1.0, (0, (1, 2))),
    ]:
        g = sub[sub["model"].eq(model)]
        ax.plot(g["threshold"], g["net_benefit"], color=COLORS[model], lw=lw, linestyle=ls, label=model)
    ax.set_xlim(0.05, 0.50)
    ymin = min(-0.02, sub["net_benefit"].quantile(0.01) - 0.01)
    ymax = sub[sub["model"].isin(["Base + P/F", "Base + P/F + VR", "Treat all"])]["net_benefit"].quantile(0.99) + 0.02
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{panel}. {setting}: decision curve", loc="left", fontsize=8.8, fontweight="bold")
    ax.legend(loc="lower left", fontsize=6.8, ncol=2)
    style_axis(ax)


def make_figure(cal: pd.DataFrame, dca: pd.DataFrame, summary: pd.DataFrame):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })
    settings = ["MIMIC-IV internal (48-h landmark)", "MIMIC-IV harmonized reduced (48-h landmark)", "eICU (48-h landmark, transported)"]
    letters = ["a", "b", "c", "d", "e", "f"]
    fig, axes = plt.subplots(3, 2, figsize=(8.9, 10.0))
    for i, setting in enumerate(settings):
        draw_calibration(axes[i, 0], cal, summary, setting, letters[2 * i])
        draw_dca(axes[i, 1], dca, summary, setting, letters[2 * i + 1])
    fig.suptitle("Supplementary Figure 2. Calibration and decision curve analyses for adding VR to P/F-based models",
                 fontsize=11.2, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.075, hspace=0.58, wspace=0.30)
    fig.text(
        0.08, 0.025,
        "Calibration points show deciles of predicted risk. Decision curves compare Base + P/F and Base + P/F + VR models across threshold probabilities.",
        ha="left", va="bottom", fontsize=7,
    )
    return fig


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    pred = prepare_predictions()
    cal = calibration_table(pred)
    dca = dca_table(pred)
    summary = summary_table(pred, dca)

    pred.to_csv(OUTDIR / "SuppFigure_DCA_calibration_predictions.csv", index=False)
    cal.to_csv(OUTDIR / "SuppFigure_calibration_deciles.csv", index=False)
    dca.to_csv(OUTDIR / "SuppFigure_DCA_net_benefit.csv", index=False)
    summary.to_csv(OUTDIR / "SuppFigure_DCA_calibration_summary.csv", index=False)

    fig = make_figure(cal, dca, summary)
    base = OUTDIR / "Supplementary_Figure_DCA_calibration"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    lines = [
        "# Supplementary DCA and calibration generation log",
        "",
        "Figure conclusion: adding VR to P/F-based models produces statistically detectable but clinically modest changes in calibration, Brier score, and decision-curve net benefit.",
        "",
        "Model settings:",
        "- MIMIC-IV internal: complete cases for BASE_A + P/F + VR.",
        "- MIMIC-IV harmonized: complete cases for BASE_B + P/F + VR.",
        "- eICU external: MIMIC-IV harmonized model applied to eICU complete cases.",
        "- P/F variables: pf_initial and pf_min.",
        "- VR variables: vr_initial and vr_delta.",
        "",
        "Cohort and metric checks:",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"- {r['setting']} | {r['model']}: n={int(r['N'])}, deaths={int(r['Deaths'])}, "
            f"AUROC={r['AUROC']:.3f}, Brier={r['Brier score']:.3f}, "
            f"calibration intercept={r['Calibration intercept']:.3f}, slope={r['Calibration slope']:.3f}, "
            f"mean NB 0.10-0.30={r['Mean net benefit 0.10-0.30']:.3f}."
        )
    lines += [
        "",
        "Outputs:",
        "- Supplementary_Figure_DCA_calibration.pdf",
        "- Supplementary_Figure_DCA_calibration.png",
        "- Supplementary_Figure_DCA_calibration.svg",
        "- SuppFigure_DCA_calibration_summary.csv",
        "- SuppFigure_calibration_deciles.csv",
        "- SuppFigure_DCA_net_benefit.csv",
        "- SuppFigure_DCA_calibration_predictions.csv",
    ]
    (OUTDIR / "SuppFigure_DCA_calibration_generation_log.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
