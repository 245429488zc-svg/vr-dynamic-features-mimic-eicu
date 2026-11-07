#!/usr/bin/env python3
"""Generate Spearman correlation matrix for early VR dynamic features."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path("/Volumes/tongtong/数据库/public/mimmc4/mimmc/项目2/审稿材料")
DATA = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "mimic_analysis_ready_round2.csv"
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))

FEATURES = [
    ("vr_initial", "VR initial"),
    ("vr_delta", "VR delta"),
    ("vr_mean", "VR mean"),
    ("vr_max", "VR maximum"),
    ("vr_min", "VR minimum"),
    ("vr_last", "VR last"),
    ("vr_auc", "VR AUC"),
    ("vr_slope", "VR slope"),
]


def p_fmt(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_correlation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cols = [c for c, _ in FEATURES]
    labels = [label for _, label in FEATURES]
    rho_mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    p_mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, (ci, li) in enumerate(FEATURES):
        for j, (cj, lj) in enumerate(FEATURES):
            rho, p = spearmanr(df[ci], df[cj], nan_policy="omit")
            n = int(df[[ci, cj]].dropna().shape[0])
            rho_mat.loc[li, lj] = rho
            p_mat.loc[li, lj] = p
            rows.append({
                "feature_1": li,
                "feature_2": lj,
                "spearman_rho": rho,
                "p_value": p,
                "p_value_formatted": p_fmt(p),
                "n": n,
            })
    long_df = pd.DataFrame(rows)
    return rho_mat, long_df


def make_figure(rho_mat: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    vals = rho_mat.values.astype(float)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(np.arange(len(rho_mat.columns)))
    ax.set_yticks(np.arange(len(rho_mat.index)))
    ax.set_xticklabels(rho_mat.columns, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(rho_mat.index)
    ax.tick_params(length=0)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            rho = vals[i, j]
            ax.text(
                j, i, f"{rho:.2f}",
                ha="center", va="center",
                fontsize=7.6,
                color="white" if abs(rho) >= 0.65 else "black",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Supplementary Figure 1. Correlations among early VR features",
        loc="left", fontsize=10.2, fontweight="bold", pad=14,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Spearman rho", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.text(
        0.08, 0.025,
        "MIMIC-IV main analysis cohort. Values are Spearman correlation coefficients.",
        ha="left", va="bottom", fontsize=7,
    )
    fig.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.20)
    return fig


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    if len(df) != 12694:
        raise ValueError(f"Expected n=12,694, got n={len(df):,}")
    deaths = int(pd.to_numeric(df["hospital_expire_flag"], errors="coerce").sum())
    if deaths != 2514:
        raise ValueError(f"Expected deaths n=2,514, got n={deaths:,}")
    missing = df[[c for c, _ in FEATURES]].isna().sum()
    if int(missing.sum()) != 0:
        raise ValueError(f"Missing VR features detected:\n{missing[missing > 0]}")

    rho_mat, long_df = build_correlation_tables(df)
    rho_mat.to_csv(OUTDIR / "SuppFigure_VR_feature_spearman_matrix.csv")
    long_df.to_csv(OUTDIR / "SuppFigure_VR_feature_spearman_long.csv", index=False)

    fig = make_figure(rho_mat)
    base = OUTDIR / "Supplementary_Figure_VR_feature_correlation"
    fig.savefig(base.with_suffix(".pdf"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mask = ~np.eye(len(rho_mat), dtype=bool)
    abs_rhos = np.abs(rho_mat.values.astype(float)[mask])
    high_pairs = long_df[
        (long_df["feature_1"] < long_df["feature_2"]) &
        (long_df["spearman_rho"].abs() >= 0.70)
    ].sort_values("spearman_rho", key=lambda x: x.abs(), ascending=False)
    lines = [
        "# VR dynamic feature correlation generation log",
        "",
        f"Data source: {DATA}",
        f"MIMIC-IV main analysis cohort: n={len(df):,}; deaths={deaths:,}.",
        "",
        "Features: " + ", ".join([label for _, label in FEATURES]) + ".",
        "Method: Spearman rank correlation, two-sided P values.",
        f"Maximum absolute off-diagonal rho: {float(abs_rhos.max()):.3f}.",
        f"Number of off-diagonal feature pairs with |rho| >= 0.70: {len(high_pairs)}.",
        "",
        "Interpretation: VR burden-related features are strongly intercorrelated and should not be interpreted as independent competing metrics or ranked by OR magnitude.",
        "",
        "Outputs:",
        "- Supplementary_Figure_VR_feature_correlation.pdf",
        "- Supplementary_Figure_VR_feature_correlation.png",
        "- Supplementary_Figure_VR_feature_correlation.svg",
        "- SuppFigure_VR_feature_spearman_matrix.csv",
        "- SuppFigure_VR_feature_spearman_long.csv",
    ]
    if len(high_pairs):
        lines += ["", "High-correlation pairs:"]
        for _, r in high_pairs.iterrows():
            lines.append(f"- {r['feature_1']} vs {r['feature_2']}: rho={r['spearman_rho']:.3f}, P {r['p_value_formatted']}.")
    (OUTDIR / "SuppFigure_VR_feature_correlation_generation_log.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
