import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


OUT = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
DATA = OUT / "mimic_analysis_ready_round2.csv"


np.random.seed(20260722)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
})

VR_FEATURES = [
    ("vr_initial", "VR initial"),
    ("vr_mean", "VR mean"),
    ("vr_min", "VR minimum"),
    ("vr_last", "VR last"),
    ("vr_delta", "VR delta"),
]

PF_FEATURES = [
    ("pf_initial", "P/F initial"),
    ("pf_mean", "P/F mean"),
    ("pf_min", "P/F minimum"),
    ("pf_last", "P/F last"),
    ("pf_delta", "P/F delta"),
]

SCATTER_PANELS = [
    ("pf_initial", "vr_initial", "P/F initial", "VR initial", "Initial status"),
    ("pf_mean", "vr_mean", "P/F mean", "VR mean", "Average burden"),
    ("pf_delta", "vr_delta", "P/F delta", "VR delta", "Early change"),
]

COLORS = {
    0: "#3B7CB1",
    1: "#F05A65",
}


def p_fmt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def spearman_table(df):
    rows = []
    for vr_col, vr_label in VR_FEATURES:
        for pf_col, pf_label in PF_FEATURES:
            rho, p = spearmanr(df[vr_col], df[pf_col], nan_policy="omit")
            n = df[[vr_col, pf_col]].dropna().shape[0]
            rows.append({
                "VR feature": vr_label,
                "P/F feature": pf_label,
                "Spearman rho": rho,
                "P value": p,
                "N": n,
            })
    return pd.DataFrame(rows)


def draw_heatmap(ax, corr_df):
    mat = corr_df.pivot(index="VR feature", columns="P/F feature", values="Spearman rho")
    mat = mat[[label for _, label in PF_FEATURES]]
    mat = mat.loc[[label for _, label in VR_FEATURES]]

    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(mat.index)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.45 else "black")
    ax.set_title("a. Spearman correlations between VR and P/F features",
                 loc="left", fontweight="bold", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman rho", fontsize=7)
    cbar.ax.tick_params(labelsize=7)


def clip_limits(s):
    lo, hi = np.nanpercentile(s, [1, 99])
    if np.isclose(lo, hi):
        lo, hi = np.nanmin(s), np.nanmax(s)
    return lo, hi


def draw_scatter(ax, df, x, y, xlabel, ylabel, subtitle, panel_letter, stats_df):
    plot = df[[x, y, "hospital_expire_flag"]].dropna().copy()
    xl = clip_limits(plot[x])
    yl = clip_limits(plot[y])
    plot = plot[(plot[x].between(*xl)) & (plot[y].between(*yl))]

    if len(plot) > 3500:
        sample = plot.sample(n=3500, random_state=20260722)
    else:
        sample = plot

    for grp, label, alpha in [(0, "Survived", 0.28), (1, "Deceased", 0.42)]:
        sub = sample[sample["hospital_expire_flag"] == grp]
        ax.scatter(sub[x], sub[y], s=7, c=COLORS[grp], alpha=alpha,
                   linewidths=0, label=label)

    overall = plot.sort_values(x)
    sm = lowess(overall[y], overall[x], frac=0.28, it=0, return_sorted=True)
    ax.plot(sm[:, 0], sm[:, 1], color="black", lw=1.4, label="LOESS")

    rho, p = spearmanr(df[x], df[y], nan_policy="omit")
    n = df[[x, y]].dropna().shape[0]
    ax.text(
        0.98, 0.96,
        f"rho={rho:.2f}\nP {p_fmt(p)}\nn={n:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#BDBDBD", lw=0.6, alpha=0.92),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.grid(True, color="#EAEAEA", linewidth=0.45)
    ax.set_title(f"{panel_letter}. {subtitle}", loc="left", fontweight="bold", fontsize=9)

    stats_df.append({
        "Panel": panel_letter,
        "Relation": f"{ylabel} vs {xlabel}",
        "Spearman rho": rho,
        "P value": p,
        "N": n,
        "X axis 1st percentile": xl[0],
        "X axis 99th percentile": xl[1],
        "Y axis 1st percentile": yl[0],
        "Y axis 99th percentile": yl[1],
        "Scatter displayed N": len(sample),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    if len(df) != 12694:
        raise ValueError(f"Expected MIMIC-IV main cohort n=12,694, got n={len(df):,}")
    deaths = int(df["hospital_expire_flag"].sum())
    if deaths != 2514:
        raise ValueError(f"Expected deaths n=2,514, got n={deaths:,}")

    corr_df = spearman_table(df)
    corr_df.to_csv(OUT / "SuppFigure_PF_VR_spearman_matrix_long.csv", index=False)

    fig = plt.figure(figsize=(9.2, 6.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.05, 1.0],
        hspace=0.56,
        wspace=0.38,
    )
    ax_heat = fig.add_subplot(gs[0, :])
    draw_heatmap(ax_heat, corr_df)

    scatter_stats = []
    for idx, (x, y, xlabel, ylabel, subtitle) in enumerate(SCATTER_PANELS):
        ax = fig.add_subplot(gs[1, idx])
        draw_scatter(ax, df, x, y, xlabel, ylabel, subtitle, chr(ord("b") + idx), scatter_stats)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:3], labels[:3], loc="lower left", fontsize=7,
                      handletextpad=0.4, borderaxespad=0.2)

    fig.text(
        0.01, 0.01,
        "MIMIC-IV main analysis cohort. Scatter points are a fixed random sample for display; Spearman correlations use all available observations.",
        fontsize=7,
        ha="left",
    )
    fig.suptitle(
        "Supplementary Figure 4. Correlations between P/F and VR features",
        x=0.01, y=0.995, ha="left", fontsize=11.5, fontweight="bold",
    )

    base = OUT / "Supplementary_Figure_PF_VR_correlation"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(scatter_stats).to_csv(OUT / "SuppFigure_PF_VR_scatter_stats.csv", index=False)

    log = OUT / "SuppFigure_PF_VR_correlation_generation_log.md"
    log.write_text(
        "\n".join([
            "# Supplementary Figure: P/F-VR correlation",
            "",
            f"Data source: {DATA}",
            f"MIMIC-IV main analysis cohort: n={len(df):,}; deaths={deaths:,}.",
            "",
            "Figure conclusion: VR dynamic features are related to P/F dynamic features but are not equivalent, supporting complementary physiological information.",
            "",
            "Panels:",
            "- a: Spearman correlation matrix between selected VR and P/F dynamic features.",
            "- b: VR initial versus P/F initial with LOESS trend.",
            "- c: VR mean versus P/F mean with LOESS trend.",
            "- d: VR delta versus P/F delta with LOESS trend.",
            "",
            "Statistics:",
            "- Spearman rank correlation, two-sided P values.",
            "- Correlations use all available observations.",
            "- Scatter points are clipped to the 1st-99th percentiles for readability and sampled with random seed 20260722 when needed.",
            "",
            "Outputs:",
            f"- {base.with_suffix('.pdf')}",
            f"- {base.with_suffix('.png')}",
            f"- {base.with_suffix('.svg')}",
            f"- {OUT / 'SuppFigure_PF_VR_spearman_matrix_long.csv'}",
            f"- {OUT / 'SuppFigure_PF_VR_scatter_stats.csv'}",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
