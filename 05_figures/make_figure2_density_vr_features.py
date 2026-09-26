#!/usr/bin/env python3
"""Generate an enhanced Figure 2 using density distributions of VR features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "17_Figure2高级版"
OUTDIR.mkdir(exist_ok=True)

MAIN_PATH = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
TRAJ_PATH = ROOT.parent / "outputs" / "pf_vr_dynamic_analysis" / "trajectory_mimic_48h.csv"

GROUP_ORDER = ["Survived", "Died"]
GROUP_COLORS = {"Survived": "#4C78A8", "Died": "#E45756"}
GROUP_LABELS = {"Survived": "Survived", "Died": "Deceased"}

CORE_FEATURES = [("vr_initial", "VR initial"), ("vr_delta", "VR delta")]
OTHER_FEATURES = [
    ("vr_mean", "VR mean"),
    ("vr_max", "VR maximum"),
    ("vr_min", "VR minimum"),
    ("vr_last", "VR last"),
    ("vr_auc", "VR AUC"),
]
PANEL_FEATURES = [
    ("vr_initial", "B. VR initial"),
    ("vr_delta", "C. VR delta"),
    ("vr_mean", "D. VR mean"),
    ("vr_max", "E. VR maximum"),
    ("vr_min", "F. VR minimum"),
    ("vr_last", "G. VR last"),
    ("vr_auc", "H. VR AUC"),
]
ALL_FEATURES = [(var, label.split(". ", 1)[1]) for var, label in PANEL_FEATURES]

TIME_BINS = [0, 6, 12, 24, 36, 48.000001]
TIME_LABELS = ["0-6", "6-12", "12-24", "24-36", "36-48"]
TIME_MIDPOINTS = [3, 9, 18, 30, 42]


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "P <0.001"
    return f"P = {p:.3f}"


def fmt_med_iqr(values: pd.Series) -> str:
    return f"{values.median():.2f} [{values.quantile(0.25):.2f}-{values.quantile(0.75):.2f}]"


def style_axis(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)
    if grid:
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MAIN_PATH)
    traj = pd.read_csv(TRAJ_PATH, usecols=["stay_id", "hour", "vr"])
    if len(df) != 13145:
        raise ValueError(f"Expected MIMIC-IV final cohort n=13,145, got n={len(df)}")
    if int(df["hospital_expire_flag"].sum()) != 2579:
        raise ValueError(f"Expected deaths=2,579, got deaths={int(df['hospital_expire_flag'].sum())}")
    feature_cols = [name for name, _ in ALL_FEATURES]
    missing = df[feature_cols].isna().sum()
    if missing.any():
        raise ValueError(f"Missing VR feature values detected:\n{missing[missing > 0]}")
    df["mortality_group"] = np.where(df["hospital_expire_flag"].eq(1), "Died", "Survived")
    traj = traj.merge(df[["stay_id", "mortality_group"]], on="stay_id", how="inner")
    if traj["stay_id"].nunique() != len(df):
        raise ValueError(f"Trajectory data matched {traj['stay_id'].nunique()} stays, expected {len(df)}")
    return df, traj


def trend_summary(traj: pd.DataFrame) -> pd.DataFrame:
    work = traj[(traj["hour"].between(0, 48)) & traj["vr"].notna()].copy()
    work["time_bin"] = pd.cut(
        work["hour"], bins=TIME_BINS, labels=TIME_LABELS, right=False, include_lowest=True
    )
    rows = []
    for group in GROUP_ORDER:
        for label, midpoint in zip(TIME_LABELS, TIME_MIDPOINTS):
            values = work.loc[
                work["mortality_group"].eq(group) & work["time_bin"].eq(label), "vr"
            ].dropna()
            rows.append(
                {
                    "group": group,
                    "time_bin": label,
                    "midpoint": midpoint,
                    "n_measurements": int(values.size),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, label in ALL_FEATURES:
        survived = df.loc[df["mortality_group"].eq("Survived"), var].dropna()
        died = df.loc[df["mortality_group"].eq("Died"), var].dropna()
        p = mannwhitneyu(survived, died, alternative="two-sided").pvalue
        for group, values in [("Survived", survived), ("Died", died)]:
            rows.append(
                {
                    "feature": label,
                    "variable": var,
                    "group": group,
                    "n": int(values.size),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def robust_xlim(values: pd.Series, low_q: float = 0.0025, high_q: float = 0.9975) -> tuple[float, float]:
    lo = float(values.quantile(low_q))
    hi = float(values.quantile(high_q))
    if lo == hi:
        lo -= 1
        hi += 1
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def draw_panel_a(ax, trend: pd.DataFrame) -> None:
    for group in GROUP_ORDER:
        sub = trend[trend["group"].eq(group)]
        x = sub["midpoint"].to_numpy(dtype=float)
        y = sub["median"].to_numpy(dtype=float)
        q1 = sub["q1"].to_numpy(dtype=float)
        q3 = sub["q3"].to_numpy(dtype=float)
        x_smooth = np.linspace(x.min(), x.max(), 180)
        y_smooth = PchipInterpolator(x, y)(x_smooth)
        q1_smooth = PchipInterpolator(x, q1)(x_smooth)
        q3_smooth = PchipInterpolator(x, q3)(x_smooth)
        ax.fill_between(x_smooth, q1_smooth, q3_smooth, color=GROUP_COLORS[group], alpha=0.13)
        ax.plot(
            x_smooth,
            y_smooth,
            color=GROUP_COLORS[group],
            linewidth=2.0,
            label=f"{GROUP_LABELS[group]}",
        )
        ax.scatter(x, y, color=GROUP_COLORS[group], s=18, zorder=3)
    ax.set_title("A. VR trajectory during the first 48 h", loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Time after ICU admission (h)")
    ax.set_ylabel("Ventilatory ratio")
    ax.set_xticks(TIME_MIDPOINTS)
    ax.set_xticklabels(TIME_LABELS)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    style_axis(ax, grid=True)


def draw_density_panel(
    ax,
    df: pd.DataFrame,
    var: str,
    label: str,
) -> None:
    vals = df[[var, "mortality_group"]].dropna()
    lo, hi = robust_xlim(vals[var])
    bin_count = 42
    for group in GROUP_ORDER:
        group_values = vals.loc[vals["mortality_group"].eq(group), var]
        sns.histplot(
            group_values,
            bins=bin_count,
            binrange=(lo, hi),
            stat="density",
            element="bars",
            fill=True,
            alpha=0.20,
            color=GROUP_COLORS[group],
            edgecolor=None,
            ax=ax,
        )
        sns.kdeplot(
            group_values.clip(lo, hi),
            color=GROUP_COLORS[group],
            linewidth=1.8,
            ax=ax,
            clip=(lo, hi),
            bw_adjust=1.28,
        )
        med = group_values.median()
        ax.axvline(med, color=GROUP_COLORS[group], linewidth=1.4, linestyle=(0, (4, 2)))

    survived = vals.loc[vals["mortality_group"].eq("Survived"), var]
    died = vals.loc[vals["mortality_group"].eq("Died"), var]
    p = mannwhitneyu(survived, died, alternative="two-sided").pvalue
    stat_text = (
        f"S: {fmt_med_iqr(survived)}\n"
        f"D: {fmt_med_iqr(died)}\n"
        f"{fmt_p(p)}"
    )
    ax.text(
        0.985,
        0.955,
        stat_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        linespacing=1.08,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cfcfcf",
            "boxstyle": "round,pad=0.16",
            "alpha": 0.86,
        },
    )
    axis_label = label.split(". ", 1)[1] if ". " in label else label
    ax.set_title(label, fontsize=9.4, fontweight="normal")
    ax.set_xlabel("")
    ax.set_ylabel("Density")
    ax.set_xlim(lo, hi)
    style_axis(ax, grid=False)


def draw_legend_axis(ax, df: pd.DataFrame) -> None:
    ax.axis("off")
    n_survived = int(df["mortality_group"].eq("Survived").sum())
    n_died = int(df["mortality_group"].eq("Died").sum())
    for i, group in enumerate(GROUP_ORDER):
        ax.add_patch(
            plt.Rectangle((0.05, 0.72 - i * 0.18), 0.08, 0.08, transform=ax.transAxes, color=GROUP_COLORS[group], alpha=0.45)
        )
        n = n_survived if group == "Survived" else n_died
        ax.text(0.16, 0.76 - i * 0.18, f"{GROUP_LABELS[group]} (n={n:,})", transform=ax.transAxes, va="center", fontsize=9)
    ax.text(
        0.05,
        0.30,
        "Dashed lines indicate group medians.\nPanels show smoothed density curves\nover density-scaled histograms.\nValues are median [IQR].\nP values use Mann-Whitney U tests.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.2,
        linespacing=1.35,
    )


def make_figure(df: pd.DataFrame, trend: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )
    sns.set_style("white")
    fig = plt.figure(figsize=(13.2, 9.2))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[0.95, 1.0, 1.0, 1.0], hspace=0.72, wspace=0.42)

    ax_a = fig.add_subplot(gs[0, :])
    draw_panel_a(ax_a, trend)

    for idx, (var, label) in enumerate(PANEL_FEATURES):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        draw_density_panel(ax, df, var, label)
    ax_legend = fig.add_subplot(gs[3, 1:])
    draw_legend_axis(ax_legend, df)

    fig.suptitle("Figure 2. Early VR features by hospital mortality", fontsize=13, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.06)
    return fig


def write_log(df: pd.DataFrame, trend: pd.DataFrame, summary: pd.DataFrame) -> None:
    summary.to_csv(OUTDIR / "Figure2_density_VR_feature_summary.csv", index=False)
    trend.to_csv(OUTDIR / "Figure2_density_VR_trajectory_summary.csv", index=False)
    log = [
        "# Figure 2 enhanced density plot generation log",
        "",
        "## Data source",
        "",
        f"- Main cohort: {MAIN_PATH}",
        f"- Trajectory data: {TRAJ_PATH}",
        "",
        "## Cohort check",
        "",
        f"- MIMIC-IV final cohort: n={len(df)}",
        f"- Survived: n={int(df['mortality_group'].eq('Survived').sum())}",
        f"- Deceased: n={int(df['mortality_group'].eq('Died').sum())}",
        "- All plotted VR summary features had 0 missing values.",
        "- Trajectory data matched all final MIMIC-IV stays.",
        "",
        "## Figure design",
        "",
        "- Panel A displays median and IQR of VR across 0-6, 6-12, 12-24, 24-36, and 36-48 h.",
        "- Panels B-H use an aligned three-column layout for VR initial, delta, mean, maximum, minimum, last, and AUC.",
        "- Density histograms are overlaid with KDE curves and group median reference lines.",
        "- Median [IQR] and P values are displayed in the upper-right corner of each density panel.",
        "",
        "## Statistical test",
        "",
        "- Between-group P values use two-sided Mann-Whitney U tests.",
        "- Summary values are median [IQR].",
    ]
    (OUTDIR / "Figure2_density_generation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    df, traj = load_data()
    trend = trend_summary(traj)
    summary = feature_summary(df)
    fig = make_figure(df, trend)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUTDIR / f"Figure2_early_VR_dynamic_features_density.{ext}",
            dpi=600 if ext == "png" else None,
            facecolor="white",
        )
    plt.close(fig)
    write_log(df, trend, summary)
    print(f"Wrote enhanced Figure 2 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
