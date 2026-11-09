#!/usr/bin/env python3
"""Generate Figure 3 forest plot for Table 2 Model 3 VR associations."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(exist_ok=True)

TABLE2_PATH = Path(__file__).resolve().parent / "Table2_early_VR_mortality.csv"

FEATURE_LABELS = {
    "VR initial, per 1-SD increase": "VR initial",
    "VR mean, per 1-SD increase": "VR mean",
    "VR maximum, per 1-SD increase": "VR maximum",
    "VR minimum, per 1-SD increase": "VR minimum",
    "VR last, per 1-SD increase": "VR last",
    "VR AUC, per 1-SD increase": "VR AUC",
    "VR delta adjusted for VR initial, per 1-SD increase": "VR delta\n(adjusted for VR initial)",
    "VR slope, per 1-SD increase": "VR slope",
}

CORE_FEATURES = {"VR initial", "VR delta\n(adjusted for VR initial)"}


def parse_or_ci(value: str) -> tuple[float, float, float]:
    match = re.match(r"\s*([0-9.]+)\s*\(([0-9.]+)-([0-9.]+)\)\s*$", str(value))
    if not match:
        raise ValueError(f"Cannot parse OR value: {value}")
    return tuple(float(x) for x in match.groups())


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE2_PATH)
    rows = []
    for _, row in df.iterrows():
        feature = row["Early VR feature"]
        or_value, lower, upper = parse_or_ci(row["Model 3 OR (95% CI)"])
        label = FEATURE_LABELS.get(feature, feature)
        rows.append(
            {
                "study": label.replace("\n", " "),
                "feature": label,
                "or": or_value,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": row["Model 3 P value"],
                "n": int(row["Model 3 N"]),
                "or_ci": row["Model 3 OR (95% CI)"],
                "core": label in CORE_FEATURES,
            }
        )
    out = pd.DataFrame(rows)
    out["weight"] = 1.0
    out.to_csv(OUTDIR / "Figure3_forest_input_for_skill.csv", index=False)
    out.to_csv(OUTDIR / "Figure3_forest_model3_data.csv", index=False)
    return out


def make_forest_plot(df: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    plot_df = df.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = np.where(plot_df["core"], "#C44E52", "#4C4C4C")

    for i, row in plot_df.iterrows():
        ax.hlines(i, row["ci_lower"], row["ci_upper"], color=colors[i], linewidth=1.7 if row["core"] else 1.25)
        ax.plot(
            row["or"],
            i,
            marker="s" if row["core"] else "o",
            markersize=5.8 if row["core"] else 4.8,
            color=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.35,
            zorder=3,
        )

    ax.axvline(1, color="#7f7f7f", linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.85, 1.75)
    ax.set_xticks([0.9, 1.0, 1.2, 1.4, 1.6])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Adjusted odds ratio for hospital mortality (per 1-SD increase)")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_ylim(-0.7, len(plot_df) - 0.3)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)

    text_transform = blended_transform_factory(ax.transAxes, ax.transData)
    col_or = 1.025
    col_p = 1.305
    col_n = 1.505
    header_y = len(plot_df) - 0.1
    ax.text(col_or, header_y, "OR (95% CI)", transform=text_transform, ha="left", va="bottom", fontsize=8.5, fontweight="bold", clip_on=False)
    ax.text(col_p, header_y, "P value", transform=text_transform, ha="left", va="bottom", fontsize=8.5, fontweight="bold", clip_on=False)
    ax.text(col_n, header_y, "N", transform=text_transform, ha="left", va="bottom", fontsize=8.5, fontweight="bold", clip_on=False)
    for i, row in plot_df.iterrows():
        weight = "bold" if row["core"] else "normal"
        ax.text(col_or, i, row["or_ci"], transform=text_transform, ha="left", va="center", fontsize=8, fontweight=weight, clip_on=False)
        ax.text(col_p, i, row["p_value"], transform=text_transform, ha="left", va="center", fontsize=8, fontweight=weight, clip_on=False)
        ax.text(col_n, i, f"{int(row['n']):,}", transform=text_transform, ha="left", va="center", fontsize=8, fontweight=weight, clip_on=False)

    ax.text(
        0.0,
        -0.18,
        "Model 3 adjusted for demographics, comorbidities, severity score, ventilator parameters, vital signs, laboratory tests, and P/F indices.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.28, right=0.66, top=0.86, bottom=0.2)
    return fig


def write_log(df: pd.DataFrame) -> None:
    log = [
        "# Figure 3 forest plot generation log",
        "",
        "## Data source",
        "",
        f"- Table 2 source: {TABLE2_PATH}",
        "",
        "## Skill use",
        "",
        "- Used forest-plot-styler workflow conventions: standardized study/or/ci_lower/ci_upper input table and forest-plot output.",
        "- The packaged skill entrypoint was checked with --help and run on Figure3_forest_input_for_skill.csv.",
        "- A custom manuscript-format plot was also generated to add right-side OR, P value, and N columns.",
        "",
        "## Plotted data",
        "",
        f"- Number of VR dynamic features: {len(df)}",
        "- Estimates: Model 3 adjusted ORs from Table 2.",
        "- Core final-model features were emphasized: VR initial and VR delta adjusted for VR initial.",
        "",
        "## Output files",
        "",
        "- Figure3_VR_dynamic_features_forest.png",
        "- Figure3_VR_dynamic_features_forest.pdf",
        "- Figure3_VR_dynamic_features_forest.svg",
        "- Figure3_forest_input_for_skill.csv",
        "- Figure3_forest_model3_data.csv",
        "- Figure3_skill_native_forest.png",
        "- Figure3_skill_native_forest.pdf",
    ]
    (OUTDIR / "Figure3_generation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    df = prepare_data()
    fig = make_forest_plot(df)
    for ext in ("png", "pdf", "svg", "eps"):
        fig.savefig(
            OUTDIR / f"Figure3_VR_dynamic_features_forest.{ext}",
            dpi=600 if ext == "png" else None,
            facecolor="white",
        )
    plt.close(fig)
    write_log(df)
    print(f"Wrote Figure 3 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
