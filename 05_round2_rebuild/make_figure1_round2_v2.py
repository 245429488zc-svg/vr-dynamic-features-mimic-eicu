#!/usr/bin/env python3
"""Generate Figure 1 study flow diagram.

Outputs a black-and-white dual-column flowchart for MIMIC-IV and eICU using
the revised manuscript cohort definitions.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(exist_ok=True)


MIMIC_STEPS = [
    {
        "main": "Adult first ICU stays from MIMIC-IV\n(n=85,242)",
        "exclude": "Excluded from total ICU stays (n=94,458):\nnon-adult or non-first ICU stays\n(n=9,216)",
    },
    {
        "main": "Early invasive mechanical ventilation\nwithin 24 h after ICU admission\n(n=28,387)",
        "exclude": "Excluded: no early invasive mechanical ventilation\nwithin 24 h after ICU admission\n(n=56,855)",
    },
    {
        "main": "At least 3 valid paired P/F-VR measurements\nwithin the truncated 48-h window\n(n=12,694)",
        "exclude": "Excluded: fewer than 3 valid pairs within the\nwindow; 569 stays lost eligibility after\nwindow-boundary truncation\n(n=15,693)",
    },
    {
        "main": "Final MIMIC-IV analysis cohort\n(vital status from official admissions records)\n(n=12,694; deaths=2,514)",
        "exclude": None,
    },
]

EICU_STEPS = [
    {
        "main": "Adult ICU stays\n(n=193,153)",
        "exclude": None,
    },
    {
        "main": "Early invasive mechanical ventilation\nwithin 24 h after ICU admission\n(n=4,141)",
        "exclude": "Excluded: no early invasive mechanical ventilation\nwithin 24 h after ICU admission\n(n=189,012)",
    },
    {
        "main": "At least 3 valid paired P/F-VR measurements\nwithin the truncated 48-h window\n(n=1,425)",
        "exclude": "Excluded: fewer than 3 valid pairs within\nthe truncated window\n(n=2,716)",
    },
    {
        "main": "Final eICU external validation cohort\n(discharge status Alive/Expired required)\n(n=1,420; deaths=396)",
        "exclude": "Excluded: discharge status not Alive/Expired\n(n=5)",
    },
]

ANALYSIS_SETS = [
    "Analysis sets",
    " MIMIC-IV:  full analysis / MICE cohort                    n=12,694; deaths=2,514",
    "            fully adjusted complete case                  n=10,428; deaths=2,178  (excluded 2,266, internal-model variables)",
    "            harmonized complete case (fit for transport)  n=11,251; deaths=2,315  (excluded 1,443, harmonized-model variables)",
    "            48-h landmark prediction cohort               n=9,960; deaths after 48 h = 1,730",
    " eICU:      association cohort (full)                     n=1,420; deaths=396",
    "            harmonized complete case / 48-h landmark      n=1,332 / 1,216; deaths=376 / 274",
    "            landmark complete case (transported)          n=1,206; deaths=269",
]


def add_box(ax, center_x: float, center_y: float, width: float, height: float, text: str,
            fontsize=9.5, lw=1.4, align="center", family=None, linespacing=1.15):
    x = center_x - width / 2
    y = center_y - height / 2
    rect = Rectangle((x, y), width, height, facecolor="white", edgecolor="black", linewidth=lw)
    ax.add_patch(rect)
    if align == "left":
        ax.text(x + 0.22, center_y, text, ha="left", va="center", fontsize=fontsize,
                linespacing=linespacing, family=family)
    else:
        ax.text(center_x, center_y, text, ha="center", va="center", fontsize=fontsize,
                linespacing=linespacing)
    return rect


def add_arrow(ax, start, end, mutation_scale=16, lw=1.25):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color="black",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def wrap_text(text: str, width: int) -> str:
    parts = []
    for line in text.split("\n"):
        parts.extend(textwrap.wrap(line, width=width) if line else [""])
    return "\n".join(parts)


def draw_column(ax, title: str, steps: list[dict], x_main: float, x_excl: float, y_top: float, y_gap: float):
    ax.text(x_main, y_top + 0.75, title, ha="center", va="center", fontsize=12, fontweight="bold")

    main_w = 3.45
    main_h = 0.72
    excl_w = 3.1
    excl_h = 0.78
    ys = [y_top - i * y_gap for i in range(len(steps))]

    for i, (step, y) in enumerate(zip(steps, ys)):
        add_box(ax, x_main, y, main_w, main_h, step["main"], fontsize=8.8)
        if i > 0:
            add_arrow(ax, (x_main, ys[i - 1] - main_h / 2), (x_main, y + main_h / 2), mutation_scale=17)
        if step["exclude"]:
            excl_text = wrap_text(step["exclude"], 42)
            add_arrow(ax, (x_main + main_w / 2, y), (x_excl - excl_w / 2, y), mutation_scale=16)
            add_box(ax, x_excl, y, excl_w, excl_h, excl_text, fontsize=7.8, lw=1.25)

    return ys[-1]


def main() -> None:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(16, 8.9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(8, 8.85, "Figure 1. Study flow diagram.", ha="center", va="center", fontsize=14, fontweight="bold")

    mimic_last_y = draw_column(ax, "MIMIC-IV", MIMIC_STEPS, x_main=2.45, x_excl=5.95, y_top=7.85, y_gap=1.32)
    eicu_last_y = draw_column(ax, "eICU", EICU_STEPS, x_main=9.55, x_excl=13.10, y_top=7.85, y_gap=1.32)

    bottom_y = 0.86
    bottom_h = 1.35
    bottom_w = 14.8
    analysis_text = "\n".join(ANALYSIS_SETS)
    add_box(ax, 8.0, bottom_y, bottom_w, bottom_h, analysis_text, fontsize=7.3, lw=1.4,
            align="left", family="monospace", linespacing=1.35)

    add_arrow(ax, (2.45, mimic_last_y - 0.36), (2.45, bottom_y + bottom_h / 2), mutation_scale=18)
    add_arrow(ax, (9.55, eicu_last_y - 0.36), (9.55, bottom_y + bottom_h / 2), mutation_scale=18)

    for ext in ("png", "pdf", "svg"):
        out = OUTDIR / f"Figure1_study_flow_diagram.{ext}"
        fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    counts_lines = [
        "database,step,n,excluded_n,excluded_reason",
        "MIMIC-IV,Adult first ICU stays from MIMIC-IV,85242,9216,Non-adult or non-first ICU stays from total ICU stays n=94458",
        "MIMIC-IV,Early invasive mechanical ventilation within 24 h,28387,56855,No early invasive mechanical ventilation within 24 h",
        "MIMIC-IV,At least 3 valid paired P/F-VR measurements within truncated 48-h window,12694,15693,Fewer than 3 valid pairs within the truncated window (569 eligibility losses from window-boundary truncation)",
        "MIMIC-IV,Final analysis cohort with official vital status,12694,0,Vital status available for all stays from official admissions records",
        "eICU,Adult ICU stays,193153,,",
        "eICU,Early invasive mechanical ventilation within 24 h,4141,189012,No early invasive mechanical ventilation within 24 h",
        "eICU,At least 3 valid paired P/F-VR measurements within truncated 48-h window,1425,2716,Fewer than 3 valid pairs within the truncated window",
        "eICU,Final external validation cohort with official discharge status,1420,5,Discharge status not Alive/Expired (Other/unknown, n=5)",
    ]
    (OUTDIR / "Figure1_cohort_flow_counts.csv").write_text("\n".join(counts_lines) + "\n", encoding="utf-8")

    log = [
        "# Figure 1 generation log",
        "",
        "Generated a black-and-white dual-column study flow diagram using the revised manuscript cohort counts.",
        "",
        "## Outputs",
        "",
        "- Figure1_study_flow_diagram.png",
        "- Figure1_study_flow_diagram.pdf",
        "- Figure1_study_flow_diagram.svg",
        "- Figure1_cohort_flow_counts.csv",
        "",
        "## Key notes",
        "",
        "- The old cohort_flow_combined.csv was not used because it reflects an earlier analysis version.",
        "- The >=2 paired-measurement intermediate step is not displayed in the figure to keep the MIMIC-IV and eICU flows aligned.",
        "- Final analysis cohorts are MIMIC-IV n=12,694 and eICU n=1,420 (round-2 rebuild).",
        "- Complete-case analysis sets are shown separately at the bottom of the figure.",
    ]
    (OUTDIR / "Figure1_generation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote Figure 1 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
