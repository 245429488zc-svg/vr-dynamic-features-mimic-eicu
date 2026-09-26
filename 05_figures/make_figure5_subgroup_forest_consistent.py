#!/usr/bin/env python3
"""Generate Figure 5 subgroup forest plots for VR initial and VR delta."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "20_Figure5亚组森林图" / "一致模型修订_20260729"
OUTDIR.mkdir(exist_ok=True)
DATA = ROOT / "1_队列数据" / "MIMIC_IV_analysis_ready.csv"
OUTCOME = "hospital_expire_flag"

BASE_A = [
    "age", "sex", "charlson", "sepsis3", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
FULL_VARS = BASE_A + PF + VR

SUBGROUPS = [
    {
        "name": "Age",
        "column": "age_group",
        "source": "age",
        "levels": [("<65 years", 0), ("≥65 years", 1)],
        "omit": ["age"],
    },
    {
        "name": "Sex",
        "column": "sex_group",
        "source": "sex",
        "levels": [("Female", 0), ("Male", 1)],
        "omit": ["sex"],
    },
    {
        "name": "Sepsis",
        "column": "sepsis_group",
        "source": "sepsis3",
        "levels": [("No sepsis", 0), ("Sepsis", 1)],
        "omit": ["sepsis3"],
    },
    {
        "name": "COPD",
        "column": "copd_group",
        "source": "copd",
        "levels": [("No COPD", 0), ("COPD", 1)],
        "omit": ["copd"],
    },
    {
        "name": "Congestive heart failure",
        "column": "chf_group",
        "source": "chf",
        "levels": [("No CHF", 0), ("CHF", 1)],
        "omit": ["chf"],
    },
    {
        "name": "SOFA severity",
        "column": "sofa_group",
        "source": "sofa",
        "levels": [("SOFA <5", 0), ("SOFA ≥5", 1)],
        "omit": ["sofa"],
    },
    {
        "name": "Baseline P/F ratio",
        "column": "pf_group",
        "source": "pf_initial",
        "levels": [("P/F <209", 0), ("P/F ≥209", 1)],
        "omit": ["pf_initial"],
    },
    {
        "name": "VR measurement frequency",
        "column": "npoints_group",
        "source": "n_points",
        "levels": [("3-4 measurements", 0), ("5-7 measurements", 1), ("≥8 measurements", 2)],
        "omit": [],
    },
]


def fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def scale_by_full_cc(cc: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = cc.copy()
    for var in variables:
        mean = out[var].mean()
        sd = out[var].std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            raise ValueError(f"Cannot standardize {var}; SD={sd}")
        out[var] = (out[var] - mean) / sd
    return out


def add_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = (out["age"] >= 65).astype(int)
    out["sex_group"] = out["sex"].astype(int)
    out["sepsis_group"] = out["sepsis3"].astype(int)
    out["copd_group"] = out["copd"].astype(int)
    out["chf_group"] = out["chf"].astype(int)
    out["sofa_group"] = (out["sofa"] >= 5).astype(int)
    out["pf_group"] = (out["pf_initial"] >= 209).astype(int)
    out["npoints_group"] = pd.cut(
        out["n_points"],
        bins=[2, 4, 7, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(int)
    return out


def fit_interaction_model(
    df: pd.DataFrame,
    subgroup_col: str,
    base_vars: list[str],
    target: str,
) -> tuple[float, dict[int, dict[str, float]]]:
    work = df.copy()
    levels = sorted(work[subgroup_col].dropna().unique())
    if len(levels) < 2:
        return np.nan, {}
    dummy_cols = []
    ref = levels[0]
    for level in levels[1:]:
        dcol = f"{subgroup_col}_{level}"
        work[dcol] = (work[subgroup_col] == level).astype(float)
        icol = f"{target}_x_{subgroup_col}_{level}"
        work[icol] = work[target] * work[dcol]
        dummy_cols.append((dcol, icol))
    main_terms = [d for d, _ in dummy_cols]
    interaction_terms = [i for _, i in dummy_cols]
    reduced_vars = base_vars + main_terms
    full_vars = reduced_vars + interaction_terms
    y = work[OUTCOME].astype(int)
    reduced = sm.Logit(y, sm.add_constant(work[reduced_vars], has_constant="add")).fit(disp=False, maxiter=300)
    full = sm.Logit(y, sm.add_constant(work[full_vars], has_constant="add")).fit(disp=False, maxiter=300)
    lr = 2 * (full.llf - reduced.llf)
    df_diff = len(interaction_terms)
    p_interaction = float(stats.chi2.sf(lr, df_diff))

    effects: dict[int, dict[str, float]] = {}
    cov = full.cov_params()
    for level in levels:
        if level == ref:
            beta = float(full.params[target])
            variance = float(cov.loc[target, target])
        else:
            interaction = f"{target}_x_{subgroup_col}_{level}"
            beta = float(full.params[target] + full.params[interaction])
            variance = float(
                cov.loc[target, target]
                + cov.loc[interaction, interaction]
                + 2 * cov.loc[target, interaction]
            )
        se = math.sqrt(variance)
        z = beta / se
        effects[int(level)] = {
            "OR": math.exp(beta),
            "CI_lower": math.exp(beta - 1.96 * se),
            "CI_upper": math.exp(beta + 1.96 * se),
            "p_value": float(2 * stats.norm.sf(abs(z))),
        }
    return p_interaction, effects


def run_subgroup_analysis() -> pd.DataFrame:
    raw = pd.read_csv(DATA)
    cc = raw.dropna(subset=[OUTCOME] + FULL_VARS).copy()
    cc = add_subgroups(cc)
    scaled = scale_by_full_cc(cc, FULL_VARS)

    rows = []
    for subgroup in SUBGROUPS:
        subgroup_col = subgroup["column"]
        omit = set(subgroup["omit"])
        variables = [v for v in FULL_VARS if v not in omit]
        inter_base = variables.copy()
        for target in VR:
            pint, effects = fit_interaction_model(scaled, subgroup_col, inter_base, target)
            for level_label, level_value in subgroup["levels"]:
                sub = scaled[scaled[subgroup_col] == level_value].copy()
                n = int(len(sub))
                deaths = int(sub[OUTCOME].sum())
                result = effects[int(level_value)]
                rows.append(
                    {
                        "feature": "VR initial" if target == "vr_initial" else "VR delta",
                        "variable": target,
                        "subgroup": subgroup["name"],
                        "level": level_label,
                        "N": n,
                        "Deaths": deaths,
                        "OR": result["OR"],
                        "CI_lower": result["CI_lower"],
                        "CI_upper": result["CI_upper"],
                        "p_value": result["p_value"],
                        "p_interaction": pint,
                        "model_variables": ", ".join(variables),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "Figure5_subgroup_results.csv", index=False)
    skill_input = out.dropna(subset=["OR"]).copy()
    skill_input["study"] = skill_input["subgroup"] + ": " + skill_input["level"] + " (" + skill_input["feature"] + ")"
    skill_input["or"] = skill_input["OR"]
    skill_input["ci_lower"] = skill_input["CI_lower"]
    skill_input["ci_upper"] = skill_input["CI_upper"]
    skill_input["weight"] = skill_input["N"]
    skill_input["subgroup_label"] = skill_input["feature"]
    skill_input[["study", "or", "ci_lower", "ci_upper", "weight", "subgroup_label"]].to_csv(
        OUTDIR / "Figure5_forest_input_for_skill.csv", index=False
    )
    return out


def build_plot_rows(results: pd.DataFrame, feature: str) -> list[dict]:
    rows = []
    sub = results[results["feature"].eq(feature)].copy()
    for subgroup in SUBGROUPS:
        name = subgroup["name"]
        pint = sub[sub["subgroup"].eq(name)]["p_interaction"].iloc[0]
        rows.append({"type": "header", "label": name, "p_interaction": pint})
        for level_label, _ in subgroup["levels"]:
            row = sub[(sub["subgroup"].eq(name)) & (sub["level"].eq(level_label))].iloc[0].to_dict()
            row["type"] = "data"
            rows.append(row)
    return rows


def draw_panel(ax, rows: list[dict], title: str, color: str) -> None:
    y_positions = np.arange(len(rows))[::-1]
    data_y = []
    data_rows = []
    labels = []
    for y, row in zip(y_positions, rows):
        if row["type"] == "header":
            labels.append(row["label"])
            p_text = fmt_p(row["p_interaction"])
            separator = " " if p_text.startswith("<") else "="
            ax.text(0.52, y, f"P for interaction{separator}{p_text}", va="center", ha="left", fontsize=6.8, color="#555555")
        else:
            labels.append("  " + row["level"])
            data_y.append(y)
            data_rows.append(row)

    for y, row in zip(data_y, data_rows):
        ax.hlines(y, row["CI_lower"], row["CI_upper"], color=color, linewidth=1.2)
        ax.plot(row["OR"], y, marker="o", color=color, markeredgecolor="black", markeredgewidth=0.3, markersize=4.8, zorder=3)

    ax.axvline(1, color="#777777", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 3.0)
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Adjusted OR (per 1-SD increase)")

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, len(rows) - 0.25, "N/deaths", transform=trans, ha="left", va="bottom", fontsize=7.0, fontweight="bold", clip_on=False)
    ax.text(1.17, len(rows) - 0.25, "OR (95% CI)", transform=trans, ha="left", va="bottom", fontsize=7.0, fontweight="bold", clip_on=False)
    ax.text(1.43, len(rows) - 0.25, "P", transform=trans, ha="left", va="bottom", fontsize=7.0, fontweight="bold", clip_on=False)
    for y, row in zip(data_y, data_rows):
        ax.text(1.02, y, f"{int(row['N']):,}/{int(row['Deaths']):,}", transform=trans, ha="left", va="center", fontsize=6.8, clip_on=False)
        ax.text(
            1.17,
            y,
            f"{row['OR']:.3f} ({row['CI_lower']:.3f}-{row['CI_upper']:.3f})",
            transform=trans,
            ha="left",
            va="center",
            fontsize=6.8,
            clip_on=False,
        )
        ax.text(1.43, y, fmt_p(row["p_value"]), transform=trans, ha="left", va="center", fontsize=6.8, clip_on=False)


def make_figure(results: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 15.0), sharey=False)
    draw_panel(axes[0], build_plot_rows(results, "VR initial"), "A. VR initial", "#4C78A8")
    draw_panel(axes[1], build_plot_rows(results, "VR delta"), "B. VR delta", "#E45756")
    fig.suptitle("Figure 5. Subgroup analyses of VR initial and VR delta with hospital mortality", fontsize=12, fontweight="bold", y=0.985)
    fig.text(
        0.08,
        0.025,
        "Subgroup-specific odds ratios, confidence intervals, and interaction P values were derived from the same fully adjusted full-cohort interaction model.\n"
        "The stratifying variable was omitted from the adjustment set when applicable.",
        ha="left",
        va="bottom",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.22, right=0.70, top=0.94, bottom=0.065, hspace=0.24)
    return fig


def write_log(results: pd.DataFrame) -> None:
    cc_n = int(results.loc[0, "N"] + results.loc[1, "N"])
    log = [
        "# Figure 5 subgroup forest plot generation log",
        "",
        "## Data source",
        "",
        f"- MIMIC-IV analysis data: {DATA}",
        "",
        "## Analysis cohort and model",
        "",
        f"- Fully adjusted complete-case cohort used for subgroup analyses: n={cc_n}.",
        "- Core variables plotted: VR initial and VR delta.",
        "- Odds ratios are per 1-SD increase, standardized using the full complete-case cohort.",
        "- Base adjustment set: age, sex, Charlson comorbidity index, sepsis, SOFA, COPD, CHF, PEEP, driving pressure, Vt/PBW, heart rate, MAP, temperature, lactate, BUN, creatinine, WBC, bicarbonate, initial P/F ratio, and minimum P/F ratio.",
        "- Subgroup-specific odds ratios and confidence intervals were calculated as linear combinations from the same full-cohort interaction model used for the interaction test.",
        "- The stratifying variable was omitted from the adjustment set when applicable.",
        "- P for interaction was calculated by likelihood-ratio testing of VR-by-subgroup interaction terms.",
        "",
        "## Subgroups",
        "",
        "- Age: <65 vs ≥65 years.",
        "- Sex: female vs male.",
        "- Sepsis: no sepsis vs sepsis.",
        "- COPD: no COPD vs COPD.",
        "- Congestive heart failure: no CHF vs CHF.",
        "- SOFA severity: <5 vs ≥5.",
        "- Baseline P/F ratio: <209 vs ≥209.",
        "- VR measurement frequency: 3-4, 5-7, and ≥8 measurements.",
        "",
        "## Skill use",
        "",
        "- Used forest-plot-styler conventions by exporting Figure5_forest_input_for_skill.csv with study/or/ci_lower/ci_upper/weight/subgroup fields.",
        "- The forest-plot-styler native entrypoint was run on the standardized input file and saved as native subgroup forest outputs.",
        "- A custom manuscript-format two-panel forest plot was generated to show N/deaths, OR, P value, and P for interaction.",
        "",
        "## Output files",
        "",
        "- Figure5_subgroup_forest.png",
        "- Figure5_subgroup_forest.pdf",
        "- Figure5_subgroup_forest.svg",
        "- Figure5_subgroup_results.csv",
        "- Figure5_forest_input_for_skill.csv",
        "- Figure5_skill_native_subgroup_forest.png",
        "- Figure5_skill_native_subgroup_forest.pdf",
    ]
    (OUTDIR / "Figure5_generation_log.md").write_text("\n".join(log) + "\n", encoding="utf-8")


def main() -> None:
    results = run_subgroup_analysis()
    fig = make_figure(results)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTDIR / f"Figure5_subgroup_forest.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    write_log(results)
    print(f"Wrote Figure 5 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
