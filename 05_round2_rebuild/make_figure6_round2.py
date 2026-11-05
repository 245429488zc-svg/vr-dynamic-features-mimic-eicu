#!/usr/bin/env python3
"""Generate Figure 6 restricted cubic spline curves for VR initial and VR delta."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUTDIR.mkdir(exist_ok=True)

DATA = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "mimic_analysis_ready_round2.csv"
OUTCOME = "hospital_expire_flag"

BASE_A = [
    "age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
    "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
    "bun", "creatinine", "wbc", "bicarbonate",
]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
FULL_VARS = BASE_A + PF + VR

RCS_SPECS = [
    {
        "variable": "vr_initial",
        "label": "VR initial",
        "panel": "A. VR initial",
        "reference": "median",
        "adjust_extra": ["vr_delta"],
        "color": "#4C78A8",
    },
    {
        "variable": "vr_delta",
        "label": "VR delta",
        "panel": "B. VR delta",
        "reference": 0.0,
        "adjust_extra": ["vr_initial"],
        "color": "#E45756",
    },
]


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def scale_covariates(cc: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = cc.copy()
    for var in variables:
        mean = out[var].mean()
        sd = out[var].std(ddof=0)
        if sd and np.isfinite(sd):
            out[var] = (out[var] - mean) / sd
    return out


def get_complete_case() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    cc = df.dropna(subset=[OUTCOME] + FULL_VARS).copy()
    if len(cc) != 10428 or int(cc[OUTCOME].sum()) != 2178:
        raise ValueError(
            f"Unexpected complete-case cohort: n={len(cc)}, deaths={int(cc[OUTCOME].sum())}; expected n=10428, deaths=2178"
        )
    return cc


def likelihood_ratio_p(full_llf: float, reduced_llf: float, df_diff: int) -> float:
    lr = 2 * (full_llf - reduced_llf)
    return float(stats.chi2.sf(lr, df_diff))


def build_reference_frame(cc_raw: pd.DataFrame, spec: dict, grid: np.ndarray, ref_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = {}
    for var in FULL_VARS:
        if var == spec["variable"]:
            continue
        if cc_raw[var].dropna().isin([0, 1]).all():
            base[var] = float(cc_raw[var].mode().iloc[0])
        else:
            base[var] = float(cc_raw[var].median())
    pred = pd.DataFrame({spec["variable"]: grid})
    ref = pd.DataFrame({spec["variable"]: np.repeat(ref_value, len(grid))})
    for var, value in base.items():
        pred[var] = value
        ref[var] = value
    return pred, ref


def fit_rcs_for_variable(cc_raw: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    target = spec["variable"]
    raw_predictors = [v for v in BASE_A + PF + spec["adjust_extra"] if v != target]
    scale_vars = raw_predictors.copy()
    cc = scale_covariates(cc_raw, scale_vars)

    knots = np.quantile(cc[target], [0.05, 0.35, 0.65, 0.95])
    lower, upper = np.quantile(cc[target], [0.01, 0.99])
    grid = np.linspace(lower, upper, 220)
    if spec["reference"] == "median":
        ref_value = float(cc[target].median())
    else:
        ref_value = float(spec["reference"])

    spline_expr = f"cr({target}, knots=({knots[0]}, {knots[1]}, {knots[2]}, {knots[3]}))"
    cov_expr = " + ".join(raw_predictors)
    full_formula = f"{spline_expr} + {cov_expr}"
    linear_formula = f"{target} + {cov_expr}"
    reduced_formula = cov_expr

    y = cc[OUTCOME].astype(int)
    x_full = patsy.dmatrix(full_formula, cc, return_type="dataframe")
    x_linear = patsy.dmatrix(linear_formula, cc, return_type="dataframe")
    x_reduced = patsy.dmatrix(reduced_formula, cc, return_type="dataframe")

    full_res = sm.Logit(y, x_full).fit(disp=False, maxiter=300)
    linear_res = sm.Logit(y, x_linear).fit(disp=False, maxiter=300)
    reduced_res = sm.Logit(y, x_reduced).fit(disp=False, maxiter=300)
    p_overall = likelihood_ratio_p(full_res.llf, reduced_res.llf, x_full.shape[1] - x_reduced.shape[1])
    p_nonlinear = likelihood_ratio_p(full_res.llf, linear_res.llf, x_full.shape[1] - x_linear.shape[1])

    pred_raw, ref_raw = build_reference_frame(cc_raw, spec, grid, ref_value)
    for var in scale_vars:
        mean = cc_raw[var].mean()
        sd = cc_raw[var].std(ddof=0)
        if sd and np.isfinite(sd):
            pred_raw[var] = (pred_raw[var] - mean) / sd
            ref_raw[var] = (ref_raw[var] - mean) / sd
    x_pred = patsy.build_design_matrices([x_full.design_info], pred_raw, return_type="dataframe")[0]
    x_ref = patsy.build_design_matrices([x_full.design_info], ref_raw, return_type="dataframe")[0]
    contrast = np.asarray(x_pred - x_ref)
    beta = np.asarray(full_res.params)
    cov = np.asarray(full_res.cov_params())
    log_or = contrast @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", contrast, cov, contrast))
    out = pd.DataFrame(
        {
            "variable": target,
            "x": grid,
            "OR": np.exp(log_or),
            "CI_lower": np.exp(log_or - 1.96 * se),
            "CI_upper": np.exp(log_or + 1.96 * se),
            "reference": ref_value,
            "p_overall": p_overall,
            "p_nonlinearity": p_nonlinear,
            "knot_5": knots[0],
            "knot_35": knots[1],
            "knot_65": knots[2],
            "knot_95": knots[3],
        }
    )
    meta = {
        "variable": target,
        "label": spec["label"],
        "N": int(len(cc)),
        "Deaths": int(y.sum()),
        "reference": ref_value,
        "x_min": lower,
        "x_max": upper,
        "p_overall": p_overall,
        "p_nonlinearity": p_nonlinear,
        "knots": knots.tolist(),
        "adjusted_variables": raw_predictors,
    }
    return out, meta


def draw_panel(ax, curve: pd.DataFrame, meta: dict, spec: dict) -> None:
    color = spec["color"]
    ax.fill_between(curve["x"], curve["CI_lower"], curve["CI_upper"], color=color, alpha=0.18, linewidth=0)
    ax.plot(curve["x"], curve["OR"], color=color, linewidth=2.0)
    ax.axhline(1, color="#777777", linestyle=(0, (4, 3)), linewidth=0.9)
    ax.axvline(meta["reference"], color=color, linestyle=(0, (3, 2)), linewidth=1.0)
    ymax = min(max(float(curve["CI_upper"].quantile(0.98)) * 1.05, 2.0), 8.0)
    ax.set_ylim(0, ymax)
    ax.set_xlim(meta["x_min"], meta["x_max"])
    ax.set_title(spec["panel"], loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel(spec["label"])
    ax.set_ylabel("Adjusted OR for hospital mortality")
    ref_label = f"Reference={meta['reference']:.2f}"
    ax.text(
        0.04,
        0.96,
        f"{ref_label}\nP overall {fmt_p(meta['p_overall'])}\nP nonlinearity {fmt_p(meta['p_nonlinearity'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d8d8d8", "boxstyle": "round,pad=0.22", "alpha": 0.92},
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
    ax.set_axisbelow(True)


def make_figure(curves: pd.DataFrame, meta_df: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, spec in zip(axes, RCS_SPECS):
        curve = curves[curves["variable"].eq(spec["variable"])]
        meta = meta_df[meta_df["variable"].eq(spec["variable"])].iloc[0].to_dict()
        draw_panel(ax, curve, meta, spec)
    fig.suptitle("Figure 6. Restricted cubic spline analyses of core VR features", fontsize=12, fontweight="bold", y=0.98)
    fig.text(
        0.08,
        0.02,
        "Models were adjusted as in the fully adjusted MIMIC-IV complete-case model. Curves are shown across the 1st-99th percentile range.",
        ha="left",
        va="bottom",
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.2, wspace=0.28)
    return fig


def write_log(meta_df: pd.DataFrame) -> None:
    lines = [
        "# Figure 6 RCS generation log",
        "",
        "## Data source",
        "",
        f"- MIMIC-IV analysis data: {DATA}",
        "",
        "## Analysis cohort and model",
        "",
        "- Fully adjusted complete-case cohort: n=10,428; deaths=2,178.",
        "- RCS variables: VR initial and VR delta.",
        "- Knots used the 5th, 35th, 65th, and 95th percentiles of the target VR variable.",
        "- Curves are plotted across the 1st to 99th percentile range.",
        "- VR initial model adjusted for VR delta and the primary fully adjusted covariate set.",
        "- VR delta model adjusted for VR initial and the primary fully adjusted covariate set.",
        "- P overall compares the spline model with a model excluding the target VR variable.",
        "- P nonlinearity compares the spline model with a model containing a linear target VR term.",
        "",
        "## Results",
        "",
    ]
    for _, row in meta_df.iterrows():
        lines.append(
            f"- {row['label']}: reference={row['reference']:.3f}, range={row['x_min']:.3f}-{row['x_max']:.3f}, "
            f"P overall={fmt_p(row['p_overall'])}, P nonlinearity={fmt_p(row['p_nonlinearity'])}."
        )
    lines += [
        "",
        "## Output files",
        "",
        "- Figure6_RCS_VR_initial_delta.png",
        "- Figure6_RCS_VR_initial_delta.pdf",
        "- Figure6_RCS_VR_initial_delta.svg",
        "- Figure6_RCS_curve_data.csv",
        "- Figure6_RCS_model_summary.csv",
    ]
    (OUTDIR / "Figure6_generation_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cc = get_complete_case()
    curve_parts = []
    meta_rows = []
    for spec in RCS_SPECS:
        curve, meta = fit_rcs_for_variable(cc, spec)
        curve_parts.append(curve)
        meta_rows.append(meta)
    curves = pd.concat(curve_parts, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows)
    curves.to_csv(OUTDIR / "Figure6_RCS_curve_data.csv", index=False)
    meta_df.to_csv(OUTDIR / "Figure6_RCS_model_summary.csv", index=False)
    fig = make_figure(curves, meta_df)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTDIR / f"Figure6_RCS_VR_initial_delta.{ext}", dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    write_log(meta_df)
    print(f"Wrote Figure 6 outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
