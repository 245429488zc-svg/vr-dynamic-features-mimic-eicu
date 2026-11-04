#!/usr/bin/env python3
"""Standardized anchor-point analysis (R1.3): one measurement per 12-h interval.

For each patient in the MIMIC-IV complete-case base, selects the paired P/F-VR
measurement closest to the midpoint of each 12-h interval of the 0-48 h window
(<= 4 anchors per patient), re-derives all eight VR features from these anchors,
and refits (a) the joint VR initial + VR delta fully adjusted model and
(b) each single-feature fully adjusted model. Comparisons against the published
primary estimates (Table 2 Model 3 / Table 4 primary row).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUT.mkdir(exist_ok=True)
TRAJ = ROOT / "outputs" / "pf_vr_dynamic_analysis" / "trajectory_mimic_48h.csv"
ARD = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "mimic_analysis_ready_round2.csv"

OUTCOME = "hospital_expire_flag"
BASE_A = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR_VARS = ["vr_initial", "vr_last", "vr_mean", "vr_max", "vr_min", "vr_auc",
           "vr_delta", "vr_slope"]
VR_LABELS = {
    "vr_initial": "VR initial", "vr_last": "VR last", "vr_mean": "VR mean",
    "vr_max": "VR maximum", "vr_min": "VR minimum", "vr_auc": "VR AUC",
    "vr_delta": "VR delta (last - first)", "vr_slope": "VR slope",
}
def load_primary_single():
    """Round-2 Table 2 Model-3 single-feature estimates, read live from the rebuilt results."""
    t2 = pd.read_csv(Path(__file__).resolve().parent / "table2_round2.csv")
    m3 = t2[t2["Model"].astype(str).eq("3")] if "Model" in t2.columns else t2
    out = {}
    for _, r in m3.iterrows():
        out[str(r["Feature"])] = f"{r['OR']:.3f} ({r['CI_lower']:.3f}-{r['CI_upper']:.3f})"
    return out


PRIMARY_OR = load_primary_single()
def load_primary_joint():
    """Round-2 primary joint estimates (Table 4's primary complete-case row), read live."""
    t4 = pd.read_csv(Path(__file__).resolve().parent / "Table4_round2.csv")
    r = t4[t4["Analysis"].astype(str).str.startswith("primary")].iloc[0]
    return {"vr_initial": str(r["VR initial OR (95% CI)"]),
            "vr_delta": str(r["VR delta OR (95% CI)"])}


PRIMARY_JOINT = load_primary_joint()
INTERVALS = [(0, 12), (12, 24), (24, 36), (36, 48)]
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def fmt_or(o: float, lo: float, hi: float) -> str:
    return f"{o:.3f} ({lo:.3f}-{hi:.3f})"


def fit_logit(cc: pd.DataFrame, variables: list[str], vr_vars: tuple[str, ...]) -> dict:
    x = pd.DataFrame(StandardScaler().fit_transform(cc[variables]), columns=variables, index=cc.index)
    x = sm.add_constant(x, has_constant="add")
    y = cc[OUTCOME].astype(int)
    res = sm.Logit(y, x).fit(disp=False, maxiter=300)
    pred = res.predict(x)
    rows = []
    for var in vr_vars:
        beta, se = float(res.params[var]), float(res.bse[var])
        z = beta / se
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        rows.append({"variable": var,
                     "OR_CI": fmt_or(math.exp(beta), math.exp(beta - 1.96 * se), math.exp(beta + 1.96 * se)),
                     "p": "<0.001" if p < 0.001 else f"{p:.3f}"})
    return {"N": len(cc), "Deaths": int(y.sum()),
            "AUROC": f"{roc_auc_score(y, pred):.3f}",
            "Brier": f"{brier_score_loss(y, pred):.3f}"}, pd.DataFrame(rows)


def main() -> None:
    pts = pd.read_csv(TRAJ)
    pts = pts[pts["vr"].between(0.2, 10) & pts["pf"].between(30, 700)].copy()
    stamp(f"逐点数据(有效VR): {len(pts)} 点 / {pts['stay_id'].nunique()} 人, hour {pts['hour'].min():.0f}-{pts['hour'].max():.0f}")

    ard = pd.read_csv(ARD)
    inc_ids = set(ard["stay_id"].astype(int))
    stamp(f"trajectory 覆盖 included: {len(inc_ids & set(pts['stay_id'].astype(int)))}/12,694")

    # ---- 锚点选择：每个12h区间取最接近区间中点的1个点
    anchors = []
    for sid, g in pts.groupby("stay_id", sort=False):
        g = g.sort_values("hour")
        for lo, hi in INTERVALS:
            mid = (lo + hi) / 2
            inside = g[(g["hour"] >= lo) & (g["hour"] <= hi)]
            if inside.empty:
                continue
            best = inside.iloc[(inside["hour"] - mid).abs().argmin()]
            anchors.append({"stay_id": sid, "interval": f"{lo}-{hi}", "hour": float(best["hour"]), "vr": float(best["vr"])})
    anch = pd.DataFrame(anchors)
    cnt = anch.groupby("stay_id").size()
    stamp(f"锚点选择完成: {len(anch)} 锚点 / {len(anch['stay_id'].unique())} 人; "
          f"锚点数分布: {dict(cnt.value_counts().sort_index())}")

    keep = set(cnt[cnt >= 2].index)
    anch2 = anch[anch["stay_id"].isin(keep)]

    # ---- 锚点特征重算
    rows = []
    for sid, g in anch2.sort_values(["stay_id", "hour"]).groupby("stay_id", sort=False):
        h = g["hour"].to_numpy(float)
        v = g["vr"].to_numpy(float)
        r = {"stay_id": sid, "n_anchors": len(g), "vr_initial": v[0], "vr_last": v[-1],
             "vr_delta": v[-1] - v[0], "vr_mean": v.mean(), "vr_max": v.max(), "vr_min": v.min()}
        span = h[-1] - h[0]
        r["vr_auc"] = float(np.trapz(v, h) / max(span, 1e-6))
        r["vr_slope"] = (v[-1] - v[0]) / max(span, 1e-6)  # two-point net slope (unified definition)
        rows.append(r)
    feat = pd.DataFrame(rows)
    med_anch = feat["n_anchors"].median()
    stamp(f"锚点特征重算: {len(feat)} 人; 每人锚点数中位数 = {med_anch:.0f}")

    # ---- 与完整病例基线对齐（丢弃主分析VR列，统一使用锚点重算版）
    cc = ard.dropna(subset=[OUTCOME] + BASE_A + PF).drop(columns=VR_VARS).copy()
    cc = cc.merge(feat, on="stay_id", how="inner")
    stamp(f"锚点分析队列(完整病例 ∩ >=2锚点): n={len(cc)}, 死亡={int(cc[OUTCOME].sum())}")

    # ---- joint 模型（VR initial + VR delta）
    joint_vars = BASE_A + PF + ["vr_initial", "vr_delta"]
    joint_out, joint_or = fit_logit(cc, joint_vars, ("vr_initial", "vr_delta"))
    stamp(f"[joint] n={joint_out['N']}, 死亡={joint_out['Deaths']}, AUROC={joint_out['AUROC']}, Brier={joint_out['Brier']}")
    for _, r in joint_or.iterrows():
        stamp(f"[joint] {r.variable}: {r.OR_CI}, P={r.p}")

    # ---- 单特征模型（Table 2 口径；delta 额外以 initial 为条件）
    single_rows = []
    for f in VR_VARS:
        if f == "vr_delta":
            variables, vr_vars = BASE_A + PF + ["vr_initial", "vr_delta"], ("vr_delta",)
        else:
            variables, vr_vars = BASE_A + PF + [f], (f,)
        out, or_df = fit_logit(cc, variables, vr_vars)
        row = or_df.iloc[0]
        single_rows.append({"variable": f, "label": VR_LABELS[f], "N": out["N"], "Deaths": out["Deaths"],
                            "OR_CI": row.OR_CI, "p": row.p, "AUROC": out["AUROC"]})
        stamp(f"[single] {VR_LABELS[f]}: {row.OR_CI}, P={row.p}, AUROC={out['AUROC']}")

    single_df = pd.DataFrame(single_rows)
    single_df["Primary OR (95% CI)"] = single_df["variable"].map(PRIMARY_OR)
    single_df.to_csv(OUT / "anchor_single_feature_OR.csv", index=False)
    joint_or.to_csv(OUT / "anchor_joint_OR_details.csv", index=False)
    feat.to_csv(OUT / "anchor_features_per_stay.csv", index=False)

    # ---- docx（与 Supp Table 3 同款式）
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Cm, Pt
    doc = Document()
    sec = doc.sections[0]
    w, h = sec.page_width, sec.page_height
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = h, w
    t0 = doc.add_paragraph()
    r0 = t0.add_run("Supplementary Table 3. Standardized anchor-point analysis using one measurement per 12-hour interval.")
    r0.font.name = "Times New Roman"; r0.font.size = Pt(9); r0.font.bold = True
    n1 = doc.add_paragraph()
    rr = n1.add_run(
        "To reduce the influence of informative measurement frequency, each patient contributed one standardized paired P/F-VR "
        "measurement closest to the midpoint of each 12-hour interval of the 48-hour window (maximum of four anchors per patient; "
        f"median anchors per patient = {med_anch:.0f}), and all VR features were re-derived from these anchors. Analyses used the "
        "complete-case cohort and the same fully adjusted models as the primary analyses; odds ratios are per 1-SD increase and are "
        "compared with the published primary estimates. VR delta was modeled conditional on VR initial.")
    rr.font.name = "Times New Roman"; rr.font.size = Pt(6.5)
    tbl = [["Variable", "N", "Deaths", "Anchor-point OR (95% CI)", "P value", "Primary OR (95% CI)"]]
    tbl.append([f"Joint model: VR initial (n = {joint_out['N']:,}; {joint_out['Deaths']} deaths after assessment)",
                joint_out["N"], joint_out["Deaths"],
                str(joint_or.loc[joint_or.variable.eq("vr_initial"), "OR_CI"].iloc[0]),
                str(joint_or.loc[joint_or.variable.eq("vr_initial"), "p"].iloc[0]),
                PRIMARY_JOINT["vr_initial"]])
    tbl.append(["Joint model: VR delta (conditional on VR initial)", joint_out["N"], joint_out["Deaths"],
                str(joint_or.loc[joint_or.variable.eq("vr_delta"), "OR_CI"].iloc[0]),
                str(joint_or.loc[joint_or.variable.eq("vr_delta"), "p"].iloc[0]),
                PRIMARY_JOINT["vr_delta"]])
    for _, r in single_df.iterrows():
        tbl.append([f"{r.label} (single-feature model)", r.N, r.Deaths, r.OR_CI, r.p, r["Primary OR (95% CI)"]])
    t = doc.add_table(rows=len(tbl), cols=len(tbl[0]))
    t.style = doc.styles["Table Grid"]
    widths = [Cm(9), Cm(2.2), Cm(2.2), Cm(4.6), Cm(2.2), Cm(4.6)]
    for i, row in enumerate(tbl):
        for j, val in enumerate(row):
            cell = t.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.name = "Times New Roman"; run.font.size = Pt(6.5)
            run.font.bold = (i == 0)
            cell.width = widths[j]
    doc.save(OUT / "Supplementary Table 3.docx")
    stamp(f"docx 已保存: {OUT / 'Supplementary Table 3.docx'}")
    (OUT / "anchor_analysis_log.md").write_text(
        "# 标准化锚点分析日志（R1.3）\n\n"
        "- 数据：22_研究归档 trajectory_mimic_48h.csv（0-48h 逐点配对 P/F-VR）。\n"
        "- 锚点：12h 区间 [0-12][12-24][24-36][36-48]，每区间取最接近区间中点的1次测量；每人最多4个锚点；锚点<2人剔除。\n"
        "- 基线：完整病例队列（与 Table 4 同基线），协变量与主分析一致。\n"
        "- 特征：锚点上重算 initial/last/delta/mean/max/min/AUC(trapz/跨度)/slope(锚点线性拟合)；标准化后 per 1-SD。\n"
        "- 对照列：published Table 2 Model 3 与 Table 4 primary 的 OR。\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa
    from sklearn.preprocessing import StandardScaler  # noqa
    main()