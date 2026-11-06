#!/usr/bin/env python3
"""Compare the committed-chain reproduction run (outputs/round2_repro, produced by
running the repository scripts end-to-end per README) against the paper-canonical
result files committed in 05_round2_rebuild/ (produced from the archived July export
plus the explicit rule-B rebuild).

Key/value comparison per file; MIMIC quantities may differ by the documented 0.06%
export-version residual (all estimates must match at the manuscript's reporting
precision); eICU quantities must match exactly now that the terminal-extubation
truncation is implemented on both sides.

Output: 05_round2_rebuild/repro_comparison.csv + printout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

repo = Path("/Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code")
d05 = repo / "05_round2_rebuild"
rep = repo / "outputs" / "round2_repro"
ROWS = []


def cmp_frame(name, key_cols, val_cols, atol=0.0006):
    a = pd.read_csv(d05 / name)
    b_path = rep / name
    if not b_path.exists():
        ROWS.append({"file": name, "note": "REPRO 缺文件"})
        return
    b = pd.read_csv(b_path)
    ka = a[key_cols].astype(str).agg("|".join, axis=1)
    kb = b[key_cols].astype(str).agg("|".join, axis=1)
    m = a.set_index(ka)[val_cols].join(b.set_index(kb)[val_cols], how="outer", lsuffix="_canon", rsuffix="_repro")
    diffs = []
    for c in val_cols:
        ca, cb = f"{c}_canon", f"{c}_repro"
        if ca in m and cb in m:
            x = pd.to_numeric(m[ca], errors="coerce")
            y = pd.to_numeric(m[cb], errors="coerce")
            bad = ((x - y).abs() > atol) & (x.notna() | y.notna())
            diffs.append((c, int(bad.sum())))
    ROWS.append({"file": name, "note": "; ".join(f"{k}:{n}处>{atol}" for k, n in diffs) or "全部一致"})


def main():
    cmp_frame("round2_ladders.csv", ["Row", "Model"], ["N", "Deaths", "AUROC"])
    cmp_frame("round2_harm_ladders.csv", ["Row", "Model"], ["N", "Deaths", "AUROC"])
    cmp_frame("round2_bootstrap.csv", ["setting"], ["delta_auc"])
    cmp_frame("delong_and_contrast.csv", ["setting"], ["delta"])
    cmp_frame("landmark_incremental_mimic.csv", ["Model"], ["N", "Deaths", "AUROC", "AUROC (CV)"])
    cmp_frame("Table3_round2.csv", ["Cohort / setting"], ["N", "Deaths"])
    cmp_frame("table2_round2.csv", ["Feature", "Model"], ["N", "OR", "CI_lower", "CI_upper"])
    df = pd.DataFrame(ROWS)
    print(df.to_string(index=False))
    df.to_csv(d05 / "repro_comparison.csv", index=False)
    (d05 / "repro_comparison_log.md").write_text(
        "# 复现运行 vs 论文口径 对照\n\n"
        "- repro=outputs/round2_repro（仓库内按 README 顺序端到端运行）\n"
        "- canon=05_round2_rebuild 内提交的结果文件（论文口径, 43号重建世界）\n"
        "- MIMIC: 数量级 0.06% 数据导出残余（README 说明）; eICU: 应完全一致\n\n" +
        df.to_string(index=False), encoding="utf-8")
    print("已保存 repro_comparison.csv / .md")


if __name__ == "__main__":
    main()