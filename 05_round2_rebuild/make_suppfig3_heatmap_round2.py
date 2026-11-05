#!/usr/bin/env python3
"""Supplementary Figure 3: joint relationship of VR initial, VR delta, and
predicted hospital mortality (R1.9). Filled contours show model-based predicted
mortality across the initial x delta plane from the fully adjusted joint model
(other covariates at cohort means); thin density contours show observed
survivors and non-survivors."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
OUT = Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent))
OUT.mkdir(exist_ok=True)
OUTCOME = "hospital_expire_flag"
BASE_A = ["age", "sex", "charlson", "sepsis3", "copd", "chf", "sofa", "peep",
          "driving_pressure", "vt_pbw", "hr", "map", "temp", "lactate",
          "bun", "creatinine", "wbc", "bicarbonate"]
PF = ["pf_initial", "pf_min"]
VR = ["vr_initial", "vr_delta"]
VARS = BASE_A + PF + VR

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

df = pd.read_csv(Path(os.environ.get("ROUND2_OUT", Path(__file__).resolve().parent)) / "landmark_cohort_round2.csv")
cc = df.dropna(subset=[OUTCOME] + VARS).copy()
y = cc[OUTCOME].astype(int).values

scaler = StandardScaler().fit(cc[VARS])
X = scaler.transform(cc[VARS])
m = LogisticRegression(penalty=None, max_iter=3000, random_state=20260923)
m.fit(X, y)

# 网格（自然单位，观测1–99分位）
xi_lo, xi_hi = cc["vr_initial"].quantile([0.01, 0.99])
dx_lo, dx_hi = cc["vr_delta"].quantile([0.01, 0.99])
gx = np.linspace(xi_lo, xi_hi, 90)
gy = np.linspace(dx_lo, dx_hi, 90)
GX, GY = np.meshgrid(gx, gy)

# 简化：直接在标准化空间构造（VR两列用网格标准化值，其余列=0即协变量均值）
z_init = (GX - cc["vr_initial"].mean()) / cc["vr_initial"].std(ddof=0)
z_delta = (GY - cc["vr_delta"].mean()) / cc["vr_delta"].std(ddof=0)
Xg = np.zeros((GX.size, len(VARS)))
i_init, i_delta = VARS.index("vr_initial"), VARS.index("vr_delta")
Xg[:, i_init] = z_init.ravel()
Xg[:, i_delta] = z_delta.ravel()
pred = m.predict_proba(Xg)[:, 1].reshape(GX.shape)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
levels = np.linspace(0.10, 0.50, 9)
cf = ax.contourf(GX, GY, pred, levels=levels, cmap="RdBu_r", alpha=0.85)
cbar = fig.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label("Predicted hospital mortality", fontsize=10)
cs_line = ax.contour(GX, GY, pred, levels=levels, colors="k", linewidths=0.4, alpha=0.5)

from scipy.stats import gaussian_kde

for flag, color, label in [(0, "#3B6FB6", "Survived"), (1, "#C0392B", "Deceased")]:
    sub = cc[cc[OUTCOME] == flag]
    pts = np.vstack([sub["vr_initial"].values, sub["vr_delta"].values])
    if len(sub) > 3000:
        rng = np.random.default_rng(7)
        pts = pts[:, rng.choice(len(sub), 3000, replace=False)]
    kde = gaussian_kde(pts)
    dens_data = kde(pts)
    level = float(np.quantile(dens_data, 0.25))
    zz = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    ax.contour(GX, GY, zz, levels=[level], colors=color, linewidths=1.1, alpha=0.85)
    ax.plot([], [], color=color, lw=1.4, label=f"{label} density")
ax.set_xlabel("VR initial", fontsize=11)
ax.set_ylabel("VR delta (last − first)", fontsize=11)
ax.tick_params(labelsize=9)
ax.legend(frameon=False, fontsize=9, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "Supplementary_Figure_3.png", dpi=300)
fig.savefig(OUT / "Supplementary_Figure_3.pdf")
fig.savefig(OUT / "Supplementary_Figure_3.svg")
im = Image.open(OUT / "Supplementary_Figure_3.png").convert("RGB")
im.save(OUT / "Supplementary_Figure_3_300ppi_LZW.tif", compression="tiff_lzw", dpi=(300, 300))
print(f"图已输出到 {OUT}")
print(f"网格范围: vr_initial {xi_lo:.2f}-{xi_hi:.2f}, vr_delta {dx_lo:.2f}-{dx_hi:.2f}")
print(f"预测概率范围: {pred.min():.3f}-{pred.max():.3f}")