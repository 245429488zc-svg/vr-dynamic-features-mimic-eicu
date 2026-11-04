#!/usr/bin/env python3
"""Round-2 rebuild (R2-3② + R2-2), eICU side.

1) Features: re-derive ALL trajectory features under Rule B (explicit truncation):
   window_end(h from unit admit) = min(48, ICU discharge offset, hospital
   discharge/death offset, terminal extubation = the last mechvent episode end in
   ventilation_events (cut only when the last recorded event is an end);
   ongoing ventilation (last event = start) is bounded by discharge only);
   pairs kept only if both the ABG and the MV charting fall inside the window.
   Gate: pf_points >=3 AND vr_points >=3 (unchanged).

2) Vitals (R2-2): hr / map / temperature recomputed NATIVELY as first-24 h MEANS from
   pivoted_vital nurse charting (replacing the apacheapsvar admission-extremes build
   flagged by the reviewer). BP = nibp_mean with ibp_mean fallback; temperature
   converted F->C when >= 55 (eICU known mixed-unit artifact).

3) Sepsis: the admission-diagnosis definition (admitdxname contains 'sepsis') is kept
   as the harmonized-family sepsis (sepsis_admdx here; MIMIC side gets the matching
   ICD-coded definition in the master pipeline). The old Sepsis-3-style column is
   retained for reference.

Output: eicu_analysis_ready_round2.csv + logs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Volumes/tongtong/论文/icuVR与死亡的关联")
EICU = Path("/Volumes/tongtong/数据库/public/mimmc4/mimmc/项目4/原始数据/eICU/eICU")
TRAJ = Path("/Volumes/tongtong/数据库/public/mimmc4/mimmc/项目2/outputs/pf_vr_dynamic_analysis/trajectory_eicu_48h.csv")
READY = ROOT / "1_队列数据" / "eICU_analysis_ready.csv"
OUTDIR = ROOT / "43_第二轮重分析"
OUTDIR.mkdir(exist_ok=True)
LOG: list[str] = []


def stamp(msg: str) -> None:
    LOG.append(msg)
    print(msg)


def features_explicit(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in points.groupby("stay_id"):
        sub = sub.sort_values("hour")
        vr = sub["vr"].astype(float).dropna()
        h_v = sub.loc[vr.index, "hour"].astype(float)
        span = float(h_v.iloc[-1] - h_v.iloc[0]) if len(vr) >= 2 else 0.0
        rec = {"stay_id": int(sid)}
        if len(vr) >= 1:
            rec["vr_initial"] = float(vr.iloc[0])
            rec["vr_last"] = float(vr.iloc[-1])
            rec["vr_mean"] = float(vr.mean())
            rec["vr_max"] = float(vr.max())
            rec["vr_min"] = float(vr.min())
        if len(vr) >= 2:
            rec["vr_delta"] = float(vr.iloc[-1] - vr.iloc[0])
            rec["vr_slope"] = float(rec["vr_delta"] / max(span, 1e-6))
            rec["vr_slope_ols"] = float(np.polyfit(h_v, vr, 1)[0])
            rec["vr_auc"] = float(np.trapz(vr.values, h_v.values) / max(span, 1e-6))
        pf = sub["pf"].astype(float).dropna()
        if len(pf) >= 1:
            rec["pf_initial"] = float(pf.iloc[0])
            rec["pf_min"] = float(pf.min())
            rec["pf_last"] = float(pf.iloc[-1])
            rec["pf_mean"] = float(pf.mean())
        if len(pf) >= 2:
            rec["pf_delta"] = float(pf.iloc[-1] - pf.iloc[0])
        pc = sub["pco2"].astype(float).dropna()
        ph = sub["ph"].astype(float).dropna()
        if len(pc) >= 1:
            rec["paco2_initial"] = float(pc.iloc[0])
        if len(ph) >= 1:
            rec["ph_initial"] = float(ph.iloc[0])
        rec["n_points"] = int(len(sub))
        rec["vr_points"] = int(len(vr))
        rec["pf_points"] = int(len(pf))
        rec["vr_time_span_h"] = span
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    traj = pd.read_csv(TRAJ)
    ready = pd.read_csv(READY)
    stamp(f"输入: eICU trajectory {len(traj)}点/{traj['stay_id'].nunique()}座; ready {len(ready)}行")

    # ---- offsets from official patient table ----
    pat = pd.read_csv(EICU / "patient.csv",
                      usecols=["patientunitstayid", "hospitaldischargeoffset", "hospitaldischargestatus"],
                      low_memory=False)
    pat = pat.rename(columns={"patientunitstayid": "stay_id"})
    det = pd.read_csv(EICU / "icustay_detail.csv", low_memory=False)
    det = det.rename(columns={"patientunitstayid": "stay_id"})
    unit_cols = [c for c in ["stay_id", "unitdischargeoffset", "unitdischargestatus", "apache_iv"] if c in det.columns]
    det_u = det[unit_cols].drop_duplicates("stay_id") if unit_cols else det[["stay_id"]].drop_duplicates("stay_id")

    tr = traj.merge(pat.rename(columns={"patientunitstayid": "stay_id"}), on="stay_id", how="left")
    tr = tr.merge(det_u, on="stay_id", how="left")

    # terminal invasive-ventilation boundary from ventilation_events (mechvent semantics):
    # last event = start  -> still ventilated, window bounded by discharge only;
    # last event = end    -> extubated at that time, window ends there.
    ev = pd.read_csv(EICU / "ventilation_events.csv", usecols=["patientunitstayid", "event", "hrs"],
                     low_memory=False)
    ev = ev.rename(columns={"patientunitstayid": "stay_id"})
    ev["hrs"] = pd.to_numeric(ev["hrs"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    e_start = ev[ev["event"].eq("mechvent start")].groupby("stay_id")["hrs"].max().rename("e_start")
    e_end = ev[ev["event"].eq("mechvent end")].groupby("stay_id")["hrs"].max().rename("e_end")
    tr = tr.merge(e_start.reset_index(), on="stay_id", how="left").merge(e_end.reset_index(), on="stay_id", how="left")
    vent_cut = np.where(
        tr["e_end"].notna() & (tr["e_end"].fillna(-1e9) >= tr["e_start"].fillna(-1e9)),
        tr["e_end"], np.inf)
    tr["vent_cut_h"] = vent_cut  # hours from unit admit (NOT minutes)
    tr["t_mv"] = tr["hour"] + pd.to_numeric(tr["pair_lag_hours"], errors="coerce")
    t_end = pd.to_numeric(tr["hospitaldischargeoffset"], errors="coerce")
    if "unitdischargeoffset" in tr.columns:
        unitd = pd.to_numeric(tr["unitdischargeoffset"], errors="coerce")
        t_end = t_end.mask(unitd.notna() & (unitd < t_end.fillna(np.inf)), unitd)
    t_end_h = t_end / 60.0
    t_end_h = t_end_h.mask(tr["vent_cut_h"].notna() & (tr["vent_cut_h"] < t_end_h.fillna(np.inf)),
                           tr["vent_cut_h"])
    tr["t_end_h"] = np.minimum(t_end_h.fillna(48.0), 48.0)
    keep = tr["t_mv"].notna() & (tr["hour"] <= tr["t_end_h"]) & (tr["t_mv"] <= tr["t_end_h"])
    removed = tr[~keep]
    decomp = {
        "abg超死亡/出科": int((removed["t_mv"].notna() & (removed["hour"] > removed["t_end_h"]))
                             .sum()),
        "mv越界(配对滞后越界)": int((removed["t_mv"].notna() & (removed["hour"] <= removed["t_end_h"]) &
                                 (removed["t_mv"] > removed["t_end_h"])).sum()),
    }
    stamp(f"点过滤(统一分钟→小时): {len(tr)} -> {int(keep.sum())} (删除 {int((~keep).sum())}; 分解: {decomp})")

    feat = features_explicit(tr[keep])
    feat = feat[(feat["vr_points"] >= 3) & (feat["pf_points"] >= 3)].copy()
    stamp(f"显式截断后 >=3配对座位: {len(feat)}（原口径 1,498 筛选/1,324 分析）")

    # ---- native first-24h vitals (R2-2 fix) ----
    stays = set(feat["stay_id"])
    chunks = []
    for chunk in pd.read_csv(EICU / "pivoted_vital.csv", usecols=[
            "patientunitstayid", "chartoffset", "heartrate", "nibp_mean", "ibp_mean", "temperature"],
            chunksize=1_000_000, low_memory=False):
        chunk = chunk[chunk["patientunitstayid"].isin(stays)]
        chunk = chunk[(chunk["chartoffset"] >= 0) & (chunk["chartoffset"] < 1440)]
        if chunk.empty:
            continue
        chunks.append(chunk)
    vit = pd.concat(chunks, ignore_index=True)
    stamp(f"pivoted_vital 首日行: {len(vit)} (覆盖 {vit['patientunitstayid'].nunique()} 座)")
    vit["t"] = pd.to_numeric(vit["temperature"], errors="coerce")
    vit.loc[vit["t"] >= 55, "t"] = (vit.loc[vit["t"] >= 55, "t"] - 32) * 5.0 / 9.0
    vit["bp"] = pd.to_numeric(vit["nibp_mean"], errors="coerce").fillna(pd.to_numeric(vit["ibp_mean"], errors="coerce"))
    agg = vit.groupby("patientunitstayid").agg(
        hr_native=("heartrate", "mean"), map_native=("bp", "mean"), temp_native=("t", "mean"),
        n_hr=("heartrate", "count"), n_bp=("bp", "count"), n_temp=("t", "count")).reset_index()
    agg = agg.rename(columns={"patientunitstayid": "stay_id"})
    stamp(f"原生均值覆盖: hr {int((agg['n_hr'] > 0).sum())}, map {int((agg['n_bp'] > 0).sum())}, "
          f"temp {int((agg['n_temp'] > 0).sum())} / {len(agg)}")

    # ---- assemble round-2 ready ----
    feat_out = feat.copy()
    feat_out = feat_out.merge(agg, on="stay_id", how="left")
    off = pat[["stay_id", "hospitaldischargestatus"]].copy()
    feat_out = feat_out.merge(off, on="stay_id", how="left")
    miss_off = int(feat_out["hospitaldischargestatus"].isna().sum())
    feat_out["hospital_expire_flag"] = (feat_out["hospitaldischargestatus"].fillna("") == "Expired").astype(float)
    ok_status = feat_out["hospitaldischargestatus"].isin(["Alive", "Expired"])
    stamp(f"官方结局覆盖: 缺失/异常状态 {int((~ok_status).sum())}/{len(feat_out)} 已排除; 院内死亡 "
          f"{int(feat_out.loc[ok_status, 'hospital_expire_flag'].sum())}/{int(ok_status.sum())} "
          f"({feat_out.loc[ok_status, 'hospital_expire_flag'].mean():.1%}) （旧口径 1,324/364=27.7%）")
    feat_out = feat_out[ok_status].copy()
    drop_old = ["n_points", "vr_initial", "vr_max", "vr_mean", "vr_last", "vr_min", "vr_auc",
                "vr_delta", "vr_slope", "vr_time_span_h", "pf_initial", "pf_min",
                "paco2_initial", "ph_initial", "hr", "map", "temp", "hospital_expire_flag"]
    old_rest = ready.drop(columns=[c for c in drop_old if c in ready.columns])
    new_ready = feat_out.merge(old_rest, on="stay_id", how="left")
    new_ready.to_csv(OUTDIR / "eicu_analysis_ready_round2.csv", index=False)
    stamp(f"已保存: eicu_analysis_ready_round2.csv ({len(new_ready)}行×{new_ready.shape[1]}列)")

    (OUTDIR / "eicu_rebuild_log.md").write_text(
        "# eICU 规则B特征重建＋原生首日均值日志（R2-3②/R2-2）\n\n"
        "- 截断：min(2880min, ICU出科offset, 院内出科/死亡offset, 末次MV charting)。\n"
        "- vitals：pivoted_vital 0-1440min 化为均值；BP=nibp_mean(缺失用ibp_mean)；温度F→C(>=55判华氏)。\n"
        "- labs/charlson/vent参数：沿用 labsfirstday 与 charlson/respiratorycharting 既有构建（定义已同构）。\n\n" + "\n".join(LOG),
        encoding="utf-8")
    stamp("完成")


if __name__ == "__main__":
    main()