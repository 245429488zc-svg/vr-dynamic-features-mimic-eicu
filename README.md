# Early ventilatory ratio (VR) features derived from repeated measurements and hospital mortality — analysis code

Code repository for the study **“Early ventilatory ratio features derived from repeated measurements and hospital mortality in mechanically ventilated patients: derivation and external validation”** (MIMIC-IV v3.1 derivation; eICU-CRD v2.0 external validation).

## Data access and placement

This repository contains **no patient data**. MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed users through PhysioNet (https://physionet.org) under the applicable data use agreements; the data may not be redistributed.

**No bespoke SQL is used.** Extraction operates on the official PhysioNet flat-table exports with pandas. Download/prepare the following tables and place them as shown; later sensitivity scripts also require their named intermediate outputs.

| Placed at | Source |
|---|---|
| `原始数据/mimiciv_derived/mimiciv_derived/` | mimic-iv derived tables (`icustay_detail`, `ventilation`, `ventilator_setting`, `bg`, `first_day_height`, `first_day_weight`, `first_day_*`, `sofa`, `charlson`, `sepsis3`, `norepinephrine_equivalent_dose`, `rrt`) |
| `原始数据/core/icustays.csv`, `原始数据/core/admissions.csv`, `原始数据/core/diagnoses_icd.csv` (flat CSV) | official MIMIC-IV core tables (window truncation, vital status, ICD-coded sepsis) |
| `原始数据/eICU/eICU/` | eICU-CRD tables (`icustay_detail`, `ventilation_events`, `pivoted_bg`, `respiratorycharting`, `pivoted_vital`, `labsfirstday`, `admissiondx`, `charlson`, `apacheapsvar`, `patient`) |

## Input/output map (run order)

1. `01_cohort_construction/build_outcome_vent7d.py` — raw tables -> per-stay outcome files `outputs/pf_vr_dynamic_analysis/{mimic,eicu}_outcome_vent7d.csv` (prerequisite of the trajectory builder; reads no downstream product)
2. `01_cohort_construction/build_trajectory_48h.py` — raw tables + step-1 outcomes -> paired 0-48h point tables (**explicit truncation** at the earliest of death, terminal extubation, ICU discharge or hospital discharge; pairs kept only when both timestamps lie inside the window)
3. `01_cohort_construction/build_pf_vr_analysis_data.py` — legacy feature builder (`outputs/pf_vr_dynamic_analysis/{mimic,eicu}_analysis_ready.csv`); retained for the pre-round-2 scripts in `03_main_analyses/`/`04_revision_analyses/` and **not** required by the round-2 chain (steps 4-7)
4. `02_covariates/enhanced_covariates_analysis.py` — raw tables -> admission/first-day covariates (vitals as first-day means on **both** databases; eICU recomputed natively from nurse charting, temperature F->C; sepsis flags: official Sepsis-3 in MIMIC-IV, admissionDx text in eICU)
5. `05_round2_rebuild/run_mimic_rebuild_round2.py` + `run_eicu_rebuild_round2.py` -> rebuilt analysis-ready cohorts (MIMIC-IV n=12,694; eICU n=1,420); the MIMIC rebuild additionally derives `sepsis3_48h` — the 48-hour-determinable Sepsis-3 flag (official Sepsis-3 with suspected-infection onset **and** qualifying SOFA assessment both ≤ 48 h) used by every 48-h landmark prediction model
6. `05_round2_rebuild/` ladder masters: `run_master_part1.py` (harmonized slot `sepsis48` = 48-h Sepsis-3 in MIMIC-IV / admissionDx-recorded sepsis in eICU; eICU landmark frames; harmonized fit + transported + eICU-refit ladders), `run_master_part2.py` (harmonized own-cohort ladder + transported + Table-2 association ORs), `run_landmark_incremental_mimic.py` (fully-adjusted landmark ladder + 5-fold CV), `run_bootstrap_round2.py` (ΔAUROC/IDI/NRI with B=2,000), `run_delong_contrast.py` (DeLong P values + corrected-spec association-framework continuity run)
7. `03_main_analyses/` + `04_revision_analyses/` — Table 2 association models, Table 4 sensitivities, anchor-point, included-vs-excluded, MICE, internal validation (COPD/CHF included), narrow-window analyses
8. `05_round2_rebuild/make_*` maker scripts — with the working directory set by `ROUND2_OUT` (e.g. `ROUND2_OUT=outputs/round2_repro`), every manuscript table and figure regenerates from the cohorts produced in steps 1-5 and the result files produced in steps 5-7; `05_round2_rebuild/` verification (truncation verification, published-vs-rebuilt comparison) closes the chain

## What changed in the second round (code-manuscript alignment)

- The internal-validation scripts include COPD and congestive heart failure; re-running reproduced all previously reported values (CV AUROC 0.840/0.846).
- The feature builder implements the stated truncation explicitly; 569 MIMIC-IV stays lost eligibility and 836 surviving members had trajectory features recomputed; initial-value features were unchanged.
- Vital status comes from the official source tables, recovering stays previously lost to legacy file-construction artifacts. The eICU window truncation additionally ends the observation window at the last recorded mechanical-ventilation (mechvent) episode end in ventilation_events; the eICU cohort under the complete rule is n=1,420 (deaths 396, 27.9%).
- Prediction (48-h landmark) models use sepsis flags determinable by the landmark time: `sepsis3_48h` in the fully adjusted internal model, and the harmonized/transported slot `sepsis48` (same MIMIC-IV definition in fitting; admissionDx-recorded sepsis in eICU). The two databases do not use identical sepsis case definitions. In the current eICU landmark cohort, all 194 admission-diagnosis sepsis positives have a recorded entry offset within 48 h. Under the previous any-time Sepsis-3 flag, 436 of the 9,960 MIMIC-IV landmark stays carried sepsis positivity not determinable by 48 h. Association-framework models retain the any-time Sepsis-3 flag with the declared concurrent-adjustment interpretation.
- The harmonized model includes COPD and CHF and uses sepsis48 as above; eICU vitals are native first-day means.
- VR slope is defined as the two-point net rate of change, (last - first)/observation span, in the primary analysis and in the anchor-point analysis alike.

## Reproduction note (Reviewer 2, Comment 3)

The earlier repository run (`outputs/round2_repro`) produced the following cohort counts. These counts alone do not verify the revised landmark prediction results:

| Quantity | Paper | Committed-pipeline run |
|---|---|---|
| MIMIC-IV: adult first ICU stays | 85,242 | 85,242 (exact) |
| MIMIC-IV: early invasive ventilation | 28,387 | 28,387 (exact) |
| MIMIC-IV: >=3 paired P/F-VR points in the truncated window | 12,694 | 12,694 (exact) |
| eICU: early invasive ventilation | 4,141 | 4,141 (exact) |
| eICU: >=3 paired P/F-VR points in the truncated window | 1,425 | 1,425 (exact) |

The comparison is recorded in `05_round2_rebuild/repro_comparison_log.md`. After the sepsis temporal-validity fix the reproduction chain was re-run inside the repository: all seven comparison files — including `Table3_round2.csv`, which had been missing in the earlier run — are now present, and every compared quantity matches the canonical result files (zero cells beyond tolerance).

Two boundary-level implementation defects were found and fixed while establishing this: the
window truncation now links each stay's hospital death/discharge through the official core
table (the hadm carried by chart rows can differ for multi-hospitalization subjects), and the
paired ventilator-setting timestamp is added in the same order as the published rebuild
(ABG time + pair lag) so the identical boundary set is retained on both sides.

## Environment

Python 3.12; `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `python-docx`, `Pillow`.

## License

Code: MIT. Data: PhysioNet data use agreements (not included, not redistributable).
