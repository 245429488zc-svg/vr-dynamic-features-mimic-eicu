# Early ventilatory ratio (VR) features derived from repeated measurements and hospital mortality — analysis code

Code repository for the study **“Early ventilatory ratio features derived from repeated measurements and hospital mortality in mechanically ventilated patients: derivation and external validation”** (MIMIC-IV v3.1 derivation; eICU-CRD v2.0 external validation).

## Data access and placement

This repository contains **no patient data**. MIMIC-IV v3.1 and eICU-CRD v2.0 are available to credentialed users through PhysioNet (https://physionet.org) under the applicable data use agreements; the data may not be redistributed.

**No bespoke SQL is used.** Extraction operates on the official PhysioNet flat-table exports with pandas. Download/prepare the following tables and place them as shown; every script's input is either a PhysioNet table or an output produced by an earlier script in this repository, so the pipeline runs end-to-end:

| Placed at | Source |
|---|---|
| `raw_data/mimiciv_derived/mimiciv_derived/` | mimic-iv derived tables (`icustay_detail`, `ventilation`, `ventilator_setting`, `bg`, `first_day_height`, `first_day_weight`, `first_day_*`, `sofa`, `charlson`, `sepsis3`, `norepinephrine_equivalent_dose`, `rrt`) |
| `raw_data/core/icustays.csv`, `raw_data/hosp/admissions.csv`, `raw_data/hosp/diagnoses_icd.csv` | official MIMIC-IV core tables (window truncation, vital status, ICD-coded sepsis) |
| `raw_data/eICU/eICU/` | eICU-CRD tables (`icustay_detail`, `ventilation_events`, `pivoted_bg`, `respiratorycharting`, `pivoted_vital`, `labsfirstday`, `admissiondx`, `charlson`, `apacheapsvar`, `patient`) |
| `1_队列数据/` | rebuilt analysis-ready cohort files produced by `05_round2_rebuild/run_mimic_rebuild_round2.py` and `run_eicu_rebuild_round2.py` |

## Input/output map (run order)

1. `01_cohort_construction/build_trajectory_48h.py` — raw tables -> paired 0-48h point tables (**explicit truncation** at the earliest of death, terminal extubation, ICU discharge or hospital discharge; pairs kept only when both timestamps lie inside the window)
2. `02_covariates/enhanced_covariates_analysis.py` — raw tables -> admission/first-day covariates (vitals as first-day means on **both** databases; eICU recomputed natively from nurse charting, temperature F->C; sepsis by recorded diagnosis: ICD in MIMIC-IV, admissionDx text in eICU)
3. `05_round2_rebuild/run_mimic_rebuild_round2.py` + `run_eicu_rebuild_round2.py` -> rebuilt analysis-ready cohorts (MIMIC-IV n=12,694; eICU n=1,491)
4. `03_main_analyses/` — Table 2 association models; the incremental ladder is reported in the 48-h landmark framework (`05_round2_rebuild/run_landmark_incremental_mimic.py`), with the harmonized model fitted in MIMIC-IV and transported with fixed coefficients to eICU; Table 4 sensitivities
5. `04_revision_analyses/` — landmark, included-vs-excluded, anchor-point, bootstrap CI, MICE, internal validation (COPD/CHF included), narrow-window analyses
6. `05_round2_rebuild/` — second-round verification: truncation verification, DeLong tests, published-vs-rebuilt comparison, all tables/figures regenerated

## What changed in the second round (code-manuscript alignment)

- The internal-validation scripts include COPD and congestive heart failure; re-running reproduced all previously reported values (CV AUROC 0.840/0.846).
- The feature builder implements the stated truncation explicitly; 569 MIMIC-IV stays lost eligibility and 836 surviving members had trajectory features recomputed; initial-value features were unchanged.
- Vital status comes from the official source tables, recovering stays previously lost to legacy file-construction artifacts (110 MIMIC-IV, 172 eICU).
- The harmonized model includes COPD and CHF and uses sepsis defined by recorded diagnosis in both databases; eICU vitals are native first-day means.
- VR slope is defined as the two-point net rate of change, (last - first)/observation span, in the primary analysis and in the anchor-point analysis alike.

## Environment

Python 3.12; `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `python-docx`, `Pillow`.

## License

Code: MIT. Data: PhysioNet data use agreements (not included, not redistributable).
