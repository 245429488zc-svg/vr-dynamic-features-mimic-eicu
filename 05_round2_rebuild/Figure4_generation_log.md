# Figure 4 generation log

## Data source

- MIMIC-IV analysis data: /Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code/outputs/round2_repro/landmark_cohort_round2.csv
- eICU analysis data: /Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code/outputs/round2_repro/eicu_landmark_round2.csv
- Table 3 reference: /Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code/outputs/round2_repro/Table3_round2.csv

## Model comparison

- Compared Base + P/F against Base + P/F + VR.
- P/F variables: pf_initial and pf_min.
- VR variables: vr_initial and vr_delta.
- MIMIC-IV internal model used BASE_A variables, including SOFA.
- MIMIC-IV harmonized and eICU external validation used BASE_B variables.

## Cohort checks

- MIMIC-IV internal (48-h landmark): n=9960, deaths=1730, AUROC 0.830->0.834, Delta=0.004, IDI=0.007, NRI=0.155.
- MIMIC-IV harmonized reduced (48-h landmark): n=10770, deaths=1856, AUROC 0.795->0.800, Delta=0.006, IDI=0.007, NRI=0.175.
- eICU (48-h landmark, transported): n=1256, deaths=279, AUROC 0.717->0.728, Delta=0.011, IDI=0.008, NRI=0.243.

## Validation against Table 3

- Recalculated values matched Table 3 within rounding tolerance.

## Output files

- Figure4_incremental_value_VR_beyond_PF.png
- Figure4_incremental_value_VR_beyond_PF.pdf
- Figure4_incremental_value_VR_beyond_PF.svg
- Figure4_ROC_curve_points.csv
- Figure4_incremental_metrics_summary.csv
