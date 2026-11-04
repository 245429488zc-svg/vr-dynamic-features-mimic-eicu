# 05_round2_rebuild — Second-round revision (verification and reanalysis)

Scripts implementing the second-round reviewer requests (Reviewer 2, Comment 3; Reviewer 1 comments 1-3):

| Script | Purpose |
|---|---|
| run_truncation_verification.py | Verifies the truncation of the observation window: explicit rule vs published points; boundary decomposition |
| run_mimic_rebuild_round2.py / run_eicu_rebuild_round2.py | Re-derive all trajectory features under the explicit window rule; re-joint vital status from the official source tables |
| run_landmark_cv_internal.py / run_landmark_incremental_mimic.py | Prediction ladder (Base / Base+P/F / Base+P/F+VR) with 5-fold CV on the landmark cohorts |
| run_master_part1.py / run_master_part2.py | sepsis_h (recorded-diagnosis sepsis) in both databases; harmonized model fitted in MIMIC-IV and transported with fixed coefficients to eICU; eICU refit; Table 2 models |
| run_bootstrap_round2.py / run_delong_contrast.py | Bootstrap CIs (B=2000) for dAUROC/NRI/IDI and the DeLong tests, importing the published DeLong implementation |
| run_table3_docx.py / run_table4_round2.py | Rebuilt Table 3 (landmark ladder) and Table 4 (sensitivities) manuscripts tables |
| run_mice_round2.py | Multiple imputation (m=10) with the outcome in the imputation model, rebuilt cohort |
| run_anchor_round2.py / run_narrow_round2.py | Anchor-point and 1-h-window sensitivity analyses on the rebuilt cohort; slope definition unified to the two-point net rate of change |
| make_*_round2.py | All manuscript tables and figures regenerated from the rebuilt analysis files |

All rebuilt analysis files and final tables/figures are in the manuscript submission package, not in this repository.
