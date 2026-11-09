# VR dynamic feature correlation generation log

Data source: /Volumes/tongtong/论文/icuVR与死亡的关联/42_GitHub仓库/vr-dynamic-features-code/outputs/round2_repro/mimic_analysis_ready_round2.csv
MIMIC-IV main analysis cohort: n=12,694; deaths=2,514.

Features: VR initial, VR delta, VR mean, VR maximum, VR minimum, VR last, VR AUC, VR slope.
Method: Spearman rank correlation, two-sided P values.
Maximum absolute off-diagonal rho: 0.983.
Number of off-diagonal feature pairs with |rho| >= 0.70: 13.

Interpretation: VR burden-related features are strongly intercorrelated and should not be interpreted as independent competing metrics or ranked by OR magnitude.

Outputs:
- Supplementary_Figure_VR_feature_correlation.pdf
- Supplementary_Figure_VR_feature_correlation.png
- Supplementary_Figure_VR_feature_correlation.svg
- SuppFigure_VR_feature_spearman_matrix.csv
- SuppFigure_VR_feature_spearman_long.csv

High-correlation pairs:
- VR AUC vs VR mean: rho=0.983, P <0.001.
- VR delta vs VR slope: rho=0.926, P <0.001.
- VR maximum vs VR mean: rho=0.910, P <0.001.
- VR AUC vs VR maximum: rho=0.877, P <0.001.
- VR AUC vs VR minimum: rho=0.841, P <0.001.
- VR mean vs VR minimum: rho=0.839, P <0.001.
- VR last vs VR mean: rho=0.811, P <0.001.
- VR AUC vs VR last: rho=0.805, P <0.001.
- VR initial vs VR mean: rho=0.777, P <0.001.
- VR initial vs VR maximum: rho=0.762, P <0.001.
- VR last vs VR minimum: rho=0.748, P <0.001.
- VR AUC vs VR initial: rho=0.717, P <0.001.
- VR last vs VR maximum: rho=0.703, P <0.001.