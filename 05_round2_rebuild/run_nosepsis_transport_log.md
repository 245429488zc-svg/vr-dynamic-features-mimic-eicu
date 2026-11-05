# NO-sepsis transported sensitivity (Reviewer 2-2)

- Harmonized/transported 48-h landmark ladder WITHOUT the sepsis covariate in both databases.
- N2: MIMIC harm own cohort (stratified bootstrap refit, B=2,000); N3: fixed no-sepsis MIMIC coefficients + MIMIC-fitted scaler applied to eICU landmark (fixed-coeff bootstrap).

[N2] dAUROC 0.0064, DeLong P = 5.28e-05
[N2] MIMIC harm landmark, NO-sepsis spec (sepsis48 in cohort: 7697/10770 — removed from models)
[N2 ladder] Base: AUROC=0.7677
[N2 ladder] Base + P/F: AUROC=0.7694
[N2 ladder] Base + P/F + VR: AUROC=0.7758
[N2 MIMIC harm NO-sepsis ladder (VR step)] ΔAUROC 0.0064 (0.0037-0.0094); NRI 0.192 (0.134-0.248); events -0.027, non-events 0.219; IDI 0.009 (0.005-0.014)
[N3] eICU landmark n=1206, 48h后死亡=269
[N3 transported] Base: AUROC=0.7260, 校准 0.281/0.831
[N3 transported] Base + P/F: AUROC=0.7325, 校准 0.335/0.867
[N3 transported] Base + P/F + VR: AUROC=0.7406, 校准 0.348/0.871
[N3] dAUROC 0.0081, DeLong P = 0.177
[N3 eICU landmark NO-sepsis transported (fixed coeffs)] ΔAUROC 0.0080 (-0.0040-0.0204); NRI 0.241 (0.107-0.377); events 0.003, non-events 0.238; IDI 0.010 (0.004-0.017)
ST6 docx 完成