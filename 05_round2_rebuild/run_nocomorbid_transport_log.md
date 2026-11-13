# NO-comorbidity transported sensitivity (Reviewer 2 time-availability)

- Harmonized/transported 48-h landmark ladder WITHOUT the comorbidity covariates in both databases; sepsis and all remaining covariates are determinable by the landmark time.
- H2: MIMIC harm own cohort (stratified bootstrap refit, B=2,000); H3: fixed no-comorbidity MIMIC coefficients + MIMIC-fitted scaler applied to eICU landmark (fixed-coeff bootstrap).

[H2] dAUROC 0.0055, DeLong P = 0.000184
[H2 ladder] Base: AUROC=0.7595
[H2 ladder] Base + P/F: AUROC=0.7613
[H2 ladder] Base + P/F + VR: AUROC=0.7668
[H2 MIMIC harm NO-comorbidity ladder (VR step)] ΔAUROC 0.0056 (0.0030-0.0086); NRI 0.157 (0.101-0.208); events -0.048, non-events 0.204; IDI 0.008 (0.004-0.012)
[H3] eICU landmark n=1256, 48h后死亡=279
[H3 transported] Base: AUROC=0.7122, 校准 0.341/0.736
[H3 transported] Base + P/F: AUROC=0.7178, 校准 0.404/0.776
[H3 transported] Base + P/F + VR: AUROC=0.7287, 校准 0.457/0.803
[H3] dAUROC 0.0109, DeLong P = 0.0318
[H3 eICU landmark NO-comorbidity transported (fixed coeffs)] ΔAUROC 0.0108 (0.0011-0.0207); NRI 0.252 (0.118-0.384); events 0.053, non-events 0.199; IDI 0.008 (0.003-0.014)
ST7 docx 完成