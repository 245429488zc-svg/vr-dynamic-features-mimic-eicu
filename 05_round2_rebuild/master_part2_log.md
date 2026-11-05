# Master Part A 日志

读入: MIMIC 12694, eICU 1420
[harm-cc MIMIC] n=11251, 死亡=2315（旧全队列harmonized: 11,647）
[harm-landmark MIMIC] n=10770, 48h后死亡=1856
[MIMIC harm·own] Base: N=10770, AUROC=0.7767, Brier=0.122
[MIMIC harm·own] Base + P/F: N=10770, AUROC=0.7779, Brier=0.121
[MIMIC harm·own] Base + P/F + VR: N=10770, AUROC=0.7838, Brier=0.120
[诊断] eICU-lm 协变量均值: hr=90.69, map=77.93, temp=36.84, bun=33.13, creatinine=1.89, wbc=16.80, bicarbonate=21.00
[诊断] MIMIC-harm-lm 协变量均值: hr=85.99, map=76.05, temp=36.87, bun=29.53, creatinine=1.70, wbc=17.22, bicarbonate=20.60
[eICU landmark] n=1206, 48h后死亡=269
[transported·own-cohort-coeffs] Base: AUROC=0.7217, 校准 0.511/0.765
[transported·own-cohort-coeffs] Base + P/F: AUROC=0.7266, 校准 0.556/0.795
[transported·own-cohort-coeffs] Base + P/F + VR: AUROC=0.7367, 校准 0.601/0.818
[eICU refit] Base: AUROC=0.7392, —
[eICU refit] Base + P/F: AUROC=0.7467, —
[eICU refit] Base + P/F + VR: AUROC=0.7556, VR initial 1.575 (1.229-2.018) P=0.000331; VR delta 1.384 (1.106-1.732) P=0.00449
[Table2 cc] n=10428, 死亡=2178
[Table2] vr_initial M1: OR 1.411 (1.350-1.474) P=0
[Table2] vr_initial M2: OR 1.416 (1.350-1.485) P=0
[Table2] vr_initial M3: OR 1.286 (1.209-1.369) P=1.78e-15
[Table2] vr_last M1: OR 1.441 (1.378-1.507) P=0
[Table2] vr_last M2: OR 1.460 (1.391-1.532) P=0
[Table2] vr_last M3: OR 1.366 (1.283-1.455) P=0
[Table2] vr_mean M1: OR 1.512 (1.445-1.581) P=0
[Table2] vr_mean M2: OR 1.554 (1.479-1.633) P=0
[Table2] vr_mean M3: OR 1.440 (1.344-1.542) P=0
[Table2] vr_max M1: OR 1.519 (1.453-1.588) P=0
[Table2] vr_max M2: OR 1.535 (1.462-1.611) P=0
[Table2] vr_max M3: OR 1.348 (1.263-1.439) P=0
[Table2] vr_min M1: OR 1.362 (1.302-1.424) P=0
[Table2] vr_min M2: OR 1.392 (1.325-1.461) P=0
[Table2] vr_min M3: OR 1.309 (1.228-1.396) P=2.22e-16
[Table2] vr_auc M1: OR 1.499 (1.433-1.567) P=0
[Table2] vr_auc M2: OR 1.540 (1.466-1.618) P=0
[Table2] vr_auc M3: OR 1.426 (1.332-1.527) P=0
[Table2] vr_delta M1: OR 0.936 (0.893-0.980) P=0.00496
[Table2] vr_delta M2: OR 1.358 (1.277-1.444) P=0
[Table2] vr_delta M3: OR 1.330 (1.234-1.434) P=1.18e-13
[Table2] vr_slope M1: OR 0.971 (0.927-1.019) P=0.231
[Table2] vr_slope M2: OR 0.973 (0.924-1.025) P=0.307
[Table2] vr_slope M3: OR 0.987 (0.927-1.051) P=0.693