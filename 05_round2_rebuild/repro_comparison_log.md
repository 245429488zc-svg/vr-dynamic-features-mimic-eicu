# 复现运行 vs 论文口径 对照

- repro=outputs/round2_repro（仓库内按 README 顺序端到端运行）
- canon=05_round2_rebuild 内提交的结果文件（论文口径, 43号重建世界）
- MIMIC: 数量级 0.06% 数据导出残余（README 说明）; eICU: 应完全一致

                          file                                                                 note
            round2_ladders.csv                       N:0处>0.0006; Deaths:0处>0.0006; AUROC:0处>0.0006
       round2_harm_ladders.csv                       N:0处>0.0006; Deaths:0处>0.0006; AUROC:0处>0.0006
          round2_bootstrap.csv                                                  delta_auc:0处>0.0006
       delong_and_contrast.csv                                                      delta:0处>0.0006
landmark_incremental_mimic.csv N:0处>0.0006; Deaths:0处>0.0006; AUROC:0处>0.0006; AUROC (CV):0处>0.0006
             Table3_round2.csv                                        N:0处>0.0006; Deaths:0处>0.0006
             table2_round2.csv    N:0处>0.0006; OR:0处>0.0006; CI_lower:0处>0.0006; CI_upper:0处>0.0006