# Master M1 日志

读入: MIMIC 12694, eICU 1420
MIMIC sepsis_h(ICD): 3185/12694 (25.1%); 对比 Sepsis-3 9609 (75.7%)
MIMIC sepsis48 (48-h Sepsis-3): 9058/12694 (71.4%)
eICU 协变量回填(来自公开builder, 旧值优先): 填补 3184 个空位
eICU 官方状态双检后: 1420, 死亡=396
eICU sepsis_h(admitDx): 247/1420 (17.4%)
eICU harmonized 变量齐备: 19/19; 缺: []
eICU 完整病例(harmonized): 1332, 死亡=376
eICU landmark(48h时在院): n=1216, 48h后死亡=274
MIMIC landmark(带sepsis_h重取): n=9960, 死亡=1730
[MIMIC landmark FULL SPEC (internal)] Base: N=9960, 死亡=1730, AUROC=0.8222
[MIMIC landmark FULL SPEC (internal)] Base + P/F: N=9960, 死亡=1730, AUROC=0.8223
[MIMIC landmark FULL SPEC (internal)] Base + P/F + VR: N=9960, 死亡=1730, AUROC=0.8266
[MIMIC landmark HARMONIZED spec (fit for transport)] Base: N=9960, 死亡=1730, AUROC=0.7829
[MIMIC landmark HARMONIZED spec (fit for transport)] Base + P/F: N=9960, 死亡=1730, AUROC=0.7852
[MIMIC landmark HARMONIZED spec (fit for transport)] Base + P/F + VR: N=9960, 死亡=1730, AUROC=0.7911
[transported] Base: eICU AUROC=0.7001, 校准 -0.085/0.671（旧: 0.771/0.787, −0.27/0.47）
[transported] Base + P/F: eICU AUROC=0.7082, 校准 -0.022/0.707（旧: 0.771/0.787, −0.27/0.47）
[transported] Base + P/F + VR: eICU AUROC=0.7164, 校准 0.039/0.741（旧: 0.771/0.787, −0.27/0.47）
[eICU landmark refit (replication)] Base: N=1216, 死亡=274, AUROC=0.7189
[eICU landmark refit (replication)] Base + P/F: N=1216, 死亡=274, AUROC=0.7300
[eICU landmark refit (replication)] Base + P/F + VR: N=1216, 死亡=274, AUROC=0.7381