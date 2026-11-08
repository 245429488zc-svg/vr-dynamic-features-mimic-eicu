# Master M1 日志

读入: MIMIC 12694, eICU 1420
MIMIC sepsis_h(ICD): 3185/12694 (25.1%); 对比 Sepsis-3 9609 (75.7%)
MIMIC sepsis48 (48-h Sepsis-3): 9058/12694 (71.4%)
eICU 协变量回填(来自公开builder, 旧值优先): 填补 3184 个空位
eICU 官方状态双检后: 1420, 死亡=396
eICU sepsis_h(admitDx): 247/1420 (17.4%)
eICU 生理变量已归一为原生首日均值列(hr/map/temp); 移除旧回填列: ['hr', 'map', 'temp', 'rr_vitalsign']
eICU harmonized 变量齐备: 19/19; 缺: []
eICU 完整病例(harmonized): 1374, 死亡=382
eICU landmark(48h时在院): n=1256, 48h后死亡=279
MIMIC landmark(带sepsis_h重取): n=9960, 死亡=1730
[MIMIC landmark FULL SPEC (internal)] Base: N=9960, 死亡=1730, AUROC=0.8222
[MIMIC landmark FULL SPEC (internal)] Base + P/F: N=9960, 死亡=1730, AUROC=0.8223
[MIMIC landmark FULL SPEC (internal)] Base + P/F + VR: N=9960, 死亡=1730, AUROC=0.8266
[MIMIC landmark HARMONIZED spec (fit for transport)] Base: N=9960, 死亡=1730, AUROC=0.7829
[MIMIC landmark HARMONIZED spec (fit for transport)] Base + P/F: N=9960, 死亡=1730, AUROC=0.7852
[MIMIC landmark HARMONIZED spec (fit for transport)] Base + P/F + VR: N=9960, 死亡=1730, AUROC=0.7911
[transported] Base: eICU AUROC=0.7139, 校准 -0.017/0.726（旧: 0.771/0.787, −0.27/0.47）
[transported] Base + P/F: eICU AUROC=0.7193, 校准 0.041/0.759（旧: 0.771/0.787, −0.27/0.47）
[transported] Base + P/F + VR: eICU AUROC=0.7293, 校准 0.115/0.802（旧: 0.771/0.787, −0.27/0.47）
[eICU landmark refit (replication)] Base: N=1256, 死亡=279, AUROC=0.7346
[eICU landmark refit (replication)] Base + P/F: N=1256, 死亡=279, AUROC=0.7416
[eICU landmark refit (replication)] Base + P/F + VR: N=1256, 死亡=279, AUROC=0.7510