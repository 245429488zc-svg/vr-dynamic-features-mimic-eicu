# Task A - MIMIC landmark ladder（重建后）

- 数据：mimic_analysis_ready_round2 的重建 landmark 队列
- Ladder：Base(18含copd/chf) → Base+P/F → Base+P/F+VR；5折分层CV（seed=20260721）

重建landmark队列: n=9960, 死亡=1730
Base: apparent 0.822, CV 0.819, optimism 0.0030, OOF校准 -0.023/0.980
Base + P/F: apparent 0.822, CV 0.819, optimism 0.0036, OOF校准 -0.026/0.978
Base + P/F + VR: apparent 0.827, CV 0.823, optimism 0.0037, OOF校准 -0.025/0.978
全模型中 vr_initial: OR 1.460 (1.332-1.601)
全模型中 vr_delta: OR 1.202 (1.109-1.303)