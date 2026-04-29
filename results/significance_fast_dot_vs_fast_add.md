# BLEU / chrF 配对 Bootstrap 显著性报告

## 设定

- **系统 A** (fast_dot): `runs/fast_dot/test_eval/predictions.jsonl`
- **系统 B** (fast_add): `runs/fast_add/test_eval/predictions.jsonl`
- **配对句数** N = 2500
- **Bootstrap 次数** B = 2000（同一套句子重采样下同时计算 BLEU 差与 chrF 差）。
- **RNG 种子** 42。
- **Δ** = score(A) − score(B)。

## BLEU（SacreBLEU corpus，与评估流水线一致）

| 指标 | 值 |
|------|-----|
| BLEU(A) | 16.0667 |
| BLEU(B) | 8.1633 |
| **Δ BLEU（观测）** | **+7.9034** |
| Δ 的 bootstrap 95% CI（百分位） | [7.4959, 8.3281] |
| p-value（近似，双侧启发式） | 0.0000 |

**区间与 0 无交：** A 相对 B 的 BLEU 优势在该 bootstrap 设定下较一致。

## chrF（word_order=0）

| 指标 | 值 |
|------|-----|
| chrF(A) | 37.6701 |
| chrF(B) | 26.1255 |
| **Δ chrF（观测）** | **+11.5446** |
| Δ 的 bootstrap 95% CI | [11.1288, 11.9670] |
| p-value（近似） | 0.0000 |

**区间与 0 无交：** chrF 差异方向较稳定。

## 汇总解读（显著 vs 随机波动）

- **更可能超越噪声**：至少一种度量下 **Δ 的 CI 与 0 无交集**，且效应方向一致（同为正或同为负）。当前： BLEU [区间不含 0]；chrF [区间不含 0]。
- **更像随机波动**：两度量 CI **均**跨过 0，或观测 Δ 很小而 CI 很宽。
- **p-value**：文内数值为常见 bootstrap 启发式，**请与 CI 联合判断**；正式推断应固定假设与多重比较校正（若多次两两对比）。

---

*输入须为 `evaluate_test.py` 生成的完整 `predictions.jsonl`（逐句 hypothesis/reference）。*
