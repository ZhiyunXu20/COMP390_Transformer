# Run commit integrity（训练时 git 快照与核心代码一致性）

本文档依据各 **`runs/<run>/metrics.json`** 与 **`runs/<run>/resolved_config.json`** 中记录的 **`git_commit` / `git_dirty`**，用本地 **`git rev-parse <commit>:<path>`** 比对 **blob hash**，核对「跨 run 比较是否在字节一致的核心训练/评估代码上执行」。

- **生成时仓库 HEAD（仅供参考）**：`79af012d7f45862d042d4d58f2155369d02a6cad`（abbrev `79af012d7f45`）。
- **勿修改本文档引用的 JSON**：仅阅读 `metrics.json` / `resolved_config.json`；不改动任何 run 产物。

---

## Per-run recorded metadata

以下为 **`metrics.json`** 中的字段（训练脚本结束时写入）。**`resolved_config.json`** 在训练启动时写入；若二者 **`git_commit` 不一致**，在 **notes** 中说明。

| run | attention_type | git_commit (short) | git_dirty | pkg | notes |
|-----|----------------|---------------------|-----------|-----|-------|
| fast_add | additive | `2c605436fe` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_add_s1 | additive | `1ec4ccc52a` | `true` | small_try | **`metrics.json` 与 `resolved_config.json` 的 `git_commit` 不一致**（resolved=`97a3bfcb31`）；下列 blob 表以 **各文件实际采用的溯源** 为准时，应优先核对 **训练日志/起止时间**；此处将 **metrics** 视作训练结束快照 |
| fast_add_s2 | additive | `1ec4ccc52a` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_add_s3 | additive | `1ec4ccc52a` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_dot | dot_product | `2c605436fe` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_dot_s1 | dot_product | `97a3bfcb31` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_dot_s2 | dot_product | `97a3bfcb31` | `true` | small_try | 与 `resolved_config.json` 一致 |
| fast_dot_s3 | dot_product | `97a3bfcb31` | `true` | small_try | 与 `resolved_config.json` 一致 |
| head_1h_dot | dot_product | `2c605436fe` | `true` | small_head | 与 `resolved_config.json` 一致 |
| swap_fr_dot | dot_product | `2c605436fe` | `true` | small_swap | 与 `resolved_config.json` 一致 |
| var_bilinear | bilinear | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |
| var_entmax15 | entmax15 | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |
| var_gated_dot_additive | gated_dot_additive | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |
| var_global_local | global_local | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |
| var_local_window | local_window | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |
| var_sparsemax | sparsemax | `a750583efd` | `true` | small_try | 与 `resolved_config.json` 一致 |

**`pkg` 推断规则**：`head_*` → `small_head`；`swap_*` → `small_swap`；其余默认 `small_try`。

---

## Unique `git_commit` values (full SHA)

以下为 **`metrics.json`** 中出现的 **4** 个互异完整哈希（时间序：旧 → 新）：

| Order | Full SHA | Short | Present locally? (`git cat-file -e <sha>^{commit}`) |
|-------|----------|-------|-----------------------------------------------------|
| 1 | `2c605436fe6015ff15c1ec7166667e016b5a1f1f` | `2c605436fe` | OK |
| 2 | `97a3bfcb31d2decf902496e221d230b9b0954f26` | `97a3bfcb31` | OK |
| 3 | `1ec4ccc52aa6eaf53f1d0778bc566e9db41e574a` | `1ec4ccc52a` | OK |
| 4 | `a750583efd916d3a4ff8adb513ab18a5a4253f7a` | `a750583efd` | OK |

在本机仓库中 **四枚 commit 均存在**；若在某次 clone 中 `cat-file` 失败，应标注 **verification not possible (commit not in local history)** 并跳过下文 blob 比对。

---

## Core file blob hash matrix

对下列 **核心文件** 执行 **`git rev-parse <commit>:<path>`**；并与 **当前 HEAD** 的 blob 对比。表中 **单元格** 为 **abbrev 12 位**；后缀 **`(same as HEAD)`** 表示该 commit 下文件内容与 **生成本文档时的 HEAD** 字节一致。

**核心文件列表**（与任务书一致）：

- `small_try/attention.py`
- `small_try/model.py`
- `small_try/train.py`
- `small_head/train.py`
- `small_swap/train.py`
- `mt_eval.py`
- `evaluate_test.py`
- `scripts/significance_test.py`
- `scripts/ablation_lib.py`

| Core file | `2c605436fe` | `97a3bfcb31` | `1ec4ccc52a` | `a750583efd` | HEAD (`79af012d7f45`) |
|-----------|--------------|--------------|--------------|--------------|----------------------|
| `small_try/attention.py` | `19efa698384a` | `19efa698384a` | `caa2d2c3ce25` (same as HEAD) | `caa2d2c3ce25` (same as HEAD) | `caa2d2c3ce25` |
| `small_try/model.py` | `d34880a57f23` (same as HEAD) | `d34880a57f23` (same as HEAD) | `d34880a57f23` (same as HEAD) | `d34880a57f23` (same as HEAD) | `d34880a57f23` |
| `small_try/train.py` | `eb23856347e0` | `eb23856347e0` | `19c6b9250fea` (same as HEAD) | `19c6b9250fea` (same as HEAD) | `19c6b9250fea` |
| `small_head/train.py` | `3aa5fa4fdb49` | `3aa5fa4fdb49` | `ebdab4236b00` (same as HEAD) | `ebdab4236b00` (same as HEAD) | `ebdab4236b00` |
| `small_swap/train.py` | `191a566b8042` | `191a566b8042` | `7e2c9b998167` (same as HEAD) | `7e2c9b998167` (same as HEAD) | `7e2c9b998167` |
| `mt_eval.py` | `cbdb7c8b45f3` (same as HEAD) | `cbdb7c8b45f3` (same as HEAD) | `cbdb7c8b45f3` (same as HEAD) | `cbdb7c8b45f3` (same as HEAD) | `cbdb7c8b45f3` |
| `evaluate_test.py` | `cc04ac413e17` | `cc04ac413e17` | `0508056396ef` (same as HEAD) | `0508056396ef` (same as HEAD) | `0508056396ef` |
| `scripts/significance_test.py` | `da63ef127781` | `9ab823ee6b1e` (same as HEAD) | `9ab823ee6b1e` (same as HEAD) | `9ab823ee6b1e` (same as HEAD) | `9ab823ee6b1e` |
| `scripts/ablation_lib.py` | `9bf54d03bd0f` (same as HEAD) | `9bf54d03bd0f` (same as HEAD) | `9bf54d03bd0f` (same as HEAD) | `9bf54d03bd0f` (same as HEAD) | `9bf54d03bd0f` |

**读表要点**：**并非**所有 commit 在 **`small_try/attention.py`** 上与 HEAD 一致——`2c605436fe` 与 `97a3bfcb31` 共享 **较旧** blob `19efa698384a`；自 **`1ec4ccc52a`** 起与当前 HEAD 对齐。

---

## Diffs between oldest and newest training commit (summary only)

在 **`2c605436fe6015ff15c1ec7166667e016b5a1f1f`**（最早）与 **`a750583efd916d3a4ff8adb513ab18a5a4253f7a`**（最晚）之间，对 **blob 发生过变化** 的文件执行 **`git diff <old>..<new> -- <file>`**，**仅汇总统计**（不粘贴完整 diff）：

| File | Summary (`2c605436fe` → `a750583efd`) |
|------|----------------------------------------|
| `small_try/attention.py` | **1 file changed, 28 insertions(+), 6 deletions(-)** |
| `small_try/train.py` | **1 file changed, 19 insertions(+)** |
| `small_head/train.py` | **1 file changed, 47 insertions(+)** |
| `small_swap/train.py` | **1 file changed, 47 insertions(+)** |
| `evaluate_test.py` | **1 file changed, 65 insertions(+)** |
| `scripts/significance_test.py` | **1 file changed, 46 insertions(+), 11 deletions(-)**（跨 commit 的净变化；其中 **仅 `2c605436fe` 与后续 commit 不同**，见上表） |

`small_try/model.py`、`mt_eval.py`、`scripts/ablation_lib.py` 在上述四枚 commit 之间 **无变化**（与 HEAD blob 一致）。

---

## Conclusion

1. **是否 16 个 run 均在字节一致的核心代码上训练？**  
   **否。** 证据见上表：**`small_try/attention.py`**、**`small_try/train.py`**、**`small_head/train.py`**、**`small_swap/train.py`**、**`evaluate_test.py`** 在 **`2c605436fe` / `97a3bfcb31`** 与 **`1ec4ccc52a` / `a750583efd`** 之间 **blob 不同**；另外 **`scripts/significance_test.py`** 在 **`2c605436fe`** 单列与后续 commit 不一致。

2. **各 run 对应哪一版核心文件（简表）**  
   - **`2c605436fe`**（`fast_dot`, `fast_add`, `head_1h_dot`, `swap_fr_dot`）：**Epoch-1A** — 旧的 `attention.py` / 各包 `train.py` / `evaluate_test.py`；**且** 旧的 `significance_test.py`。  
   - **`97a3bfcb31`**（`fast_dot_s1`–`s3`）：**Epoch-1B** — 与 Epoch-1A **相同** 的 `attention.py` 与 `small_try/train.py` 等（同上旧 blob），但 **`significance_test.py` 已与 HEAD 对齐**。  
   - **`1ec4ccc52a`**（`fast_add_s1`–`s3` 在 **metrics** 中；**`fast_add_s1` 的 resolved 与 metrics 的 commit 不一致**，见上表 notes）：**Epoch-2** — 上述核心文件（除已全历史一致的 `model.py` / `mt_eval.py` / `ablation_lib.py`）与 **当前 HEAD** 对齐。  
   - **`a750583efd`**（全部 `var_*`）：与 Epoch-2 **同一组 blob**（与 HEAD 对齐）。

3. **哪些跨 run 比较相对安全？哪些必须加限定？**  
   - **较安全（同 commit / 同 Epoch blob 集）**：`fast_dot` vs `fast_add`（同 `2c605436fe`）；`fast_dot_s1`–`s3` 彼此（`97a3bfcb31`）；`fast_add_s2` vs `fast_add_s3`（同 `1ec4ccc52a`）；各 `var_*` 彼此（`a750583efd`）。  
   - **需强调代码版本漂移 caveat**：  
     - **`fast_dot` / `fast_add`（2c）** vs **`fast_add_s1`–`s3`（1ec4，metrics）**：核心实现 **非字节相同**（attention / train / evaluate_test 等）。统计结论若在论文中引用，应注明 **训练代码随仓库演进**。  
     - **`fast_dot`（2c）** vs **`fast_dot_s1`–`s3`（97a）**：**attention / small_try train 等仍与 2c 相同**；**`significance_test.py` 在 2c 与 97a 之间已变**——若某结论依赖该脚本在 **训练当期** 的版本，需交代。  
     - **任意 `var_*`（a750）** vs **主 EN→FR baseline（2c）**：核心训练栈已在 **`1ec4ccc52a`** 起更新后再训探索变体；与 2c 主 run **不是**同一字节快照，比较应限定为「同一研究项目下先后冻结快照」，而非「严格控制变量的单次代码版本实验」。

---

## `git_dirty` caveat

- **本 archive 中全部 16 个 run 的 `metrics.json` 均记录 `git_dirty: true`**：训练时工作区相对已提交快照 **存在未提交修改**。
- **这些未提交修改无法从 git 历史完整恢复**（没有对应 tree/blob），因此 **即使 commit 相同，磁盘上的真实训练代码也可能与 `git checkout <sha>` 不完全一致**。
- **论文建议**：在 **Threats to validity / Limitations** 中明确写出：**(i)** 跨 commit；**(ii)** `git_dirty=True`；**(iii)** 核心文件 blob 在部分 commit 间已变（见上表）。**推荐做法**：今后训练强制 **`git status` 干净** 或导出 **`git bundle` / `diff` 补丁** 随 run 归档。

---

## 重现本文档的核验命令（摘录）

```bash
# 某 commit 是否存在
git cat-file -e <full_sha>^{commit}

# 某 commit 下文件的 blob
git rev-parse <full_sha>:small_try/attention.py

# 与最早/最晚训练 commit 的统计 diff（不展开 patch）
git diff --stat 2c605436fe6015ff15c1ec7166667e016b5a1f1f..a750583efd916d3a4ff8adb513ab18a5a4253f7a -- small_try/attention.py
```
