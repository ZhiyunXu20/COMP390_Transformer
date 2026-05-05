# AutoDL H800 — Repository health check

Read-only audit run on this clone (2026-05-05). No training, evaluation, or edits to tracked artifacts were performed as part of this check. A new dependency freeze was written to `environment_freeze_h800.txt` at the repository root per checklist. Artifact checker output was saved to `/tmp/artifact_check_before.log`.

---

## Repository state

| Item | Value |
|------|--------|
| **Current branch** | `fix/research-rigor` |
| **HEAD** | `79af012d7f45862d042d4d58f2155369d02a6cad` |
| **Working tree** | Clean (`git status --short` empty — no dirty / unstaged entries) |

**Recent history (`git log --oneline -10`):**

```
79af012 chore(runs): track untrained attention variant summaries
a750583 chore(results): refresh ablation tables; track fast_add_s1–s3 run summaries
1ec4ccc feat(research): attention variants, fairness audit, seed runs summaries
97a3bfc docs: remove stale conclusion/ references; document base_1 scope
5d170d6 Track baseline test_eval predictions and branch-local significance reports
3b44001 chore(runs): track per-run summary JSON/Markdown under runs/
ecd03b4 chore: post-pipeline eval, significance outputs, refreshed predictions/results
2c60543 feat(research): rigor updates — splits/tokenizers, eval tooling, small_try/swap/head
3427e55 chore: add push_github_main.sh (network_turbo + PAT from token.txt)
2733ad9 docs: clarify main push scope; ignore .cursor
```

**History depth / shallow clone**

- **`.git/shallow`**: absent → **not** a shallow clone (`git fetch --unshallow` not required for depth).
- **Spot-check commits** (full hashes as requested; all resolve on this clone):

| Prefix | Result |
|--------|--------|
| `2c605436fe` | Present → `2c60543 feat(research): rigor updates — splits/tokenizers, eval tooling, small_try/swap/head` |
| `1ec4ccc52a` | Present → `1ec4ccc feat(research): attention variants, fairness audit, seed runs summaries` |
| `97a3bfcb31` | Present → `97a3bfc docs: remove stale conclusion/ references; document base_1 scope` |
| `a750583efd` | Present → `a750583 chore(results): refresh ablation tables; track fast_add_s1–s3 run summaries` |

- **Commit counts**: `git rev-list --count HEAD` = **11** (ancestry of current branch tip); `git rev-list --all --count` = **40** (all refs).

---

## Python and GPU environment

| Item | Value |
|------|--------|
| **Python** | 3.12.3 |
| **pip** | 24.0 (`/root/miniconda3/lib/python3.12/site-packages/pip`) |
| **GPU model** | NVIDIA H800 PCIe |
| **Driver** | 580.82.07 |
| **CUDA (reported by SMI)** | 13.0 |
| **Total VRAM** | 81559 MiB (~79.6 GiB) |
| **GPU utilization at check** | 0%; no processes |

**Dependency freeze:** `python -m pip freeze > environment_freeze_h800.txt` (202 lines).

---

## pytest result

Command: `python -m pytest -q`

| Outcome | Count |
|---------|--------|
| **Passed** | 18 |
| **Failed** | 0 |
| **Errors** | 0 |
| **Duration** | ~1.63 s |

---

## Data and tokenizer files

| Path | Status |
|------|--------|
| `data/splits/en_fr_50k_seed42/train.tsv` | [present] |
| `data/splits/en_fr_50k_seed42/val.tsv` | [present] |
| `data/splits/en_fr_50k_seed42/test.tsv` | [present] |
| `data/splits/en_fr_50k_seed42/split_metadata.json` | [present] |
| `data/tokenizer_src_train_only.json` | [present] |
| `data/tokenizer_tgt_train_only.json` | [present] |
| `data/tokenizer_metadata.json` | [present] |

---

## Runs inventory

**Top-level run directories (18):** `fast_add`, `fast_add_s1`–`s3`, `fast_dot`, `fast_dot_s1`–`s3`, `head_1h_dot`, `swap_fr_dot`, `var_bilinear`, `var_entmax15`, `var_gated_dot_additive`, `var_global_local`, `var_local_window`, `var_sparsemax`.

**Classification rules (as requested):**

- **COMPLETE:** `metrics.json`, `resolved_config.json`, `training_meta.json`, and `test_eval/metrics_test.json` all present.
- **PARTIAL:** at least one of the above missing (checkpoints may still exist).
- **CHECKPOINT_ONLY:** `best.pt` or `last.pt` present **and** no `test_eval/` directory.

**Summary table**

| run | status | has_best_pt | has_predictions_jsonl | notes |
|-----|--------|-------------|------------------------|-------|
| fast_add | COMPLETE | yes | yes | |
| fast_add_s1 | COMPLETE | yes | yes | also `final_eval_meta.json`, `predictions_val_sample.jsonl` |
| fast_add_s2 | COMPLETE | yes | yes | same extras as s1 |
| fast_add_s3 | COMPLETE | yes | yes | same extras as s1 |
| fast_dot | COMPLETE | yes | yes | |
| fast_dot_s1 | COMPLETE | yes | yes | `final_eval_meta.json`, `predictions_val_sample.jsonl` |
| fast_dot_s2 | COMPLETE | yes | yes | same extras |
| fast_dot_s3 | COMPLETE | yes | yes | same extras |
| head_1h_dot | PARTIAL | yes | yes | missing `training_meta.json` |
| swap_fr_dot | PARTIAL | yes | yes | missing `training_meta.json` |
| var_bilinear | COMPLETE | yes | yes | `final_eval_meta.json`, `predictions_val_sample.jsonl` |
| var_entmax15 | COMPLETE | yes | yes | same extras |
| var_gated_dot_additive | COMPLETE | yes | yes | same extras |
| var_global_local | COMPLETE | yes | yes | same extras |
| var_local_window | COMPLETE | yes | yes | same extras |
| var_sparsemax | COMPLETE | yes | yes | same extras |

**CHECKPOINT_ONLY:** none — every run under `runs/` has a `test_eval/` directory.

**Per-run contents** (files under each run, depth ≤ 3; paths relative to `runs/<run>/`):

<details>
<summary><code>fast_add</code></summary>

```
best.pt
last.pt
metrics.json
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
training_meta.json
```

</details>

<details>
<summary><code>fast_add_s1</code> / <code>s2</code> / <code>s3</code></summary>

```
best.pt
final_eval_meta.json
last.pt
metrics.json
predictions_val_sample.jsonl
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
training_meta.json
```

</details>

<details>
<summary><code>fast_dot</code></summary>

```
best.pt
last.pt
metrics.json
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
training_meta.json
```

</details>

<details>
<summary><code>fast_dot_s1</code> / <code>s2</code> / <code>s3</code></summary>

```
best.pt
final_eval_meta.json
last.pt
metrics.json
predictions_val_sample.jsonl
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
training_meta.json
```

</details>

<details>
<summary><code>head_1h_dot</code></summary>

```
best.pt
last.pt
metrics.json
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
```

</details>

<details>
<summary><code>swap_fr_dot</code></summary>

```
best.pt
last.pt
metrics.json
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
```

</details>

<details>
<summary><code>var_*</code> (bilinear, entmax15, gated_dot_additive, global_local, local_window, sparsemax)</summary>

```
best.pt
final_eval_meta.json
last.pt
metrics.json
predictions_val_sample.jsonl
resolved_config.json
test_eval/examples.md
test_eval/metrics_test.json
test_eval/predictions.jsonl
training_meta.json
```

</details>

**Checkpoints (`find runs -name 'best.pt' -o -name 'last.pt'`):** all 18 runs include both `best.pt` and `last.pt` under `runs/<name>/`.

**Predictions (`find runs -path '*/test_eval/predictions.jsonl'`):** all 18 runs have `test_eval/predictions.jsonl`.

**Runs lacking `test_eval/predictions.jsonl`:** **none.**

---

## Artifact consistency checker

Command: `python scripts/check_artifact_consistency.py 2>&1 | tee /tmp/artifact_check_before.log`

**Exit code:** 1 (issues reported; expected per instructions).

**Reported issues:**

1. `small_swap/results/manifest_swap.json`: `runs.dot` path uses `small_swap/runs/swap_fr_dot`; checker expects repository-root `runs/…` layout.
2. `runs/head_1h_dot`: missing `training_meta.json` (training run stats).
3. `runs/swap_fr_dot`: missing `training_meta.json` (training run stats).

---

## Critical reminder

- **`metrics.json`** records **validation / training-time** evaluation (per training script and `eval_split`, typically **val**). It is **not** the same population or protocol as the held-out test evaluation.
- **`test_eval/metrics_test.json`** comes from **held-out `test.tsv`** evaluation (e.g. via `evaluate_test.py`). **Do not compare or rank runs by directly equating these two files** — they answer different questions.

---

## Recommended next actions

1. **Restore or regenerate `training_meta.json`** for `runs/head_1h_dot` and `runs/swap_fr_dot` so they match the COMPLETE artifact set and pass `scripts/check_artifact_consistency.py` (unless omission is intentional and documented).
2. **Align `small_swap/results/manifest_swap.json`** with the canonical `runs/swap_fr_dot` path at repo root, or adjust the checker’s expected convention — otherwise manifest-driven tooling may point at the wrong checkpoint directory.
3. **Keep using `test_eval/metrics_test.json` + `predictions.jsonl`** for cross-run comparisons on the **test** split; cite `metrics.json` only in **val / training** contexts.
4. **Retain `environment_freeze_h800.txt`** (or merge into your environment docs) if you need to reproduce this AutoDL H800 stack elsewhere.
