# V3 release checklist (A16 — end-to-end verification)

Generated: 2026-05-06 (Asia/Shanghai). Commands were run from repository root `/root/autodl-tmp`.

## Tests

- **Command:** `python -m pytest -q`
- **Result:** **PASS** — `41 passed in ~1.5s`

## Sanity

- **Command:** `python scripts/sanity_check_runs.py`
- **Outputs:** `results/runs_sanity_report.md`, `results/runs_sanity_report.json`
- **Findings:** Table lists all indexed runs. **Expected failures / warnings:**
  - `var_local_window` and A15 probes (`var_local_window_fp32`, `var_local_window_lr1e4`, `var_local_window_window16`): **`failed_nan`** (BLEU 0, empty hyps) — documented in `results/local_window_stability_report.md`
  - `lr_sweep_add_lr1e4`: **`warning_low_quality`** (low BLEU)
- **Script exit:** `0`

## Aggregates

Files **regenerated during this A16 pass** (last modification time, local clock):

| Path | Mtime (local) |
|------|----------------|
| `results/runs_sanity_report.md` | 2026-05-06 17:51:14 +0800 |
| `results/runs_sanity_report.json` | 2026-05-06 17:51:14 +0800 |
| `results/attention_variants_test_summary.md` | 2026-05-06 17:51:14 +0800 |
| `results/attention_variants_test_summary.csv` | 2026-05-06 17:51:14 +0800 |
| `results/variant_fairness_audit.md` | 2026-05-06 17:51:16 +0800 |
| `results/variant_fairness_audit.json` | 2026-05-06 17:51:16 +0800 |
| `results/cross_seed_significance.md` | 2026-05-06 17:51:17 +0800 |
| `results/cross_seed_significance.json` | 2026-05-06 17:51:17 +0800 |

**Not regenerated in step 2** (still on disk from earlier work): e.g. `results/ablation_summary.{md,csv}`, `results/significance_fast_dot_vs_fast_add.{md,json}`, `results/variant_multiseed_summary.md`, `results/local_window_stability_report.md`.

**Command:** `python scripts/summarize_attention_variants.py --runs fast_dot fast_add head_1h_dot swap_fr_dot var_bilinear var_gated_dot_additive var_sparsemax var_entmax15 var_local_window var_global_local` — **PASS** (exit 0).

**Command:** `python scripts/check_variant_experiment_fairness.py` (same `--runs`) — see **Significance / fairness** below (script exit 1).

**Command:** `python scripts/cross_seed_significance.py --csv results/ablation_per_seed.csv --experiment-a fast_dot --experiment-b fast_add` — **PASS** (exit 0).

## Significance

**Cross-seed Welch** (`results/cross_seed_significance.md`, regenerated in this pass):

- **BLEU:** mean difference (fast_dot − fast_add) ≈ **7.597** (report: 7.59733); **Welch t ≈ 15.12**; df ≈ 3.71; two-sided **p ≈ 1.81×10⁻⁴**; conclusion: significant at α = 0.05.
- chrF++ and COMET blocks in the same file also favor fast_dot with small p-values.

**Fairness audit** (`results/variant_fairness_audit.md`):

- **`swap_fr_dot`:** **FAIL** vs `fast_dot` reference — mismatched **`train_path`**, **`tokenizer_src`**, **`tokenizer_tgt`** (intentional FR↔EN / tokenizer swap); **`comparable_to_fast_dot` = False**. This triggers **exit code 1** for `check_variant_experiment_fairness.py` even though artifacts are present.
- Other **WARN** rows: `fast_add`, `var_gated_dot_additive` (parameter delta); `head_1h_dot` (n_heads + repaired `training_meta`).

## Archive completeness

- **Command:** `python scripts/check_artifact_consistency.py`
- **Result:** **PASS** (exit 0) with **non-blocking WARN:** `head_1h_dot` and `swap_fr_dot` have **`training_meta.json` repaired** (`metadata_repaired=true`).

- **Command:** `python scripts/check_submission_archive.py --strict`
- **Result:** **`prepare_code_archive` validation OK**; **overall: WARN** (exit 0)
- **WARN items:** large `data/EN-FR.txt` (~898 MiB) should be excluded from submission zip; numerous `logs/*.log` files flagged as “consider omitting”.

- **Written:** `results/submission_archive_check.md` (mtime 2026-05-06 17:51:18 +0800 during this run).

## Documentation alignment

| Check | Result |
|--------|--------|
| `grep -c "not trained" docs/ATTENTION_VARIANTS_STATUS.md` | **0** (no matches) |
| `grep -c "missing_test_eval" results/attention_variants_test_summary.md` | **0** (after regeneration) |
| `grep -E "validation-sampled" small_try/report.txt small_head/report_head.txt small_swap/report_swap.txt` | **no matches** |
| `grep -E "15\\.80|7\\.60|Welch" docs/THESIS_RESULT_BOUNDARIES.md` | **≥ 1 match** (cross-seed / headline BLEU caveats present) |

## `.gitignore` spot checks

| Check | Expected | Observed |
|--------|-----------|----------|
| `git check-ignore -v runs/fast_dot_s1/test_eval/predictions.jsonl` | not ignored (non-zero exit) | **not ignored** (exit `1`, no rule) |
| `git check-ignore -v runs/fast_dot/best.pt` | ignored (exit `0`) | **ignored** by `.gitignore` `**/*.pt` (exit `0`) |

## Outstanding issues

1. **Fairness script exit 1:** `swap_fr_dot` is intentionally not comparable to `fast_dot`; treat as **documented exception** or exclude `swap_fr_dot` from strict fairness CI if a green exit is required without losing the check for other runs.
2. **Submission archive WARN:** exclude bulky raw corpus and optional logs from any zip meant for examination-only submission (see `submission_archive_check.md`).
3. **Repaired `training_meta`:** `head_1h_dot`, `swap_fr_dot` — wall time / GPU fields may not be original; `check_artifact_consistency.py` already warns.
4. **`local_window` family:** training/eval artifacts exist but metrics are NaN / zero BLEU; cite `results/local_window_stability_report.md` in prose, not as a successful variant.
5. **Git working tree:** many **untracked** paths (A15 runs, multiseed runs, new experiment scripts) — user must **review `git status`** and stage what should ship; this checklist does **not** commit.

---

## Git snapshot (no commit performed)

At verification time:

- `git status --short` showed modified tracked files (docs, `requirements.txt`, results, `small_try/*`, `train_runtime.py`) plus untracked experiments/results/runs as listed in the agent log.
- `git diff --stat HEAD` reflected **tracked** diffs only (untracked runs not included in `--stat`).

**User action:** review diffs, stage intended files, commit and push manually (no push was done from automation).
