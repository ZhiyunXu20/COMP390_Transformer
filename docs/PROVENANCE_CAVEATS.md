# Provenance Caveats

## Historical absolute paths

The repository contains paths beginning with `/root/autodl-tmp/...` in the following places (these are historical machine-local paths from the AutoDL H800 training host, **not** current source-of-truth for replay):

- `data/splits/en_fr_50k_seed42/manifest.json`: `source_path` field
- `data/splits/en_fr_50k_seed42/split_metadata.json`: `tokenizer_metadata_path`, `tokenizer_src_json`, `tokenizer_tgt_json` (and related path strings)
- `data/tokenizer_metadata.json`: `train_file`, tokenizer output paths, and related string fields (not used by the loader when training runs on relative `Config` paths)
- `results/ablation_per_seed.csv`: `checkpoint_path` column
- `results/cross_seed_significance*.json`: `csv` and similar metadata fields
- `runs/*/resolved_config.json` and `runs/*/metrics.json`: `data_path`, `train_path`, `val_path`, `test_path`, and other resolved absolute paths

Some **documentation** under `docs/` (e.g. checklists) may mention `/root/autodl-tmp` as the host root where commands were executed; that is descriptive only.

**Replay:** use **repo-relative** paths in `small_try` / `small_head` / `small_swap` `Config` defaults and in CLI overrides. The dataset loaders consume those strings relative to the repository root (after `train_runtime.materialize_path_fields` where applicable). Absolute paths embedded in committed `metrics.json` / `resolved_config.json` are **provenance-only** and **do not** control loading on a fresh checkout when you pass relative paths.

**Legacy `base_1` / `base_improve`:** default `Config` dataclasses in those packages may still use `/root/autodl-tmp/...` strings. They are older stacks; prefer `small_try` for documented replication paths, or rewrite those defaults locally—do not treat them as canonical for this archive.

## Current source-of-truth for data integrity

The authoritative integrity source is:

1. SHA-256 of the shipped `data/splits/en_fr_50k_seed42/{train,val,test}.tsv` files (verified by `scripts/check_data_integrity.py`).
2. `data/tokenizer_metadata.json`: `train_file_sha256` matches the actual `train.tsv` SHA-256 on disk.
3. `data/splits/en_fr_50k_seed42/manifest.json`: `sha256` matches the actual TSV SHA-256 (regenerated under A20; field `regenerated_from_actual_files: true` when present).

## Duplicate-count metadata mismatch

`manifest.json` reports `duplicate_pairs_removed` (e.g. **1,416,853**) from an **older** generation round; `split_metadata.json` reports `number_of_removed_duplicates` (e.g. **1,121,066**) from a **later** round. These come from **different** historical pipeline invocations. **Do not** cite either number as the single current ground truth for “how many duplicates existed globally.”

The operational ground truth for **what is in this archive** is the shipped splits: **45,000 / 2,500 / 2,500** rows under the project’s line-delimited TSV loader semantics (see `docs/THESIS_RESULT_BOUNDARIES.md`, V9-A29), as verified by `scripts/check_data_integrity.py` and `scripts/audit_parallel_data.py`.

## Audit helper

Run `python scripts/check_absolute_path_caveats.py` to list **git-tracked** files that still contain `/root/autodl-tmp` and confirm they fall into the categories above (untracked local logs or caches are out of scope). Output: `results/absolute_path_caveats_audit.md`.
