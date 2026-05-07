# Metric Provenance

## Why this matters

BLEU is fully provenance-trackable via SacreBLEU's `sacrebleu_signature` field, which records exact tokenization, smoothing, etc. BERTScore and COMET both depend on a pre-trained model whose identity must be recorded to make held-out test numbers reproducible.

## Forward-only fix (V10-A38)

As of **V10-A38**, `evaluate_test.py` writes a **`metric_provenance`** object into each newly produced `runs/<run>/test_eval/metrics_test.json`, containing:

- `sacrebleu_signature` (mirrors the top-level field)
- `sacrebleu_version`
- `bertscore_lang`, `bertscore_model_type`, `bertscore_version`
- `comet_model`, `comet_version`, `comet_device`
- `_caveat` (short pointer to this document)

The top-level **`sacrebleu_signature`** key is retained for backward compatibility with tools that predate the nested block.

## Historical 36 archived runs

The **36** archived `runs/*/test_eval/metrics_test.json` files were written **before** this block and **do not** contain `metric_provenance`. Their **`sacrebleu_signature`** is still trustworthy. Their BERTScore and COMET numbers were computed with:

- **BERTScore**: `lang="fr"` (English→French target language); `model_type=None` (defaults to **bert-base-multilingual-cased** per **bert-score**'s default for `"fr"`); package version follows **`environment_freeze_h800.txt`**.
- **COMET**: model **`Unbabel/wmt22-comet-da`**; package version follows **`environment_freeze_h800.txt`**.
- **Device**: **CUDA** on the H800 training host.

These match the defaults in **`small_try/config.py`**, **`small_head/config.py`**, **`small_swap/config.py`** at training-commit time and have not changed since A23.

## Defensible claims

The paper may quote BLEU and chrF++ with full provenance (`sacrebleu_signature` suffices). For **BERTScore** and **COMET**, cite this document as the provenance source for the historical **36** archived runs; for new evaluations after V10-A38, quote the **`metric_provenance`** object in the corresponding `metrics_test.json`.
