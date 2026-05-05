# Attention variants: code vs experiments (`attention_type`)

Scope: **`small_try` / `small_head` / `small_swap`** share the same `attention.py` pattern and `Config.attention_type` union (`small_try/config.py`). **`base_1`** only implements `dot_product` and `additive` (see `base_1/README.md`) and is **not** covered by the rows below.

**Important:** `local_window` and `global_local` are **dense masked attention** (full \(L \times L\) scores, structural bias added to logits, then softmax). They are **not** a kernel-efficient sparse implementation in the sense of Longformer, BigBird, ETC, etc.

Experiment status **trained** below means: this repository snapshot contains at least one **full training run** under `runs/` whose `resolved_config.json` / `metrics.json` reports that `attention_type` as the trained model (e.g. `fast_dot`, `fast_add`). Other variants may appear as exploratory `var_*` runs or lack a committed run for the main baseline grid—see the table.

### Single-seed caveat

Variants below were trained with one random seed (the default seed used in the variant_registry workflow). Cross-seed variation observed in fast_dot is std=0.52 BLEU and in fast_add is std=0.70 BLEU. Therefore differences between single-seed variants of magnitude less than approximately 1.0 BLEU should not be interpreted as ranked differences.

| `attention_type` | Mechanism family (scoring / normalization / connectivity) | Code status | Experiment status | Scientific purpose | Safe thesis wording |
|------------------|------------------------------------------------------------|-------------|-------------------|-------------------|---------------------|
| `dot_product` | Scaled dot-product / **softmax** / **full** (dense) | implemented | **trained** (e.g. `runs/fast_dot`, `runs/head_1h_dot`, `runs/swap_fr_dot`) | Standard Transformer baseline for comparison. | We use Vaswani-style scaled dot-product attention with softmax as our primary baseline. |
| `additive` | Additive (feedforward) compatibility scores / **softmax** / **full** | implemented | **trained** (e.g. `runs/fast_add`) | Classical Bahdanau-style compatibility as an alternative scoring function to dot-product. | We compare softmax-normalized **additive** scoring to the dot-product baseline under the same encoder–decoder stack. |
| `bilinear` | Head-specific bilinear \(q,W_h,k\) / **softmax** / **full** | implemented | **trained (single seed)** (`runs/var_bilinear`); held-out test BLEU=15.1704, chrF++=35.369, COMET=0.505744 | Tests a low-rank bilinear interaction instead of dot-product per head. | A bilinear compatibility score per head is **implemented**; single-seed held-out test scores are exploratory—see caveat above. |
| `gated_dot_additive` | Convex mix of dot and additive logits (learned gate) / **softmax** / **full** | implemented | **trained (single seed)** (`runs/var_gated_dot_additive`); held-out test BLEU=14.9050, chrF++=34.9101, COMET=0.499885 | Lets the model interpolate between dot-product and additive shapes without committing to one. | The gated dot–additive module is an **engineering option** for ablation; single-seed results are exploratory. |
| `sparsemax` | Dot-product logits / **sparsemax** (Martinez et al.) / **full** | implemented (uses `entmax` package when installed; else naive fallback) | **trained (single seed)** (`runs/var_sparsemax`); held-out test BLEU=7.64874, chrF++=25.0512, COMET=0.445171 | Sparse normalizers can yield exactly zero weights on some keys; compares normalization choice to softmax. | Sparsemax replaces softmax on **dense** scores; this studies **normalization**, not sparse attention **kernels** or sub-quadratic connectivity. |
| `entmax15` | Dot-product logits / **entmax** with α = 1.5 (bisect; falls back if deps missing) / **full** | implemented | **trained (single seed)** (`runs/var_entmax15`); held-out test BLEU=13.6606, chrF++=32.6658, COMET=0.496477 | Intermediate sparsity between softmax and sparsemax. | Same caveat as sparsemax: alternative **normalizer** on a **dense** score matrix; **not** claimed as a new efficient attention mechanism. |
| `local_window` | Dot-product logits + structural mask / **softmax** / **dense** matrix with **band-like allowed pattern** | implemented | **trained but FAILED** (val_loss=NaN, BLEU=0.0); see `results/runs_sanity_report.md` (`runs/var_local_window`) | Encodes a **prior** that keys outside a window receive large negative bias (still computed via dense matmul). | We use **dense** masked attention with a **local window pattern**; we **do not** claim Longformer-class sparse or linear-time attention. |
| `global_local` | Dot-product logits + structural mask / **softmax** / **dense** matrix with **global anchors + local band** pattern | implemented | **trained (single seed)** (`runs/var_global_local`); held-out test BLEU=7.30136, chrF++=22.7096, COMET=0.436818 | Same as above: structural prior on **who may attend**, not a sparse implementation. | **Dense** global–local **masking** is a connectivity **prior**, not a sparse kernel or memory-efficient attention implementation. |

## How to extend “experiment status”

- Training defaults in `experiments/run_ablation.py` only vary **`dot_product`** vs **`additive`** with head counts (`ABLATION_SPECS`).
- To add another committed run, train via `small_try/train.py` (or another pkg) with `--attention-type <name>` and keep `runs/<name>/` artifacts referenced by `scripts/summarize_attention_variants.py` and `results/attention_variants_test_summary.*`.

## Dependencies worth mentioning in prose

- **`sparsemax` / `entmax15`**: Optional **`entmax`** PyPI package improves speed and numerical behavior; without it, sparsemax uses an internal naive path and entmax15 may degrade (see `small_try/attention.py` imports).
- Do **not** describe `local_window` / `global_local` as reducing asymptotic complexity; cost remains that of standard dense multi-head attention for sequence length \(L\).
