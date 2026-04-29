# Attention variants: code vs experiments (`attention_type`)

Scope: **`small_try` / `small_head` / `small_swap`** share the same `attention.py` pattern and `Config.attention_type` union (`small_try/config.py`). **`base_1`** only implements `dot_product` and `additive` (see `base_1/README.md`) and is **not** covered by the rows below.

**Important:** `local_window` and `global_local` are **dense masked attention** (full \(L \times L\) scores, structural bias added to logits, then softmax). They are **not** a kernel-efficient sparse implementation in the sense of Longformer, BigBird, ETC, etc.

Experiment status **trained** below means: this repository snapshot contains at least one **full training run** under `runs/` whose `resolved_config.json` / `metrics.json` reports that `attention_type` as the trained model (e.g. `fast_dot`, `fast_add`). Anything else is **not trained** for the purposes of the main reported baselines—unit tests and `scripts/profile_model.py` may still instantiate the module.

| `attention_type` | Mechanism family (scoring / normalization / connectivity) | Code status | Experiment status | Scientific purpose | Safe thesis wording |
|------------------|------------------------------------------------------------|-------------|-------------------|-------------------|---------------------|
| `dot_product` | Scaled dot-product / **softmax** / **full** (dense) | implemented | **trained** (e.g. `runs/fast_dot`, `runs/head_1h_dot`, `runs/swap_fr_dot`) | Standard Transformer baseline for comparison. | We use Vaswani-style scaled dot-product attention with softmax as our primary baseline. |
| `additive` | Additive (feedforward) compatibility scores / **softmax** / **full** | implemented | **trained** (e.g. `runs/fast_add`) | Classical Bahdanau-style compatibility as an alternative scoring function to dot-product. | We compare softmax-normalized **additive** scoring to the dot-product baseline under the same encoder–decoder stack. |
| `bilinear` | Head-specific bilinear \(q,W_h,k\) / **softmax** / **full** | implemented | **not trained** (no dedicated committed run in `runs/`) | Tests a low-rank bilinear interaction instead of dot-product per head. | A bilinear compatibility score per head is **implemented** but **not part of our reported training grid** unless separately run. |
| `gated_dot_additive` | Convex mix of dot and additive logits (learned gate) / **softmax** / **full** | implemented | **not trained** | Lets the model interpolate between dot-product and additive shapes without committing to one. | The gated dot–additive module is an **engineering option** for ablation; we do **not** claim novel architecture beyond a standard gated combination of two classical score families. |
| `sparsemax` | Dot-product logits / **sparsemax** (Martinez et al.) / **full** | implemented (uses `entmax` package when installed; else naive fallback) | **not trained** | Sparse normalizers can yield exactly zero weights on some keys; compares normalization choice to softmax. | Sparsemax replaces softmax on **dense** scores; this studies **normalization**, not sparse attention **kernels** or sub-quadratic connectivity. |
| `entmax15` | Dot-product logits / **entmax** with α = 1.5 (bisect; falls back if deps missing) / **full** | implemented | **not trained** | Intermediate sparsity between softmax and sparsemax. | Same caveat as sparsemax: alternative **normalizer** on a **dense** score matrix; **not** claimed as a new efficient attention mechanism. |
| `local_window` | Dot-product logits + structural mask / **softmax** / **dense** matrix with **band-like allowed pattern** | implemented | **not trained** | Encodes a **prior** that keys outside a window receive large negative bias (still computed via dense matmul). | We use **dense** masked attention with a **local window pattern**; we **do not** claim Longformer-class sparse or linear-time attention. |
| `global_local` | Dot-product logits + structural mask / **softmax** / **dense** matrix with **global anchors + local band** pattern | implemented | **not trained** | Same as above: structural prior on **who may attend**, not a sparse implementation. | **Dense** global–local **masking** is a connectivity **prior**, not a sparse kernel or memory-efficient attention implementation. |

## How to extend “experiment status”

- Training defaults in `experiments/run_ablation.py` only vary **`dot_product`** vs **`additive`** with head counts (`ABLATION_SPECS`).
- To mark another row as **trained**, run `small_try/train.py` (or another pkg) with `--attention-type <name>` and keep the resulting `runs/<name>/` artifacts reproducibly referenced.

## Dependencies worth mentioning in prose

- **`sparsemax` / `entmax15`**: Optional **`entmax`** PyPI package improves speed and numerical behavior; without it, sparsemax uses an internal naive path and entmax15 may degrade (see `small_try/attention.py` imports).
- Do **not** describe `local_window` / `global_local` as reducing asymptotic complexity; cost remains that of standard dense multi-head attention for sequence length \(L\).
