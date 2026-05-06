# Local window numerical stability (A15)

## Setup

- **Original failure:** `runs/var_local_window` with default CUDA **bf16 autocast**, **lr=3e-4**, **`local_window_size=4`**, **max_steps=3000** → `val_loss=NaN`, validation BLEU=0, held-out test BLEU=0, **empty hypotheses ~100%**.
- **Diagnostics:** three runs (**max_steps=1500**, **seed=42**) with training-time `--eval-light` (BLEU/chrF on val subsample only) for wall-clock; **held-out** `evaluate_test.py` below uses **`--eval-light`** as well for the same reason.
- **New outputs:** `runs/var_local_window_fp32/`, `runs/var_local_window_lr1e4/`, `runs/var_local_window_window16/` (original directory untouched).

## Results

| experiment | precision | lr | window | val_loss progression (subset) | val final BLEU | test BLEU | empty_hyp_pct |
|------------|-----------|----|--------|------------------------------|----------------|-----------|---------------|
| original (var_local_window) | bf16 | 3e-4 | 4 | NaN | 0.0000 | 0.0000 | 100.0% |
| fp32 control | fp32 | 3e-4 | 4 | nan → nan → nan → … | 0.0000 | 0.0000 | 100.0% |
| lower lr | bf16 | 1e-4 | 4 | nan → nan → nan → … | 0.0000 | 0.0000 | 100.0% |
| larger window | bf16 | 3e-4 | 16 | nan → nan → nan → … | 0.0000 | 0.0000 | 100.0% |

**Success criterion (training health):** finite `final_val_loss`, validation `final_bleu` > 1, and non-trivial held-out BLEU / empty-hyp rate.

## Diagnosis

- If **fp32 control** trains successfully → primary cause is **bf16 numerical stability** with multi-layer **−∞ structural masks**.
- If **lower lr** trains successfully → **gradient instability** at high lr × mask interaction.
- If **larger window** trains successfully → **too-restrictive window** (weak attention support).
- If **all three** diagnostics fail → **limitations of this implementation/stack** under the probe (not bf16-, lr-, or window=4-only); in this codebase the leading story is **encoder local-band × batched padding** (see below).

**This archive:** fp32=no, lower_lr=no, larger_window=no.

The original local-window failure is attributable to an interaction between **encoder** `local_window` structural masking and **batched key-padding**—not resolved by fp32, lower lr, or `local_window_size=16` in this probe. Empirically: **`--no-bf16-autocast` still yields NaN** from the first training steps, so **bf16 alone is not the root cause**. A plausible implementation-level mechanism: for source positions deep in the **padded tail**, the allowed **|i−j| ≤ w** band can lie entirely inside keys that are **all PAD** (−∞ from `memory_key_padding` / encoder pad mask), so a softmax row is **all −∞** → **NaN attention**, which poisons `memory` and downstream logits (see `small_try/attention.py` `_structural_local_window` + `model.py` `_key_pad_mask`). `global_local` mitigates this class of failure via **global anchor columns** (see A15 vs. `runs/var_global_local`). We do NOT claim the local-window mechanism itself is unviable in general; we claim that under this repository’s **encoder local-band + batched padding** interaction, the default **bf16** run **and** the A15 mitigations (fp32 math, lower lr, wider window) **still** produce NaN training in these probes. With **encoder-side mask fixes (e.g. union with a safe visibility pattern for pad queries, global anchors, or finite large-negative logits) beyond this diagnostic grid**, training is **still not recovered in these probes**.

## Post-fix verification (A23 retrain)

**Code shipped in A23 (commit `bd1b53e` and follow-up):** `small_shared/attention_ops.safe_masked_softmax` (train-safe all-masked rows), full **encoder** `(B,1,L,L)` self-attention mask with **query-padding** rows set to −∞, and analogous **decoder** self-attention query-row bias—without changing the dense local-band *idea*.

**New run:** `runs/var_local_window_a23_retrain` — `attention_type=local_window`, `local_window_size=4`, `seed=42`, `max_steps=3000`, training with `--eval-light` (val subsample only during training).

| metric | value | where |
|--------|-------|--------|
| final_val_loss (training) | ≈3.12 (finite) | `runs/var_local_window_a23_retrain/metrics.json` |
| final_bleu (training / sampled val) | ≈18.41 | same |
| Held-out **test** BLEU | **16.82** | `runs/var_local_window_a23_retrain/test_eval/metrics_test.json` |
| Held-out **test** chrF++ | **36.69** | same |
| Held-out **test** COMET | **0.523** | same |

The A15 probe directories (`var_local_window`, `var_local_window_fp32`, etc.) remain **valid historical evidence** of failure **before** the softmax/mask patch; they were **not** re-executed on post-A23 code.

## Implications for dissertation

Use the **Results** table and the diagnosis above when writing the thesis; cite **`results/local_window_stability_report.md`**. Do **not** claim the failure is **only** bf16 autocast—A15 fp32 control still diverges. Do **not** claim `local_window` is universally untrainable; claim only that **this repository’s dense local-window + padding interaction** fails under the tested protocol. Avoid claiming a sparse Longformer-style kernel; the code path is **dense L×L masked attention**.

