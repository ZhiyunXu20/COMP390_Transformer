# Additive at matched LR (A19)

## Protocol

- **Model**: `small_try`, `attention_type=additive`, `n_heads=4`.
- **Training**: `lr=3e-3`, `max_steps=3000`, seeds **1/2/3**, val BLEU for checkpointing;
  `--no-wandb`, `--eval-light` during training (same style as other multiseed jobs).
- **Held-out test**: `evaluate_test.py` **without** `--eval-light` (full BLEU/chrF++/COMET/BERTScore).
- **Runs**: `runs/fast_add_lr3e3_s1` … `s3` (new directories only).

## Cross-seed summary (held-out test BLEU)

- **fast_dot** (reference, `lr=3e-4`, n=3): mean=15.8, std=0.5225.
- **fast_add** @ **`lr=3e-4`** (n=3): dot–additive gap **≈ 7.60 BLEU** (Welch two-sided *p* ≈ 0.0001808; see `results/cross_seed_significance.md`).
- **fast_add_lr3e3** (A19, `lr=3e-3`, n=3): mean=21.56, std=0.4181.
- **fast_dot − fast_add_lr3e3** (Welch): ΔBLEU **-5.762**, *t*=-14.914, df≈3.82, two-sided *p*=0.0001593 (`results/cross_seed_significance_lr3e3.json`).

## Interpretation (for thesis / boundaries)

With **matched** `max_steps=3000`, **additive attention trained with a tuned higher learning rate (`3e-3`) under the same 3000-step budget, while dot-product retains its baseline lr=3e-4**, the cross-seed held-out contrast **fast_dot − additive** moves from **≈7.60 BLEU** (`fast_add` @ **`3e-4`**) to **≈-5.76 BLEU** (`fast_add_lr3e3`): the **ordering reverses** (additive mean BLEU exceeds dot). The shift in the mean contrast is **≈13.36 BLEU**. In thesis language, the original **~7.60 BLEU** dot margin at `3e-4` is **sufficient to account for and reverse the originally observed dot-product advantage under the tested protocols** (**the ranking sign reverses, which is a stronger condition than mere gap reduction**). **A19 demonstrates strong optimization confounding under matched per-mechanism learning-rate protocols; it does not prove the two mechanisms are intrinsically equivalent.** Use Welch *t* / *p* above and the JSON for exact CIs.

## Citations

- `results/ablation_per_seed.csv` (`experiment=fast_add_lr3e3`).
- `results/cross_seed_significance_lr3e3.{md,json}`.
- Original `fast_dot` vs `fast_add` @ 3e-4: `results/cross_seed_significance.{md,json}`.
