# Data split overlap audit

Parser: line.rstrip('\n').split('\t')
Source files: data/splits/en_fr_50k_seed42/{train,val,test}.tsv
Note: default csv.reader is unsafe here because DCEP text contains un-escaped natural-language double quotes.

## Summary

- **row_counts**: {'train': 45000, 'val': 2500, 'test': 2500}
- **bad_rows_skipped** (len(parts) != 2): {'train': 0, 'val': 0, 'test': 0}
- **empty_src_or_tgt_rows** (after strip): {'train': 0, 'val': 0, 'test': 0}
- **src_eq_tgt_rows**: {'train': 0, 'val': 0, 'test': 0}

## Exact (source, target) pair overlap

| pair | count |
|---|---:|
| train-val | 0 |
| train-test | 0 |
| val-test | 0 |

## Source-string overlap

| pair | count |
|---|---:|
| train-val | 10 |
| train-test | 8 |
| val-test | 1 |

## Target-string overlap

| pair | count |
|---|---:|
| train-val | 8 |
| train-test | 9 |
| val-test | 1 |

## 1-to-many (union of all splits)

- **source strings with >1 distinct target**: 115
- **target strings with >1 distinct source**: 107

