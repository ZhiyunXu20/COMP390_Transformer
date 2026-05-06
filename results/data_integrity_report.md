# Data integrity report

- **split directory**: `/root/autodl-tmp/data/splits/en_fr_50k_seed42`
- **overall**: **PASS**

| file | source | recorded_hash | actual_hash | status |
|------|--------|---------------|-------------|--------|
| train.tsv | manifest.json | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | PASS |
| train.tsv | split_metadata.json (train_file_sha256_for_tokenizers) | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | PASS |
| val.tsv | manifest.json | `5d0bb2929626ebd58e5cd937d31ca8f0aefe3b1cb9980e36926364f2ae226996` | `5d0bb2929626ebd58e5cd937d31ca8f0aefe3b1cb9980e36926364f2ae226996` | PASS |
| val.tsv | split_metadata.json (val_file_sha256) | `5d0bb2929626ebd58e5cd937d31ca8f0aefe3b1cb9980e36926364f2ae226996` | `5d0bb2929626ebd58e5cd937d31ca8f0aefe3b1cb9980e36926364f2ae226996` | PASS |
| test.tsv | manifest.json | `440077a0359839a623104346abaa6a2d3a9c5467ac655ee456695e714c387667` | `440077a0359839a623104346abaa6a2d3a9c5467ac655ee456695e714c387667` | PASS |
| test.tsv | split_metadata.json (test_file_sha256) | `440077a0359839a623104346abaa6a2d3a9c5467ac655ee456695e714c387667` | `440077a0359839a623104346abaa6a2d3a9c5467ac655ee456695e714c387667` | PASS |
| train.tsv | tokenizer_metadata.json (train_file_sha256 vs train.tsv SHA-256) | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | `16d25ad3d6e61bf3419be60d181c7ba605eb003f82dfa72010807e7b84d6f5d1` | PASS |
