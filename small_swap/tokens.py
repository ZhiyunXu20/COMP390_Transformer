"""特殊符号 ID：与目标 tokenizer JSON（added_tokens id 0–3）一致。

若使用 scripts/train_tokenizers_from_train_split.py 重新训练词表，应保持 UNK/PAD/BOS/EOS 字符串与顺序不变。
"""

# 与 tokenizer_tgt*.json（added_tokens id 0–3）一致
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
