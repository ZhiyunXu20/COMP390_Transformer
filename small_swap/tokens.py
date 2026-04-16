"""特殊符号 ID：与当前 tokenizer_tgt.json（added_tokens id 0–3）一致。

若重新训练分词器，请同步此处或通过 tokenizer.id_to_token 校验。
"""

# 与 /root/autodl-tmp/data/tokenizer_tgt.json 中顺序一致
UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
