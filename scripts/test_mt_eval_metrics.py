#!/usr/bin/env python3
"""自检 BERTScore / COMET 是否可用（依赖 mt_eval，需能访问 HF 或已有本地缓存）。

正式训练里 BERTScore 的 lang 已按任务写在各包 config（勿与本脚本混为一谈）：
  - small_try / small_head（英→法）：eval_bertscore_lang = "fr"
  - small_swap（法→英）：eval_bertscore_lang = "en"

本脚本仅用 lang=en + distilbert-base-uncased 做「镜像/网络是否通」的快速探测，体积小；
不表示整条 small 流水线只用 en。

运行（AutoDL 可先开启学术加速，再执行本脚本）：
  cd /root/autodl-tmp && source /etc/network_turbo   # network_turbo 可选
  python3 scripts/test_mt_eval_metrics.py

脚本在 import 任何 Hub 相关库之前写入 HF 环境变量；默认强制 HF_ENDPOINT=hf-mirror，
避免 shell 里已存在的慢速 HF_ENDPOINT 无法被 setdefault 覆盖。走官网：HF_USE_OFFICIAL_HUB=1

若 source /etc/network_turbo 后 BERTScore/COMET 全失败，本脚本会为 Hub 域名追加 NO_PROXY；
仍异常可 HF_SKIP_NO_PROXY_FIX=1 关闭该逻辑，或不用 network_turbo 再试。
"""
from __future__ import annotations

import os
import sys

# 须在 huggingface_hub / bert_score 加载前生效（import mt_eval 在其后）
if os.environ.get("HF_USE_OFFICIAL_HUB", "").strip().lower() in ("1", "true", "yes"):
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
else:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def _merge_no_proxy_for_hf() -> None:
    """学术代理（如 AutoDL network_turbo）可能导致访问 hf-mirror/HF 走错误路径；常见域名走 NO_PROXY 直连。"""
    if os.environ.get("HF_SKIP_NO_PROXY_FIX", "").strip().lower() in ("1", "true", "yes"):
        return
    tail = "localhost,127.0.0.1,hf-mirror.com,huggingface.co,hf.co,xethub.hf.co"
    for key in ("NO_PROXY", "no_proxy"):
        cur = (os.environ.get(key) or "").strip()
        os.environ[key] = f"{tail},{cur}" if cur else tail


_merge_no_proxy_for_hf()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    from mt_eval import score_bertscore_f1, score_comet

    print(
        "说明：训练时 try/head 用 fr，swap 用 en（见各 config.eval_bertscore_lang）。\n",
        flush=True,
    )

    hyps = ["Hello world", "The cat sleeps"]
    refs = ["Hello world", "The dog runs"]
    srcs = ["Hallo Welt", "Die Katze schläft"]

    ok = 0
    # 默认 lang=en 会拉 roberta-large ~1.4GB，弱网像「卡住」；自检改用 DistilBERT ~250MB
    print(
        "=== BERTScore（自检：distilbert-base-uncased，约几百 MB；非训练默认）===",
        flush=True,
    )
    b = score_bertscore_f1(
        hyps,
        refs,
        lang="en",
        device="cpu",
        model_type="distilbert-base-uncased",
    )
    if b is None:
        print("BERTScore: 失败（见上方 stderr）")
    else:
        print(f"BERTScore F1 (mean): {b:.6f}", flush=True)
        ok += 1

    if os.environ.get("SKIP_COMET", "").strip() in ("1", "true", "yes"):
        print("=== COMET（已 SKIP_COMET=1 跳过；模型较大，可另测）===", flush=True)
    else:
        print(
            "=== COMET（wmt22-comet-da，体积较大，首次下载请耐心等待）===",
            flush=True,
        )
        c = score_comet(srcs, hyps, refs, model_name="Unbabel/wmt22-comet-da", batch_size=2)
        if c is None:
            print("COMET: 失败（见上方 stderr）")
        else:
            print(f"COMET score (mean): {c:.6f}", flush=True)
            ok += 1

    if ok == 0:
        print(
            "\n失败：请检查 HF 镜像/网络，或设置 HF_HOME；仅测 BERTScore 可暂时 SKIP_COMET=1。",
            flush=True,
        )
        return 1
    print(
        "\n至少一项成功。"
        "正式训练：英→法任务请保持 config 中 fr；法→英 swap 已为 en。"
        "若整体要换轻量 BERT，需为 fr 侧选好多语/法语模型（勿把仅英文的 distilbert 用于法语句）。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
