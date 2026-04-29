#!/usr/bin/env python3
"""Dummy 句对：一次性自检 BLEU、chrF++、BERTScore（与 small 配置一致的 fr / en）、可选 COMET。

与 test_mt_eval_metrics.py 的区别：本脚本按「训练里会用的指标」各跑一遍 dummy；
BERTScore 使用轻量模型类型，避免 fr 默认拉超大 RoBERTa 类模型。

用法（AutoDL 可先开启学术加速）：
  cd /root/autodl-tmp && source /etc/network_turbo   # 可选
  python3 scripts/test_mt_eval_metrics_dummy.py

脚本在 import 任何 Hub 相关库之前默认强制 HF_ENDPOINT=hf-mirror（覆盖 shell 里可能存在的慢速值）。
走 HuggingFace 官网：HF_USE_OFFICIAL_HUB=1 python3 scripts/test_mt_eval_metrics_dummy.py
若 source /etc/network_turbo 后 BERTScore/COMET 全为 None，已自动为 Hub 域名追加 NO_PROXY；
仍失败可 HF_SKIP_NO_PROXY_FIX=1 或不用 network_turbo 再试。
可选：SKIP_COMET=1 跳过 COMET（已缓存时不必跳过）。
"""
from __future__ import annotations

import os
import sys

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


def _ok(name: str, value: object | None) -> bool:
    if value is None:
        print(f"  [FAIL] {name}: None", flush=True)
        return False
    print(f"  [ OK ] {name}: {value}", flush=True)
    return True


def main() -> int:
    from mt_eval import (
        score_comet,
        score_bertscore_f1,
        score_corpus_bleu,
        score_corpus_chrf,
        score_corpus_chrfpp,
    )

    # 目标语法语 dummy（对应 small_try / small_head：eval_bertscore_lang=fr）
    hyps_fr = ["Bonjour le monde.", "Le chat dort sur le tapis."]
    refs_fr = ["Bonjour le monde.", "Le chat dort sur le tapis."]

    # 目标英语 dummy（对应 small_swap：eval_bertscore_lang=en）
    hyps_en = ["Hello world.", "The cat sleeps on the mat."]
    refs_en = ["Hello world.", "The cat sleeps on the mat."]

    # COMET：英→法式三元组（与 try/head 方向一致）；swap 方向若需可另测
    srcs_en = ["Hello world.", "The cat sleeps on the mat."]
    comet_hyps = hyps_fr
    comet_refs = refs_fr

    fails: list[str] = []

    print("=== BLEU / chrF / chrF++（sacrebleu，无 HF 权重）===", flush=True)
    b = score_corpus_bleu(hyps_fr, refs_fr)
    c = score_corpus_chrf(hyps_fr, refs_fr)
    cpp = score_corpus_chrfpp(hyps_fr, refs_fr)
    if not _ok("corpus_bleu (fr dummy)", b):
        fails.append("bleu")
    if not _ok("corpus_chrf (word_order=0, fr dummy)", c):
        fails.append("chrf")
    if not _ok("corpus_chrfpp (word_order=2, fr dummy)", cpp):
        fails.append("chrfpp")

    # 轻量：多语 DistilBERT，覆盖 fr 语义；与「仅英文 distilbert」不同
    fr_model = "distilbert-base-multilingual-cased"
    print(
        f"\n=== BERTScore · lang=fr（对齐 small_try/small_head；model_type={fr_model}）===",
        flush=True,
    )
    print(
        "提示：首次需下载约 500MB+ 权重，加载前可能数十秒无新输出；出现 huggingface 进度条即正常。",
        flush=True,
    )
    bf = score_bertscore_f1(
        hyps_fr,
        refs_fr,
        lang="fr",
        device="cpu",
        model_type=fr_model,
    )
    if not _ok("bertscore_f1 fr", bf):
        fails.append("bertscore_fr")

    en_model = "distilbert-base-uncased"
    print(
        f"\n=== BERTScore · lang=en（对齐 small_swap；model_type={en_model}）===",
        flush=True,
    )
    print(
        "提示：若上一步已缓存多语模型，本段仍可能首次拉英文 DistilBERT（约 250MB+），请稍候。",
        flush=True,
    )
    be = score_bertscore_f1(
        hyps_en,
        refs_en,
        lang="en",
        device="cpu",
        model_type=en_model,
    )
    if not _ok("bertscore_f1 en", be):
        fails.append("bertscore_en")

    skip_comet = os.environ.get("SKIP_COMET", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if skip_comet:
        print("\n=== COMET（SKIP_COMET=1，跳过）===", flush=True)
    else:
        print(
            "\n=== COMET（Unbabel/wmt22-comet-da；首下大，已缓存则快）===",
            flush=True,
        )
        print(
            "提示：首次约 2.3GB；加载 PyTorch Lightning 时也可能短暂无输出。",
            flush=True,
        )
        cm = score_comet(
            srcs_en,
            comet_hyps,
            comet_refs,
            model_name="Unbabel/wmt22-comet-da",
            batch_size=2,
        )
        if not _ok("comet", cm):
            fails.append("comet")

    print("", flush=True)
    if fails:
        print(
            "部分失败：" + ", ".join(fails) + "。",
            flush=True,
        )
        print(
            "若 stderr 有「未安装 bert-score / unbabel-comet」，请先："
            "pip install bert-score unbabel-comet（或与 small_try/requirements.txt 一致）。"
            "其余情况再查 HF 缓存与网络。",
            flush=True,
        )
        return 1
    print(
        "全部通过（dummy）。正式训练仍用各 config 的 eval_bertscore_lang；"
        "若未设 BERTSCORE_MODEL_TYPE，bert-score 会按 lang 选默认大模型。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
