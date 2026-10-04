"""训练 BPE 分词器（sentencepiece）。

用法：
    python -m src.tokenizer.train                          # 全量训练（契约默认）
    python -m src.tokenizer.train --corpus xxx --vocab-size 1000 --out-dir /tmp/tok

选型说明：HF tokenizers 的 BPE 训练器在 GB 级中文语料上合并阶段过慢
（实测 ~10min/merge，32000 次合并不可接受），故改用 sentencepiece——
其 BPE 训练器专为大语料设计，中文 32k 词表训练是业界标配路径。
byte_fallback=True 保证任意字符可由字节回退表示，不会 OOV，<unk> 仅作占位。
"""

import argparse
import time
from pathlib import Path

import sentencepiece as spm

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train(
    corpus_path: str = "data/skypile_corpus.txt",
    out_dir: str = "tokenizer",
    vocab_size: int = 32000,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 特殊 token id 固定：<pad>=0, <unk>=1, <bos>=2, <eos>=3（下游硬编码安全）
    _log(f"start training: corpus={corpus_path} vocab_size={vocab_size}")
    start = time.time()
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=str(out / "tokenizer"),
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        # identity：不做归一化。默认 nmt_nfkc 会把全角标点（，）折叠成半角（,），
        # 破坏 roundtrip 保真，且会让模型生成半角标点的中文
        normalization_rule_name="identity",
        # 小样本冒烟测试时语料可能凑不满词表，不强制硬限制；全量训练不受影响
        hard_vocab_limit=False,
    )
    _log(f"training done in {time.time() - start:.0f}s")

    model_path = out / "tokenizer.model"
    _log(f"saved: {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 BPE 分词器（sentencepiece）")
    parser.add_argument("--corpus", default="data/skypile_corpus.txt")
    parser.add_argument("--out-dir", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    train(args.corpus, args.out_dir, args.vocab_size)


if __name__ == "__main__":
    main()
