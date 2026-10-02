"""训练 Byte-level BPE 分词器。

用法：
    python -m src.tokenizer.train                          # 全量训练（契约默认）
    python -m src.tokenizer.train --corpus xxx --vocab-size 1000 --out-dir /tmp/tok

Byte-level BPE：初始符号为 256 个字节，逐步合并高频相邻对直至词表收敛。
任何字符都可由字节表示，因此不会 OOV，<unk> 仅作占位保留。
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train(
    corpus_path: str = "data/skypile_corpus.txt",
    out_dir: str = "tokenizer",
    vocab_size: int = 32000,
) -> Path:
    tokenizer = HFTokenizer(BPE())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # special_tokens 按给定顺序占用词表头部：<pad>=0, <unk>=1, <bos>=2, <eos>=3
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train([corpus_path], trainer)

    out_path = Path(out_dir) / "tokenizer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Byte-level BPE 分词器")
    parser.add_argument("--corpus", default="data/skypile_corpus.txt")
    parser.add_argument("--out-dir", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    out_path = train(args.corpus, args.out_dir, args.vocab_size)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
