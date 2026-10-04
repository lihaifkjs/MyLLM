"""离线编码：语料 txt → uint16 memmap（tokens.bin + tokens.meta.json）。

每行（一段）encode 后拼接 eos_id 作为段边界，全量写成原始 uint16 二进制，
训练侧用 numpy memmap 零拷贝读取。词表 32k < 65536，uint16 无溢出。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.tokenizer import Tokenizer


def encode_corpus(corpus_path: str, out_path: str,
                  tokenizer_path: str = "tokenizer/tokenizer.model",
                  chunk_lines: int = 10000) -> int:
    """返回写入的 token 总数。"""
    tok = Tokenizer(tokenizer_path)
    if tok.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(f"词表 {tok.vocab_size} 超出 uint16 表示范围")

    chunks: list[np.ndarray] = []
    num_tokens = 0
    with open(corpus_path, encoding="utf-8") as f:
        buf: list[str] = []
        for line in tqdm(f, desc="encode", unit="行"):
            line = line.strip()
            if line:
                buf.append(line)
            if len(buf) >= chunk_lines:
                num_tokens += _flush(tok, buf, chunks)
                buf.clear()
        num_tokens += _flush(tok, buf, chunks)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).tofile(out)
    meta = {"num_tokens": num_tokens, "dtype": "uint16", "eos_id": tok.eos_id}
    _meta_path(out).write_text(json.dumps(meta), encoding="utf-8")
    return num_tokens


def _flush(tok: Tokenizer, lines: list[str], chunks: list[np.ndarray]) -> int:
    if not lines:
        return 0
    ids: list[int] = []
    for line in lines:
        ids.extend(tok.encode(line))
        ids.append(tok.eos_id)
    arr = np.asarray(ids, dtype=np.uint16)
    chunks.append(arr)
    return arr.size


def _meta_path(bin_path: Path) -> Path:
    return bin_path.with_suffix(".meta.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="语料离线编码为 tokens.bin")
    ap.add_argument("--corpus", default="data/skypile_corpus.txt")
    ap.add_argument("--out", default="data/clean/tokens.bin")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    args = ap.parse_args()
    n = encode_corpus(args.corpus, args.out, args.tokenizer)
    print(f"完成：{n:,} tokens → {args.out}")


if __name__ == "__main__":
    main()
