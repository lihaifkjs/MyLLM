"""Firefly 原始数据 → 统一 sft.jsonl schema → 采样切分 train/val。

契约（docs/06-SFT.md）：
- data/sft.jsonl：{"instruction": ..., "output": ...}（instruction←input，output←target）
- data/sft_train.jsonl / data/sft_val.jsonl：默认随机采样 10 万条，末尾 1% 作验证集
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import httpx

FIREFLY_URL = ("https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M"
               "/resolve/main/firefly-train-1.1M.jsonl")


def download(url: str, out_path: str) -> None:
    """流式下载到 .part 临时文件，完成后改名，避免半截文件被当成完整数据。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r下载中 {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    tmp.rename(out)
    print(f"\n下载完成：{out}")


def convert(raw_path: str, out_path: str) -> int:
    """Firefly 原始 schema（kind/input/target）→ 统一 schema（instruction/output），丢弃空字段。
    json.dumps(ensure_ascii=False) 会把 U+2028/U+2029（Unicode 行分隔符）原样写出，
    下游按行读取时记录会被切断，故显式转义，保证严格一行一条。
    """
    n = 0
    with open(raw_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            inst, out = r["input"].strip(), r["target"].strip()
            if not inst or not out:
                continue
            row = json.dumps({"instruction": inst, "output": out}, ensure_ascii=False)
            fout.write(row.replace(chr(0x2028), "\\u2028").replace(chr(0x2029), "\\u2029") + "\n")
            n += 1
    return n


def sample_split(full_path: str, train_path: str, val_path: str,
                 num_samples: int, val_ratio: float, seed: int) -> tuple[int, int]:
    # 严格按 \n 切行：splitlines() 会把 U+2028 等也当换行，可能切断 JSON 记录
    with open(full_path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if 0 < num_samples < len(lines):
        lines = random.Random(seed).sample(lines, num_samples)
    n_val = max(1, int(len(lines) * val_ratio))
    Path(train_path).write_text("\n".join(lines[:-n_val]) + "\n", encoding="utf-8")
    Path(val_path).write_text("\n".join(lines[-n_val:]) + "\n", encoding="utf-8")
    return len(lines) - n_val, n_val


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SFT 数据准备（模块5）")
    ap.add_argument("--raw", default="data/raw/firefly-train-1.1M.jsonl", help="Firefly 原始文件，缺失则自动下载")
    ap.add_argument("--full", default="data/sft.jsonl")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft_val.jsonl")
    ap.add_argument("--num-samples", type=int, default=100_000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not Path(args.raw).exists():
        print(f"{args.raw} 不存在，开始下载 Firefly（约 1.6GB）……")
        download(FIREFLY_URL, args.raw)

    n = convert(args.raw, args.full)
    n_train, n_val = sample_split(args.full, args.train, args.val,
                                  args.num_samples, args.val_ratio, args.seed)
    print(f"转换 {n} 条 → {args.full}；采样切分 train={n_train} / val={n_val}")


if __name__ == "__main__":
    main()
