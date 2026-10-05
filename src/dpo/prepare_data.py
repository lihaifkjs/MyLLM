"""COIG-P 原始 parquet → 统一 dpo.jsonl schema → 采样切分 train/val。

契约（docs/07-DPO.md）：
- data/dpo.jsonl：{"instruction": ..., "chosen": ..., "rejected": ...}
  仅保留单轮样本（conversations 恰含 1 条 human 消息），多轮丢弃
- data/dpo_train.jsonl / data/dpo_val.jsonl：默认随机采样 3 万对，末尾 1% 作验证集
下载与采样切分复用 src/sft/prepare_data.py 的 download / sample_split。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.sft.prepare_data import download, sample_split

COIG_P_REPO = "m-a-p/COIG-P"
NUM_SHARDS = 9


def shard_url(endpoint: str, index: int) -> str:
    return (f"{endpoint}/datasets/{COIG_P_REPO}/resolve/main/"
            f"data/train-{index:05d}-of-{NUM_SHARDS:05d}.parquet")


def convert(parquet_paths: list[str], out_path: str) -> tuple[int, int]:
    """单轮样本 → 统一 schema，返回 (保留数, 丢弃数)。"""
    kept = dropped = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for path in parquet_paths:
            table = pq.read_table(path, columns=["conversations", "chosen", "rejected"])
            for conv, chosen, rejected in zip(
                    table.column("conversations").to_pylist(),
                    table.column("chosen").to_pylist(),
                    table.column("rejected").to_pylist()):
                # 多轮对话不适用我们的单轮模板，整对丢弃
                if len(conv) != 1 or conv[0]["from"] != "human":
                    dropped += 1
                    continue
                inst = conv[0]["value"].strip()
                c, r = chosen["value"].strip(), rejected["value"].strip()
                if not inst or not c or not r or c == r:
                    dropped += 1
                    continue
                fout.write(json.dumps({"instruction": inst, "chosen": c, "rejected": r},
                                      ensure_ascii=False) + "\n")
                kept += 1
    return kept, dropped


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="DPO 数据准备（模块6）")
    ap.add_argument("--raw-dir", default="data/raw", help="parquet 分片存放目录，缺失则自动下载")
    ap.add_argument("--full", default="data/dpo.jsonl")
    ap.add_argument("--train", default="data/dpo_train.jsonl")
    ap.add_argument("--val", default="data/dpo_val.jsonl")
    ap.add_argument("--num-shards", type=int, default=1, help="使用前几个分片（共 9 个，每片约 89MB / 11.2 万对）")
    ap.add_argument("--num-samples", type=int, default=30_000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="默认读 HF_ENDPOINT 环境变量（国内可设 https://hf-mirror.com）")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    paths = []
    for i in range(args.num_shards):
        p = raw_dir / f"coig-p-train-{i:05d}.parquet"
        if not p.exists():
            print(f"{p} 不存在，开始下载 COIG-P 分片 {i + 1}/{args.num_shards}（约 89MB）……")
            download(shard_url(args.endpoint, i), str(p))
        paths.append(str(p))

    kept, dropped = convert(paths, args.full)
    print(f"转换 {kept} 对（丢弃 {dropped}）→ {args.full}")
    if kept < args.num_samples:
        raise ValueError(f"单轮样本仅 {kept} 对，不足 --num-samples {args.num_samples}；"
                         f"请加大 --num-shards 或调小 --num-samples")

    n_train, n_val = sample_split(args.full, args.train, args.val,
                                  args.num_samples, args.val_ratio, args.seed)
    print(f"采样切分 train={n_train} / val={n_val}")


if __name__ == "__main__":
    main()
