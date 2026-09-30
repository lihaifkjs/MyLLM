"""去重：data/clean/cleaned.jsonl -> data/clean/deduped.jsonl

两级去重：
1. 精确去重：整段哈希，O(n) 去掉完全重复的段
2. MinHash 近似去重：
   - 把每段切成字符 3-gram 集合（shingles），两段重复度用 Jaccard 相似度衡量
   - MinHash 用 128 个哈希函数把大集合压缩成签名，签名冲突率即 Jaccard 的无偏估计
   - LSH 把相似签名分到同一桶，避免 O(n^2) 两两比较；阈值 0.8
"""
import argparse
import hashlib
import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
THRESHOLD = 0.8
SHINGLE_N = 3


def shingles(text: str, n: int = SHINGLE_N):
    return {text[i:i + n] for i in range(len(text) - n + 1)} or {text}


def minhash_of(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/clean/cleaned.jsonl")
    ap.add_argument("--out", default="data/clean/deduped.jsonl")
    args = ap.parse_args()

    stats = {"in": 0, "kept": 0, "drop_exact": 0, "drop_near": 0}
    seen = set()
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    kept_lines = []

    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            stats["in"] += 1
            text = json.loads(line)["text"]
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen:
                stats["drop_exact"] += 1
                continue
            m = minhash_of(text)
            if lsh.query(m):  # 与已有段近似重复
                stats["drop_near"] += 1
                continue
            seen.add(h)
            lsh.insert(h, m)
            kept_lines.append(line)
            stats["kept"] += 1

    Path(args.out).write_text("".join(kept_lines), encoding="utf-8")
    stats_path = Path(args.out).parent / "dedup_stats.json"  # 跟随输出目录，便于测试隔离
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
