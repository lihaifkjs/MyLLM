"""去重：data/clean/cleaned.jsonl -> data/clean/deduped.jsonl

两级去重：
1. 精确去重：整段 md5，O(n) 去掉完全重复的段
2. MinHash 近似去重：
   - 把每段切成字符 3-gram 集合（shingles），两段重复度用 Jaccard 相似度衡量
   - MinHash 用 64 个哈希函数把大集合压缩成签名，签名冲突率即 Jaccard 的无偏估计
   - LSH 把相似签名分到同一桶，避免 O(n^2) 两两比较；阈值 0.8

性能架构（千万级段必须并行）：
- 签名计算是 CPU 密集大头（每段 ~字数×64 次哈希），用 multiprocessing.Pool 分发到多核，
  imap 保序返回，主进程单线程做判定与写盘，无锁无共享状态
- LSH 索引不可跨进程共享，查询/插入只在主进程（本身很快）
- 判重通过的段立即流式写出，不做内存累积
"""
import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

from datasketch import MinHash, MinHashLSH

NUM_PERM = 64
THRESHOLD = 0.8
SHINGLE_N = 3
CHUNK_SIZE = 500        # imap 批量派发粒度
PROGRESS_EVERY = 100_000


def signature(line: str):
    """worker：计算一行的 (原文, md5, MinHash)。必须是顶层函数才能被 pickle 到子进程。"""
    text = json.loads(line)["text"]
    m = MinHash(num_perm=NUM_PERM)
    if len(text) >= SHINGLE_N:
        for i in range(len(text) - SHINGLE_N + 1):
            m.update(text[i:i + SHINGLE_N].encode("utf-8"))
    else:
        m.update(text.encode("utf-8"))
    return line, hashlib.md5(text.encode("utf-8")).hexdigest(), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/clean/skypile_cleaned.jsonl")
    ap.add_argument("--out", default="data/clean/skypile_deduped.jsonl")
    args = ap.parse_args()

    stats = {"in": 0, "kept": 0, "drop_exact": 0, "drop_near": 0}
    seen = set()
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    start = time.time()

    with open(args.inp, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout, \
         Pool() as pool:
        for line, h, m in pool.imap(signature, fin, chunksize=CHUNK_SIZE):
            stats["in"] += 1
            if h in seen:
                stats["drop_exact"] += 1
            elif lsh.query(m):  # 与已有段近似重复
                stats["drop_near"] += 1
            else:
                seen.add(h)
                lsh.insert(h, m)
                fout.write(line)
                stats["kept"] += 1
            if stats["in"] % PROGRESS_EVERY == 0:
                rate = stats["in"] / (time.time() - start)
                print(f"[progress] {stats['in']} 段, {rate:.0f} 段/s, kept={stats['kept']}",
                      flush=True)

    stats_path = Path(args.out).with_suffix(".stats.json")  # 与前缀同名，避免跨数据集覆盖
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
