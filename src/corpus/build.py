"""汇总输出：data/clean/deduped.jsonl -> data/corpus.txt（一行一段，UTF-8）

并输出统计报告 data/clean/report.json：
- 行数、总字符数
- token 粗估：32k 中文 BPE 下单字约 0.6~1 token，取总字符数为上界
- 汇总 clean/dedup 两阶段的丢弃统计
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/clean/skypile_deduped.jsonl")
    ap.add_argument("--out", default="data/skypile_corpus.txt")
    args = ap.parse_args()

    lines = Path(args.inp).read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines]
    total_chars = sum(len(t) for t in texts)

    with open(args.out, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    report = {
        "lines": len(texts),
        "total_chars": total_chars,
        "token_estimate_upper": total_chars,
        "clean_stats": _load(clean_stats_path(Path(args.inp))),
        "dedup_stats": _load(Path(args.inp).with_suffix(".stats.json")),
    }
    report_path = Path(args.out).with_suffix(".report.json")  # 与前缀同名
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def clean_stats_path(inp: Path) -> Path:
    """按命名约定 <tag>_deduped.jsonl -> <tag>_cleaned.stats.json 推导。"""
    tag = inp.stem.removesuffix("_deduped")
    return inp.with_name(f"{tag}_cleaned.stats.json")


def _load(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


if __name__ == "__main__":
    main()
