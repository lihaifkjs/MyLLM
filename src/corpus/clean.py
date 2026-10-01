"""规则清洗：data/raw/sample.jsonl -> data/clean/cleaned.jsonl

每篇文档处理为若干"段"（段即最终 corpus.txt 的一行）：
1. 去 HTML 标签 / URL / 邮箱
2. 去不可见控制字符、压缩空白（含全角空格）；保留原始中文标点（NFKC 会把，。转成半角，弃用）
3. 按换行与句末标点（。！？!?；;）拆段
4. 过滤：段长 <50 或 >10000 丢弃；中文字符占比 <30% 丢弃
各规则丢弃量写入 data/clean/clean_stats.json 供 build.py 汇总。
"""
import argparse
import json
import re
from pathlib import Path

MIN_LEN, MAX_LEN = 50, 10000
MIN_ZH_RATIO = 0.30

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SPACES = re.compile(r"[ \t　]+")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CJK = re.compile(r"[一-鿿]")
# 先按换行切块，块内再于句末标点后断开，保证长段落也能拆成合理行
BLOCK_SPLIT = re.compile(r"\n+")
SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")


def normalize(text: str) -> str:
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = CONTROL.sub("", text)
    return SPACES.sub(" ", text)


def split_paragraphs(text: str) -> list[str]:
    paras = []
    for block in BLOCK_SPLIT.split(text):
        for seg in SENT_SPLIT.split(block):
            seg = seg.strip()
            if seg:
                paras.append(seg)
    return paras


def zh_ratio(s: str) -> float:
    return len(CJK.findall(s)) / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw/skypile_sample.jsonl")
    ap.add_argument("--out", default="data/clean/skypile_cleaned.jsonl")
    args = ap.parse_args()

    stats = {"docs": 0, "kept": 0, "drop_too_short": 0, "drop_too_long": 0, "drop_low_zh": 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            stats["docs"] += 1
            text = normalize(json.loads(line)["text"])
            for para in split_paragraphs(text):
                if len(para) < MIN_LEN:
                    stats["drop_too_short"] += 1
                elif len(para) > MAX_LEN:
                    stats["drop_too_long"] += 1
                elif zh_ratio(para) < MIN_ZH_RATIO:
                    stats["drop_low_zh"] += 1
                else:
                    fout.write(json.dumps({"text": para}, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
    stats_path = Path(args.out).with_suffix(".stats.json")  # 与前缀同名，避免跨数据集覆盖
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
