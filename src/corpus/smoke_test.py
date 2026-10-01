"""冒烟测试：合成小样本（不联网）跑通 clean -> dedup -> build 管道并校验契约。

覆盖的噪声用例：HTML 标签、URL、精确重复、近似重复、过短段、英文占比过高段。
所有产物使用 smoke_ 前缀，与正式语料（skypile_）完全隔离。
运行：python3 src/corpus/smoke_test.py（默认保留产物供人工查看；加 --clean 跑完删除）
"""
import json
import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw/smoke_sample.jsonl")
CLEANED = Path("data/clean/smoke_cleaned.jsonl")
DEDUPED = Path("data/clean/smoke_deduped.jsonl")
CORPUS = Path("data/smoke_corpus.txt")
GENERATED = [RAW, CLEANED, DEDUPED, CORPUS,
             CLEANED.with_suffix(".stats.json"), DEDUPED.with_suffix(".stats.json"),
             CORPUS.with_suffix(".report.json")]

BASE = ("人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出能以人类智能相似方式反应的智能机器，"
        "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。")
SAMPLES = [
    {"text": f"<p>{BASE}</p>详见 https://example.com/news 报道。"},   # HTML + URL
    {"text": BASE},                                                   # 正常段
    {"text": BASE},                                                   # 精确重复
    {"text": BASE.replace("计算机科学", "计算机学科")},                 # 近似重复
    {"text": "太短了。"},                                             # 过短
    {"text": "This is an English paragraph with only a few 中文 words mixed in, should be dropped."},
    {"text": "第一段内容需要足够长才行，必须包含足够多的中文字符，才能顺利通过长度过滤规则设定的五十字符下限检查。\n第二段同样需要足够长才可以，主要用来验证按换行拆段的逻辑，是否能够正常地把一篇文档拆分成两个独立段落。"},
]


def run(script, *args):
    subprocess.run([sys.executable, f"src/corpus/{script}", *args], check=True)


def main():
    clean_up = "--clean" in sys.argv
    try:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW, "w", encoding="utf-8") as f:
            for s in SAMPLES:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        run("clean.py", "--inp", str(RAW), "--out", str(CLEANED))
        run("dedup.py", "--inp", str(CLEANED), "--out", str(DEDUPED))
        run("build.py", "--inp", str(DEDUPED), "--out", str(CORPUS))

        lines = CORPUS.read_text(encoding="utf-8").splitlines()
        assert lines, "corpus 为空"
        assert not any("<" in l and ">" in l for l in lines), "存在 HTML 残留"
        assert not any("http" in l for l in lines), "存在 URL 残留"
        assert len(lines) == len(set(lines)), "存在完全重复行"
        assert all(len(l) >= 50 for l in lines), "存在过短行"
        print(f"[OK] smoke test passed: {len(lines)} 段，产物保留在 data/（smoke_ 前缀）")
    finally:
        if clean_up:
            for p in GENERATED:
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
