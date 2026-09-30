"""冒烟测试：合成小样本（不联网）跑通 clean -> dedup -> build 管道并校验契约。

覆盖的噪声用例：HTML 标签、URL、精确重复、近似重复、过短段、英文占比过高段。
运行：python3 src/corpus/smoke_test.py（默认结束后自清理；加 --keep 保留产物供人工查看）
"""
import json
import subprocess
import sys
from pathlib import Path

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


def run(script):
    subprocess.run([sys.executable, f"src/corpus/{script}"], check=True)


GENERATED = [
    Path("data/raw/sample.jsonl"),
    Path("data/clean/cleaned.jsonl"),
    Path("data/clean/deduped.jsonl"),
    Path("data/clean/clean_stats.json"),
    Path("data/clean/dedup_stats.json"),
    Path("data/clean/report.json"),
    Path("data/corpus.txt"),
]


def main():
    keep = "--keep" in sys.argv  # 调试用：保留产物，默认自清理避免混入正式语料
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    try:
        with open(raw / "sample.jsonl", "w", encoding="utf-8") as f:
            for s in SAMPLES:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        for script in ["clean.py", "dedup.py", "build.py"]:
            run(script)

        lines = Path("data/corpus.txt").read_text(encoding="utf-8").splitlines()
        assert lines, "corpus.txt 为空"
        assert not any("<" in l and ">" in l for l in lines), "存在 HTML 残留"
        assert not any("http" in l for l in lines), "存在 URL 残留"
        assert len(lines) == len(set(lines)), "存在完全重复行"
        assert all(len(l) >= 50 for l in lines), "存在过短行"
        print(f"[OK] smoke test passed: {len(lines)} 段")
    finally:
        if not keep:
            for p in GENERATED:
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
