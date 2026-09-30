"""下载采样：从 HuggingFace 拉取 Skypile-150B 部分分片，累计字节达标即停。

设计要点（见 docs/02-语料处理.md）：
- 镜像优先：默认 HF_ENDPOINT=hf-mirror.com，可用 --endpoint 覆盖
- 断点续传：hf_hub_download 自带 .incomplete 续传；分片级进度记录在 download_state.json
- 可断点续跑：已完成并抽取的分片记录在 state["done"]，重跑自动跳过
"""
import argparse
import json
import os
import time
from pathlib import Path

DEFAULT_ENDPOINT = "https://hf-mirror.com"
REPO_ID = "Skywork/Skywork-150B"
TARGET_BYTES = 5 * 1024**3  # 约 5GB
RAW_DIR = Path("data/raw")
STATE_FILE = RAW_DIR / "download_state.json"
OUT_FILE = RAW_DIR / "sample.jsonl"
MAX_RETRY = 3


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"done": [], "bytes": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def download_shard(repo_files, state, target_bytes):
    """逐个分片下载（带重试），直到累计字节达标。返回下载到位的分片路径列表。"""
    from huggingface_hub import hf_hub_download

    shards = [f for f in repo_files if f.endswith((".jsonl", ".jsonl.gz", ".json"))]
    done = set(state["done"])
    for name in shards:
        if state["bytes"] >= target_bytes:
            break
        if name in done:
            continue
        path = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                path = hf_hub_download(REPO_ID, name, repo_type="dataset")
                break
            except Exception as e:  # 网络错误：指数退避后重试
                print(f"[retry {attempt}/{MAX_RETRY}] {name}: {e}")
                time.sleep(2**attempt)
        if path is None:
            print(f"[skip] {name} 下载失败，保留进度，可稍后重跑续传")
            break
        state["bytes"] += os.path.getsize(path)
        state["done"].append(name)
        save_state(state)
        yield Path(path)


def extract_text(shard_paths, out_file):
    """统一抽取正文字段，转存为 {"text": ...} 的 JSONL。"""
    n = 0
    with open(out_file, "a", encoding="utf-8") as out:
        for path in shard_paths:
            opener = open
            if str(path).endswith(".gz"):
                import gzip
                opener = gzip.open
            with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text") or obj.get("content")
                    if text:
                        out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--target-bytes", type=int, default=TARGET_BYTES)
    ap.add_argument("--append", action="store_true",
                    help="允许向已存在的 sample.jsonl 追加（续跑场景）")
    args = ap.parse_args()

    state = load_state()
    if OUT_FILE.exists() and not state["done"] and not args.append:
        raise SystemExit(
            f"{OUT_FILE} 已存在且无续跑记录，追加会混入旧数据；"
            "请先删除该文件，或确认后加 --append")

    os.environ["HF_ENDPOINT"] = args.endpoint  # 须在 import huggingface_hub 前设置
    from huggingface_hub import HfApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repo_files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    paths = list(download_shard(repo_files, state, args.target_bytes))
    n = extract_text(paths, OUT_FILE)
    print(f"done: 累计 {state['bytes'] / 1024**3:.2f}GB，本次抽取 {n} 篇 -> {OUT_FILE}")


if __name__ == "__main__":
    main()
