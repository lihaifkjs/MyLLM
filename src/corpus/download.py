"""下载采样：从镜像站直链拉取 SkyPile-150B 部分分片，累计字节达标即停。

设计要点（见 docs/02-语料处理.md）：
- 镜像优先：默认走 hf-mirror.com 的 resolve 直链（huggingface_hub 1.8 在本机不可用，
  故不依赖该库，直接 HTTP 下载），可用 --endpoint 覆盖
- 断点续传：分片内用 .part + Range 头续传；分片级进度记录在 download_state.json
- 可断点续跑：已完成并抽取的分片记录在 state["done"]，重跑自动跳过
"""
import argparse
import json
import time
from pathlib import Path

import httpx

DEFAULT_ENDPOINT = "https://hf-mirror.com"
REPO_ID = "Skywork/SkyPile-150B"
TARGET_BYTES = 5 * 1024**3  # 约 5GB
RAW_DIR = Path("data/raw")
STATE_FILE = RAW_DIR / "download_state.json"
OUT_FILE = RAW_DIR / "sample.jsonl"
MAX_RETRY = 3
CHUNK = 1 << 20  # 1MB


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"done": [], "bytes": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def list_shards(endpoint: str, repo_id: str) -> list[str]:
    """列出仓库内全部数据分片路径（如 data/2020-40_zh_head_0000.jsonl）。"""
    url = f"{endpoint}/api/datasets/{repo_id}/tree/main?recursive=true"
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    files = r.json()
    return [f["path"] for f in files
            if f.get("type") == "file" and f["path"].endswith((".jsonl", ".jsonl.gz", ".json"))]


def fetch_file(url: str, dest: Path) -> None:
    """流式下载到 dest，中途失败留下 .part，下次以 Range 头续传。"""
    part = dest.with_suffix(dest.suffix + ".part")
    downloaded = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        # 服务器忽略 Range 返回 200（全量）时，必须从头覆盖写，否则文件损坏
        mode = "ab" if downloaded and r.status_code == 206 else "wb"
        with open(part, mode) as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
    part.rename(dest)


def download_shard(shards, state, target_bytes, endpoint, repo_id):
    """逐个分片下载（带重试），直到累计字节达标。返回下载到位的分片路径列表。"""
    done = set(state["done"])
    for name in shards:
        if state["bytes"] >= target_bytes:
            break
        if name in done:
            continue
        dest = RAW_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/datasets/{repo_id}/resolve/main/{name}"
        ok = False
        for attempt in range(1, MAX_RETRY + 1):
            try:
                fetch_file(url, dest)
                ok = True
                break
            except Exception as e:  # 网络错误：指数退避后重试，.part 保留供续传
                print(f"[retry {attempt}/{MAX_RETRY}] {name}: {e}")
                time.sleep(2**attempt)
        if not ok:
            print(f"[skip] {name} 下载失败，保留进度，可稍后重跑续传")
            break
        state["bytes"] += dest.stat().st_size
        state["done"].append(name)
        save_state(state)
        yield dest


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

    shards = list_shards(args.endpoint, REPO_ID)
    print(f"发现 {len(shards)} 个分片，目标 {args.target_bytes / 1024**3:.1f}GB")
    paths = list(download_shard(shards, state, args.target_bytes, args.endpoint, REPO_ID))
    n = extract_text(paths, OUT_FILE)
    print(f"done: 累计 {state['bytes'] / 1024**3:.2f}GB，本次抽取 {n} 篇 -> {OUT_FILE}")


if __name__ == "__main__":
    main()
