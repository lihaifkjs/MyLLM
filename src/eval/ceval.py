"""C-Eval 子集评测（模块7，docs/08）。

- 数据源：HuggingFace ceval/ceval-exam（parquet），仅 val split（答案公开）
- 规模：val 全量约 1346 题，默认随机采 800 题（seed=42）
- 打分：0-shot prompt 一次前向，比较末位 logits 中 "A"/"B"/"C"/"D" 四个候选 token
  的 logprob（比较 logprob 而非生成，免除解码随机性）
- 下载复用 src/sft/prepare_data.download（httpx 流式），文件清单走 HF API 的 tree 接口
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import torch

from src.model import LlamaModel, ModelConfig
from src.sft.prepare_data import download
from src.tokenizer.tokenizer import Tokenizer

CEVAL_REPO = "ceval/ceval-exam"
LETTERS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = ("以下是中国关于{subject}的考试题目。请选出其中的正确答案。\n"
                   "{question}\n"
                   "A. {A}\nB. {B}\nC. {C}\nD. {D}\n答案：")


def build_prompt(subject: str, q: dict) -> str:
    return PROMPT_TEMPLATE.format(subject=subject, question=q["question"],
                                  A=q["A"], B=q["B"], C=q["C"], D=q["D"])


def list_val_files(endpoint: str) -> list[tuple[str, str]]:
    """HF API tree 接口列出全部科目的 val parquet（返回 [(subject, path)]，支持 hf-mirror）。

    仓库结构：ceval-exam 下每科目一个目录（如 high_school_physics/），
    内含 dev/val/test 三个 parquet，val split 答案公开。
    """
    base = f"{endpoint}/api/datasets/{CEVAL_REPO}/tree/main"
    r = httpx.get(base, follow_redirects=True, timeout=60)
    r.raise_for_status()
    subjects = [item["path"] for item in r.json() if item["type"] == "directory"]
    val_files = []
    for subject in subjects:
        r = httpx.get(f"{base}/{subject}", follow_redirects=True, timeout=60)
        r.raise_for_status()
        for item in r.json():
            name = Path(item["path"]).name
            if item["type"] == "file" and name.startswith("val-") and name.endswith(".parquet"):
                val_files.append((subject, item["path"]))
    if not val_files:
        raise RuntimeError(f"{base} 下未找到任何科目的 val parquet，请确认仓库结构未变更")
    return val_files


def load_questions(raw_dir: str, endpoint: str) -> list[dict]:
    """下载（缺失时）并读取全部 val parquet → [{subject, question, A..D, answer}]。"""
    raw = Path(raw_dir)
    questions = []
    for subject, path in list_val_files(endpoint):
        local = raw / f"{subject}_val.parquet"
        if not local.exists():
            print(f"{local} 不存在，开始下载 {path} ……")
            download(f"{endpoint}/datasets/{CEVAL_REPO}/resolve/main/{path}", str(local))
        table = pq.read_table(local)
        cols = {name: table.column(name).to_pylist() for name in
                ("question", "A", "B", "C", "D", "answer")}
        for i in range(table.num_rows):
            questions.append({"subject": subject, "question": cols["question"][i],
                              "A": cols["A"][i], "B": cols["B"][i],
                              "C": cols["C"][i], "D": cols["D"][i],
                              "answer": cols["answer"][i]})
    return questions


def letter_ids(tok: Tokenizer) -> dict[str, int]:
    """校验 "A"~"D" 均为单 token（byte_fallback 词表应满足），返回 字母→token id。"""
    ids = {}
    for letter in LETTERS:
        enc = tok.encode(letter)
        if len(enc) != 1:
            raise ValueError(f"字母 {letter} 被编为 {len(enc)} 个 token，无法用于 logprob 打分")
        ids[letter] = enc[0]
    return ids


@torch.no_grad()
def score_questions(model: LlamaModel, tok: Tokenizer, questions: list[dict],
                    device: torch.device) -> list[dict]:
    """逐题前向，用末位 logits 给四个候选字母打分，返回每题 {pred, correct}。"""
    lids = letter_ids(tok)
    results = []
    for q in questions:
        ids = tok.encode(build_prompt(q["subject"], q), add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        pred = max(LETTERS, key=lambda l: logits[lids[l]].item())
        results.append({"pred": pred, "correct": pred == q["answer"]})
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="C-Eval val 子集评测（模块7）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--num-questions", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", default="data/raw/ceval")
    ap.add_argument("--log", default="logs/ceval_report.jsonl")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                    help="默认读 HF_ENDPOINT 环境变量（国内可设 https://hf-mirror.com）")
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    args = ap.parse_args(argv)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    questions = load_questions(args.raw_dir, args.endpoint)
    if len(questions) < args.num_questions:
        raise ValueError(f"val 题目仅 {len(questions)} 道，不足 --num-questions {args.num_questions}")
    questions = random.Random(args.seed).sample(questions, args.num_questions)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LlamaModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer(args.tokenizer)

    results = score_questions(model, tok, questions, device)
    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)
    per_subject: dict[str, list[int]] = {}
    for q, r in zip(questions, results):
        per_subject.setdefault(q["subject"], []).append(int(r["correct"]))
    subject_acc = {s: sum(v) / len(v) for s, v in per_subject.items()}

    print(f"总准确率：{accuracy:.4f}（{n_correct}/{len(results)}，随机线 0.25）")
    ranked = sorted(subject_acc.items(), key=lambda kv: kv[1], reverse=True)
    print("学科前 5：" + "，".join(f"{s} {a:.2f}" for s, a in ranked[:5]))
    print("学科后 5：" + "，".join(f"{s} {a:.2f}" for s, a in ranked[-5:]))

    record = {"ckpt": args.ckpt, "benchmark": "ceval-val", "n": len(results),
              "accuracy": round(accuracy, 4), "per_subject": subject_acc,
              "date": datetime.datetime.now().isoformat(timespec="seconds")}
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已追加到 {args.log}")


if __name__ == "__main__":
    main()
